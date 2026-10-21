# Plan

Three phases: (1) a serverless depth endpoint (not started), (2) running depth
locally on a 16 GB GPU (done: the cu128 image and `run-local.sh`), (3) Cloudflare
R2 as a model source for pods and new volumes (tooling proven, not in the image yet).

# Phase 1: serverless depth endpoint

Status: not started. Goal: run the Marigold depth step as a RunPod serverless
endpoint, reusing the existing model volumes, so there is no pod to find, start
or leave idle. The vathopy GUI stays local (it already reads the depth folder).

## What stays the same

- The ComfyUI image, custom nodes and the Marigold workflows (`workflows/`).
- The model volumes and their `models/` layout. Nothing is re-downloaded.
- Output naming: `<photo>_<prediction>_00001.exr`, so local `vathopy gui` picks
  the results up from the depth folder.

## What changes

- **Handler.** Wrap the workflow in a serverless handler, based on
  [worker-comfyui](https://github.com/runpod-workers/worker-comfyui).
  Input: a photo plus options (prediction type, long-edge pixels, default
  2496). Output: the S3 key(s) of the EXR(s).
- **Volume path.** Serverless mounts the volume at `/runpod-volume`, not
  `/workspace`. Point ComfyUI's model paths there (`extra_model_paths.yaml`
  or an env var `start.sh` already reads).
- **Input transfer.** Photos at 2496 px long edge fit in the job payload as
  base64. Larger ones (panoramas) go to the volume via its S3 gateway first,
  and the job gets the key.
- **Output transfer.** A 2496 px float EXR is ~16 MB or more, over the job
  result limit. RunPod drops oversized `output` **without an error** (the job
  says COMPLETED with nothing). So the worker writes EXRs to `outputs/` on the
  volume and returns keys; the client fetches them with `aws s3api get-object`
  (the gateway rejects HeadObject, so not `aws s3 cp`).
- **Client.** `submit.sh` / `submit.py` replaces `batch.sh`: sends photos,
  polls jobs, downloads the EXRs into the depth folder, checks sizes, never
  overwrites (same rules as `pull.sh`).

## Steps

1. Build `serverless/`: handler, Dockerfile (or a build arg on the existing
   one), client. Image tag, not `:latest`.
2. Create a template and an endpoint on one seeded volume (US-NE-1 or
   US-MO-1). Max workers 1, idle timeout ~5 min, FlashBoot on,
   `allowedCudaVersions: ["13.0"]`. GPU pools: 24 GB and 48 GB Ada/Ampere,
   RTX PRO 6000 (incl. MIG slices) — whatever has stock.
3. End-to-end test with one photo: compare the EXR with a pod-made one of the
   same photo and size.
4. Check whether an endpoint can attach volumes in several datacenters; if so,
   add US-MO-1, US-NE-1 and US-IL-1 so any of them can serve.
5. Document in README/AGENTS.md (generic; no account, endpoint or volume IDs).

## Risks / open questions

- **CUDA 13 hosts are scarce.** The image's cu130 torch needs driver >= 580;
  serverless draws on the same hosts. Deploying and watching workers start is
  the only real test. Use the cu128 image (build.md) to widen the pool.
- **Cold start.** Each new worker loads the 20.5 GB backbone from the network
  volume (~2 min). Workers without a volume could fetch the models from R2 at boot
  (Phase 3), but that adds 2-8 min to every cold start: keep volumes for serverless. Idle timeout and FlashBoot keep a worker warm across a
  batch; a single photo after a quiet spell pays the full cold start.
- **Volume and endpoint must share a datacenter**, or the volume won't mount.
- **S3 region must be lowercase** (`us-ne-1`) and set explicitly
  (`AWS_REGION` / `AWS_DEFAULT_REGION`), or the gateway rejects the request.
- **Template `env` updates replace the whole list**: always pass every var.
- **Cost.** Serverless per-second rates are higher than pods: good for
  occasional batches, worse for long sessions (keep `create-pod.sh` for those).
- **Output cleanup.** EXRs accumulate in `outputs/` on a 30 GB volume; the
  client should delete them after a verified download (as `pull.sh` can).

# Phase 2: local depth on a 16 GB GPU

Status: **answered (2026-10-20), no GGUF needed.** The int8 backbone runs on a
cu128 build (comfy-kitchen's eager backend) and ComfyUI's dynamic VRAM loading
keeps a 16 GB card under its limit by holding part of the model in system RAM.
Measured on an RTX 4060 Ti 16 GB, in the cu128 image with `--gpus all`:

| Long edge | Time |
|---|---|
| 1280 px | 7 s (warm) |
| 2496 px | 24 s |
| 3488 px | 54 s |

Depth vs a pod-made map of the same photo: r = 0.9989, mean difference 0.7% of
the depth range, about what resizing the input differently changes. Fringe
renders look the same.

Done: `run-local.sh` and the README section "Running locally on your own GPU" (mounted
photo and depth folders, PUID/PGID ownership, a config file, other ports). GGUF quants (e.g. `QuantStack/Qwen-Image-Edit-2509-GGUF`)
remain an option for cards under 16 GB or machines with little RAM.

# Phase 3: Cloudflare R2 as a model source

Status: **the R2 side is proven, not yet in the image.** The models are in a
private R2 bucket, and separate tooling (outside this repo) fills a RunPod
network volume from it with a throwaway pod. A new US-IL-1 volume filled that
way in 112 s, every file matching Hugging Face's SHA-256. What's left is the
pod image: booting a pod straight from R2.

## Why

- **The S3 gateway can't fill a fresh 30GB volume with the backbone.**
  `copy-volume.py --push` fails with `QuotaExceeded` on CompleteMultipartUpload:
  the gateway counts the multipart parts and the finished object against the
  quota (~47 GB). A pod writing to `/workspace` directly never touches the gateway.
- **Not every datacenter has network volumes.** A pod with its own disk
  (`--pod-volume-gb`) needs the models at boot from somewhere: today that's
  Hugging Face, or `MODELS_S3` from a seeded volume.
- **A narrower key.** `MODELS_S3` puts the RunPod S3 key in the pod's env, and
  that key reads and writes every volume. An R2 token can be read-only and
  scoped to one bucket.
- **No download fees,** and one copy serves every datacenter (in Europe too).

## Design

- **`models.txt` gets a SHA-256 column** (taken from Hugging Face's
  `X-Linked-Etag`), and `start.sh` checks every downloaded file against it,
  whatever the source. Today the Hugging Face path checks nothing, and a
  truncated `curl` that exits 0 would pass.
- **Source order at boot:** `MODELS_SOURCE=r2,hf` (default `hf`; `s3` stays for
  `MODELS_S3`). Missing files come from the first source; a file that fails its
  checksum or download is retried from the next.
- **R2 settings from the env, no names in this repo:** `R2_ACCOUNT_ID`,
  `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` (a read-only, bucket-scoped
  token), `R2_BUCKET`, `R2_PREFIX` (default `models`). `create-pod.py` gets
  `--models-r2 ENV_FILE` to pass them from a local file; the README says how to
  make the bucket and token, generically.
- **rclone in the image, pinned** (version and SHA-256 checked at build), not
  downloaded at boot. `start.sh` calls it directly with the file list from
  `models.txt` (`rclone copy --files-from`), so this public repo doesn't depend on
  the private tooling. Flags carried over from the proven setup:
  `--modify-window 1s` (RunPod volumes keep whole-second dates; without it a
  no-op refill re-hashes everything), `no_check_bucket=true` (bucket-scoped tokens
  can't list buckets), quick retries on listing (bad credentials otherwise hang),
  4 transfers × 16 streams.
- **Free-space check** only on a pod's own disk: `df` on a network volume shows
  the whole storage cluster.

## Steps

1. SHA-256 column in `models.txt`; verify in `start.sh` (Hugging Face path first,
   on its own: it's a fix worth having anyway).
2. Pinned rclone in the Dockerfile.
3. `MODELS_SOURCE` / R2 path in `start.sh`, with per-file fallback.
4. `create-pod.py --models-r2 ENV_FILE`; build.md and README.
5. Test on a pod with its own disk in a datacenter without volumes, then on an
   empty network volume. Kill R2 mid-run (bad key) to see the fallback.
6. **Race R2 against Hugging Face on the same pod** in 2-3 datacenters (never
   measured side by side: R2 filled US-IL-1 in 112 s; Hugging Face filled a
   CA-MTL-1 pod in about 3 min, boot included). Make the faster one the default.
7. Slow-tail fix: a few of the 16 R2 streams run 2-4x slower, so a big file
   waits on its last range. Try smaller multi-thread chunks (e.g. 16M) and
   re-measure.

## Risks / open questions

- **A key on the pod.** Read-only and bucket-scoped, but anyone with the pod's
  env can read the models: fine for public weights, not for private data.
- **R2 outage or bad token:** the Hugging Face fallback covers it, if the
  checksums are in place (step 1).
- **Two copies of the fetch logic** (the private tooling's script and
  `start.sh`): keep `start.sh`'s to the minimum above, and the flags in sync.
- **Public repo:** no bucket, account or token names, and no reference to the
  private tooling repo.


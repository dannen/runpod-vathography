# Plan

Two independent phases: (1) a serverless depth endpoint (not started),
(2) running depth locally on a 16 GB GPU (answered: works with the cu128 image).

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
  volume (~2 min). Idle timeout and FlashBoot keep a worker warm across a
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

Left to do, if wanted: a README "Local GPU" section (run the cu128 image with
`--gpus all`, a local folder as `/workspace`, `batch.py --url
http://127.0.0.1:8188`). GGUF quants (e.g. `QuantStack/Qwen-Image-Edit-2509-GGUF`)
remain an option for cards under 16 GB or machines with little RAM.

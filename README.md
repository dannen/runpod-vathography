# Vathography depth (Marigold V2) on RunPod (persistent Pod)

ComfyUI running [Marigold V2](https://github.com/huawei-bayerlab/marigold-v2) monocular depth, normals and albedo estimation on a long-running RunPod **Pod**, used through ComfyUI's web UI via RunPod's HTTP proxy. This is **Phase 1** (pod exploration) of the RunPod plan in the `vathography.md` research notes (§14; not public). Its job is to measure VRAM, time and quality at each resolution, and settle which resolution, checkpoint and GPU tier a later serverless endpoint should use.

It uses a small image, with all models on a dedicated network volume mounted at `/workspace`. ComfyUI has supported Marigold V2 natively since **v0.36.0** (Comfy-Org/ComfyUI#16232), and all three official templates use core nodes only, so there are **no custom nodes**.

Only depth estimation runs here. All artistic work (`vathopy`, the Blender add-on) stays on the local workstation.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | CUDA 13.0 runtime, Python 3.12 venv, ComfyUI v0.41.3, torch 2.11.0+cu130 (cu130 is required for the int8 kernels) |
| `start.sh` | Entrypoint: wait for the volume, symlink dirs onto it, download missing models, seed workflows, run ComfyUI |
| `models.txt` | Model manifest (subdir + URL). **The only list of models.** `start.sh` and `populate-volume.sh` both read it |
| `populate-volume.sh` | Seeds the volume from this machine through RunPod's S3 gateway. `--dry-run` shows the plan |
| `copy-volume.py` | Seeds a new volume from an already-seeded one, RunPod to RunPod, on a throwaway CPU (or GPU) pod; each file checked against Hugging Face's SHA-256 |
| `workflows/image_marigold_v2_*.json` | The three official Comfy-Org templates, unmodified |
| `workflows/vathography_*.json` | Generated from those by `make_workflows.py`: input resize plus 32-bit float EXR output. Use these |
| `make_workflows.py` | Regenerates `vathography_*.json`. Every edit is asserted, so a changed upstream template fails loudly |
| `create-pod.py` / `.sh` | Create the volume; create / `--update` / `--recreate` the pod |
| `check-gpu-availability.py` / `check-gpus.sh` | Live GPU stock and pricing |
| `build.md` | Local build and push checklist |
| `sync-vathopy.sh` | Copies [`vathopy`](https://github.com/dannen/vathopy) from a checkout next to this one (`../vathopy`) into `vathopy-src/` (gitignored) for the image's GUI layer; run before every build |
| `batch.py` / `batch.sh` | Runs a folder of photos through the pod at the recommended size, waits, then runs `pull.sh`. Needs Pillow |
| `pull.py` / `pull.sh` | Downloads finished results from the pod's job history into `~/Pictures/vathography/depth`. Finds the running pod itself (cached 30 min) |
| `bench.py` | Phase 1 benchmark: runs depth/normals/albedo on the pod at given sizes, records time and peak VRAM, saves the EXRs |
| `podauth.py` | The pod's password (ComfyUI and the GUI share it) for `batch.py`, `pull.py` and `bench.py`: sent to the pod's proxy URLs only |
| `AGENTS.md` | Decisions, current state and open items, for an agent picking this up |

## Models (~26.4GB, see `models.txt`)

All from [Comfy-Org/marigold-v2-0](https://huggingface.co/Comfy-Org/marigold-v2-0).

| File | Dir | Size | Used by |
|---|---|---|---|
| `qwen_image_edit_2509_int8_convrot` | `diffusion_models/` | 20.5GB | All three workflows (shared backbone) |
| `marigold_v2_depth_log_stage2` | `loras/` | 1.7GB | Depth |
| `marigold_v2_depth_log_stage2_vae` | `vae/` | 0.25GB | Depth |
| `marigold_v2_depth_conditioning` | `embeddings/` | 2.3MB | Depth |
| `marigold_v2_normals` / `_vae` / `_conditioning` | `loras/`, `vae/`, `embeddings/` | 1.96GB | Normals |
| `marigold_v2_albedo` / `_vae` / `_conditioning` | `loras/`, `vae/`, `embeddings/` | 1.96GB | Albedo |

Only `depth/Log-stage2` is packaged for ComfyUI. The other checkpoints (Uniform-base, Disparity-base, the layered variants) exist only in the [reference pipeline](https://github.com/huawei-bayerlab/marigold-v2) (`vathography.md` §4, §6).

## Workflows

| File | Output |
|---|---|
| `vathography_depth.json` | `output/vathography/depth_*.exr`: near = bright, scaled 0–1, 32-bit float, values written unchanged |
| `vathography_normals.json` | `output/vathography/normals_*.exr`: unit normals mapped to 0–1, 32-bit float |
| `vathography_albedo.json` | `output/vathography/albedo_*.exr`: scene-linear albedo, 32-bit float |

Compared with the official templates, each one:

- **Resizes the input first, to multiples of 16.** Set the size on the **Inference size (long edge, px)** node, which sits under Load Image. The photo keeps its shape, and each side is rounded to the nearest multiple of 16 px, never above the photo's own size: a 3:2 photo at 2496 runs at 2496 × 1664. Three small collapsed nodes do the arithmetic (Get Image Size, then Width and Height math expressions) and feed an Image Scale node (lanczos, no crop).
  - **Why multiples of 16:** otherwise the VAE crops each side to a multiple of 8 and the model pads an odd side with content wrapped around from the opposite edge. The depth map would come out up to 7 px smaller than the photo, not covering the full frame. With multiples of 16 it's exactly the scaled photo's size, so scaling it back to the original's dimensions lines up. The rounding stretches the shape by at most 8 px per side (~0.5%), which scaling back undoes.
  - **Why resize at all:** the official templates run inference at the photo's own size, which for a camera photo (6000 px and up) runs out of memory. The default is **2496 px** (the multiple of 16 nearest 2500, where Phase 1 found the best depth on sharp photos; see below), the same as `batch.sh`. Change it for panoramas (~4200) or resolution tests.
- **Saves 32-bit float EXR** instead of 8-bit PNG, so the depth map isn't cut to 256 levels. That matters because the Blender add-on turns depth into real geometry and warns about banding on 8-bit maps.

All nodes on the main canvas are pinned so they can't be dragged by accident (right-click > Unpin to move one). The **Result preview** node shows the result as soon as the job finishes (an 8-bit preview; the EXR is what gets saved). The official templates' Image Compare node is removed: it starts on the input photo and only reveals the result as you drag across it, so it looked blank.

**What the depth EXR contains:**

- **Normalization:** `MarigoldV2PostProcess` averages the decoded channels, then normalizes each image by its own min and max, with near as bright. Values are therefore relative within one photo and can't be compared across photos.
- **Clipping:** ComfyUI's VAE decode clamps its output to 0–1 before that step, so any prediction outside the VAE's range is already clipped.
- **What's missing:** unlike the reference pipeline, there's no raw `.npy` of the prediction. The EXR is the closest equivalent.

## Setup

1. **Pick a GPU and datacenter.** A **24GB** card is enough (see [GPU and VRAM](#gpu-and-vram)). The recommended choice is an **RTX A5000 (24GB, ~$0.27/hr)** in **US-CA-2**. The datacenter must also serve RunPod's S3 API for step 3 and for `pull.sh` checksums (CA-MTL-1, CA-MTL-3 and US-CO-1 don't), and the pod and volume must be in the same datacenter.
   ```bash
   ./check-gpus.sh --vram 24
   ```
2. **Create the volume** in that datacenter:
   ```bash
   ./create-pod.sh --create-volume --datacenter <DC>   # vathography-storage, 50GB
   ```
3. **Seed the models** from this machine:
   ```bash
   ./populate-volume.sh <DC> <volume-id> --dry-run   # read-only: what's pending, sizes, local disk check
   ./populate-volume.sh <DC> <volume-id>             # ~26GB, smallest file first
   ```
   You can skip this step, because the pod downloads anything missing on first boot. Pods have been seen downloading from Hugging Face at under 1MB/s, which would take many hours for 26GB.

   **Or, if you already have a seeded volume in another datacenter,** copy it RunPod to RunPod instead (see [Seeding from another volume](#seeding-from-another-volume)):
   ```bash
   ./copy-volume.py --from <seeded DC>:<volume-id> --to <DC>:<volume-id>
   ```
4. **Build and push the image.** See [`build.md`](build.md).
5. **Create the pod:**
   ```bash
   ./create-pod.sh --image YOUR_DOCKERHUB_USER/vathography-comfyui:<git-sha> \
     --gpu-type-id "NVIDIA RTX A5000" --datacenter <DC> \
     --network-volume-id <volume-id>
   ```
   The script only accepts hosts with CUDA 13.0 drivers (`--allowed-cuda-versions`, default `13.0`). If it can't find stock, check that filter first. Add `--env SKIP_MODEL_DOWNLOAD=true` if step 3 is still uploading.
6. Open `https://<pod-id>-8188.proxy.runpod.net` and log in with the same user and password as the vathopy GUI (see **Passwords** below). The workflows are in the **Workflows** sidebar.

Pods are billed while running. Stop the pod when you're done, and the volume keeps everything (50GB is about $3.50/month).

## vathopy GUI on the pod

The image also runs the `vathopy` web GUI (palettes including your own, cycles, contour lines, creative focus with highlights, looks and 16-bit PNG downloads; see [`vathopy`'s README](https://github.com/dannen/vathopy)) on **port 8770**, next to ComfyUI:

- **URL:** `https://<pod-id>-8770.proxy.runpod.net`, with a password: RunPod's proxy URLs are public.
- **Input:** the depth maps in ComfyUI's output folder, `/workspace/output/vathography/`, so anything the workflows or `batch.sh` produce shows up in its list (depth EXRs only).
- **Saves:** go to `/workspace/favorites/` on the volume (PNG + `looks/<name>.json`). The page links each save for download. They're also reachable through S3 under `favorites/`.
- **Your palettes:** `/workspace/palettes/` on the volume (created at boot), the GUI's palette folder: images, `.json` or `.gpl` files, the same as on a workstation (see `vathopy`'s README, "Your own palettes"). The GUI re-reads it on every page load, so upload, then reload the page. Check them locally first, then upload each through the volume's S3 gateway (the profile, endpoint and region flags as in "Getting depth maps back" below):

  ```bash
  vathopy check-palettes ~/Pictures/vathography/palettes      # every file OK, or fix/remove it first
  aws s3api put-object --bucket <volume-id> --key palettes/dusk.png --body ~/Pictures/vathography/palettes/dusk.png \
    --profile runpods3 --endpoint-url https://s3api-<dc>.runpod.io --region <dc>
  ```

  The pod's GUI runs the same checks and skips anything that fails (its log, `/tmp/vathopy-gui.log`, says why). Looks saved with one of these palettes need it in the folder wherever they're rendered again.
- **Passwords:** `VATHOPY_GUI_AUTH` (`user:password`) in the pod's environment protects **both** the GUI (8770) and **ComfyUI** (8188). ComfyUI has no login of its own, so `start.sh` puts nginx in front of it: nginx on 8188 asks for the password (HTTP basic auth) and passes everything, websockets included, to ComfyUI on `127.0.0.1:8189`, which can't be reached any other way. Set `COMFYUI_AUTH` to give ComfyUI a different `user:password`, or `COMFYUI_AUTH=off` to leave it open. Without any password both start open and log a warning; if nginx fails to start, ComfyUI stays local-only (unreachable) rather than open. Keep the password in a local file outside the repo (mode 600, `~/.api/runpod/vathopy-gui.auth`) and pass it on every `--update` (see `build.md`).
- **The tools log in too:** `batch.sh`, `pull.sh` and `bench.py` send the password on requests to the pod's proxy URLs (only there, never to RunPod's API), from `VATHOPY_GUI_AUTH` or the file `~/.api/runpod/vathopy-gui.auth` (another file: `VATHOPY_GUI_AUTH_FILE`); see `podauth.py`. Without it they stop with a message saying where to put it.
- **Logs and control:** on the pod, `/tmp/vathopy-gui.log`. It's restarted if it exits. Set `VATHOPY_GUI=off` to not start it.
- **Building:** it has its own venv (`/opt/vathopy`), so its numpy/OpenCV/OpenEXR can't affect ComfyUI's. Run `./sync-vathopy.sh` before `docker build` (see `build.md`). The image records the source commit in `/opt/vathopy/VATHOPY_COMMIT`.

## Seeding from another volume

`copy-volume.py` seeds a new volume from one that's already seeded, without the 26GB passing through your machine. Useful when you move datacenters (a GPU type sold out, or a host fault), or keep volumes in several so a pod can start wherever there's stock.

```bash
./copy-volume.py --from US-MO-1:<seeded-volume-id> --to US-NE-1:<new-volume-id>
./copy-volume.py --from US-MO-1:<seeded-volume-id> --to US-IL-1:<new-volume-id> --gpu "NVIDIA RTX A5000"
```

1. It reads `models.txt` and gets each file's SHA-256 from Hugging Face (the `X-Linked-Etag` header).
2. It starts a throwaway pod with the **target** volume at `/workspace`. By default that's a 2 vCPU CPU pod (~$0.06/hr), trying a few CPU instance types (`--cpu cpu3c-2-4` picks one). `--gpu TYPE` uses a GPU pod instead, for datacenters with no CPU supply.
3. On the pod, a short job `get-object`s each file from the **source** volume's S3 gateway into `/workspace/models/`, smallest first, and checks its SHA-256 (3 tries per file). Files already there and correct are skipped, so rerunning resumes. It logs to `/workspace/seed.log` and leaves `seed.done` or `seed.failed`.
4. The script follows `seed.log` through the target's S3 gateway and **deletes the pod** when the job ends: the pod holds your S3 key in its env. Ctrl-C (or `--no-follow`) leaves the pod running, so delete it yourself once `seed.done` appears.

Both datacenters need RunPod's S3 API. The source is read over it, and the target's is used to follow the log. CPU pods are created through the GraphQL API (`deployCpuPod`), because the REST `/v1/pods` endpoint rejects some datacenters (US-NE-1 in October 2026).

Measured US-MO-1 → US-NE-1 (2026-10-19): 42–50 MB/s on most files. The source gateway dipped to 3 MB/s for a while, and one attempt at the 20.5GB backbone failed (the retry ran at 30 MB/s). It took ~24 minutes and cost ~$0.03. A partly downloaded file doesn't show up in S3 listings, so the log is quiet while the backbone copies.

## GPU and VRAM

Measured on an L40S (2026-10-18, `bench.py` and real photos; details in `AGENTS.md`). The INT8 model's weights take a constant **~20 GiB** once loaded, and each job adds little on top:

| Inference size (long edge) | Peak VRAM | Added by the job |
|---|---|---|
| up to 1280 px | 20.0–20.1 GiB | ~0.1 GiB |
| ~2500 px (`batch.sh` default) | 20.2–20.5 GiB | ~0.5 GiB |
| ~3500 px, and 4200 px panoramas | 21.3–21.8 GiB | up to ~1.8 GiB |
| ~4000 px | 34.8 GiB | ~15 GiB |
| ~4640 px | 38.7 GiB | ~19 GiB |

- **24GB is enough** for everything up to ~3500 px, with 2–3 GiB to spare. Only sizes past ~4000 px need more, and depth quality already falls off there. The peaks are sampled every 0.1–0.2 s, so a very short spike could be missed. If a 24GB card does run short, ComfyUI offloads part of the model to system RAM, so the job runs slower rather than failing.
- **Use a cheap, slower card.** A depth job takes seconds, while the pod stays up for minutes at a time, so the hourly rate matters far more than speed. In practice, pricier GPUs for shorter runs have cost about the same per job.
- **The GPU needs compute capability 8.0 or higher** (Ampere or newer) for the INT8 kernels, and a host with **CUDA 13 drivers** (`create-pod.py` filters for those).

Options, from `./check-gpus.sh` on 2026-10-18 (stock changes hourly):

| GPU | VRAM | $/hr | Datacenters with an S3 gateway | Notes |
|---|---|---|---|---|
| **RTX A5000** | 24GB | **0.27** | US-CA-2, US-IL-1 | **Recommended.** Ampere: expect ~20–30 s at 2500 px (not yet measured) |
| RTX 4090 | 24GB | 0.89 | US-CA-2, US-IL-1 | Faster (Ada), but more than 3× the price |
| A40 | 48GB | 0.59 | US-MO-1 | No stock with CUDA 13 hosts when tried |
| L40S | 48GB | 1.09 | US-MO-1, US-IL-1 | ~5 s at 2500 px. What Phase 1 ran on |

**Prefer US-CA-2.** It has A5000 and 4090 cards, plus 48–96GB fallbacks (RTX 6000 Ada, A100, RTX PRO 6000), and an S3 gateway. US-IL-1's S3 gateway had an outage on 2026-10-01.

**Moving a setup from US-MO-1** (which has no 24GB cards with network-volume support). Recommended order, not done yet:
1. Create a new 50GB volume in US-CA-2. Before seeding it, start an A5000 pod on it with `--env SKIP_MODEL_DOWNLOAD=true`, which proves A5000 hosts there have CUDA 13 drivers (the A40 attempt failed on that).
2. Seed it with `./populate-volume.sh US-CA-2 <new-volume-id>`, about 26GB.
3. Run `bench.py` on the A5000 at 1280, 2500 and 3500 px to confirm VRAM and speed.
4. Only then delete the L40S pod and the US-MO-1 volume, after pulling anything still on it (`./pull.sh --from-volume`).

## Batch processing

`./batch.sh <folder>` runs every photo in a folder through the pod and downloads the results:

```bash
./batch.sh ~/Pictures/some-folder                   # depth for every photo, 2496 px long edge
./batch.sh DIR --resize 4200                        # 4200 px long edge (panoramas)
./batch.sh DIR --predictions depth normals albedo   # all three
./batch.sh DIR --via-s3                             # upload through the volume's S3 gateway, not the web proxy
./batch.sh DIR --dry-run                            # show the planned sizes; upload nothing
```

1. **Sizing:** `--resize 2496` (the default, matching the workflows) or `--resize 4200` sets the long edge in pixels, keeping the photo's shape.
   - **2496:** ~2500 gave the best depth on regular photos (depth degraded by 3750); 2496 is the nearest multiple of 16.
   - **4200:** for panoramas, since 2496 leaves too few pixels top to bottom (a 4.3:1 panorama at 2496 is 2496 × 576). With the default, wide photos (2:1 or wider) are flagged in the plan.
   - **Small photos:** a photo smaller than the target is never enlarged.
2. **Preparing:** the photo is resized here, before upload (EXIF rotation applied, Lanczos, JPEG quality 95). Uploads are a few MB instead of 10–120 MB.
3. **Uploading:** by default through ComfyUI's upload, via RunPod's web proxy. `--via-s3` writes each photo straight into the pod's volume (`input/batch-<time>/`) through RunPod's S3 gateway instead, and checks the MD5 the gateway reports against the local file. That avoids the proxy and its 100 MB limit. It needs the `aws` CLI and `~/.api/runpod/s3-credentials`, like `pull.sh`.
4. **Queueing:** jobs are queued grouped by prediction (all depth, then normals, then albedo), so the model switch (30–45 s) happens once each. Results are saved on the volume as `vathography/<photo>_<prediction>_NNNNN.exr` (e.g. `IMG_0001_depth_00001.exr`), the same as the browser workflows.
5. **Downloading:** when every job has finished, it runs `pull.sh`, which downloads and checksum-verifies the results into `~/Pictures/vathography/depth/` (`--out` to change). `--no-pull` skips that.

Photos that already have a result in the output folder are skipped; `--force` reruns them. Each run uploads into a new `input/batch-<time>/` folder, so names can't collide. Needs Pillow (`pip install Pillow`). Both `batch.sh` and `pull.sh` target the running pod named `vathography-comfyui`; use `--name` for another pod, such as a test pod.

**The 100 MB upload limit:** both RunPod's web proxy (Cloudflare) and ComfyUI (`--max-upload-size`, default 100 MB) cap browser uploads at 100 MB. `batch.sh` resizes before uploading, so it never gets near that, and `--via-s3` bypasses both. SSH (`scp`) or having the pod fetch a URL would need an SSH server in the image, which it doesn't have.

## Verification

1. `curl -sf -A curl/8 -u "$(cat ~/.api/runpod/vathopy-gui.auth)" https://<pod-id>-8188.proxy.runpod.net/system_stats` returns ComfyUI (and a 401 without `-u`) `0.41.3` and a CUDA device. The server log should show `comfy_kitchen backend cuda: available: True`. If it doesn't, the int8 kernels are falling back to a slow path.
2. Load `vathography_depth.json`, upload a photo into Load Image, and queue it at the default 2496 px long edge. Check that an EXR appears under `/workspace/output/vathography/`.
3. Run `vathography_normals.json` and `vathography_albedo.json` once each.
4. Locally, run the EXR through `vathopy` (step 3 of the next section) and load it in the Blender add-on.

## Getting depth maps back to the workstation

**Usually: `./pull.sh`.** It downloads every finished result from the pod into `~/Pictures/vathography/depth/`, and checks each download against the volume's own checksums:

```bash
./pull.sh                     # pull everything new from the pod's job history
./pull.sh --list              # show what's on the pod and what's already pulled
./pull.sh --out DIR           # somewhere else
./pull.sh --refresh           # look the pod up again now
./pull.sh --from-volume       # read the volume's S3 bucket directly (no pod needed)
./pull.sh --from-volume --match 'IMG_*'   # only files whose name matches
./pull.sh --verify-existing   # re-check files already downloaded against the volume
./pull.sh --clear-volume      # delete results from the volume once safely pulled
./pull.sh --clear-inputs      # delete uploaded photos (input/) from the volume
```

- **Finding the pod:** the pod can be different each time (it gets recreated), so `pull.sh` finds the running pod named `vathography-comfyui` through the RunPod API. It caches that for 30 minutes (`~/.cache/runpod-vathography/target.json`, `--cache-minutes`), and looks it up again after that, or straight away if the cached pod stops answering. `--url` skips the lookup.
- **File names:** `<photo>_<depth|normals|albedo>_<long edge>px.exr`, plus `<same>_preview.png` when the job has a Result preview. Nothing is ever overwritten: if a file with different contents already has that name, the new one gets `-001`, `-002`, and so on (`photo_depth_1280px-001.exr`). An identical file isn't downloaded again. Previews follow the same rule.
- **Checksums:** pulling from a pod checks against the volume attached to that pod (asked from RunPod). ComfyUI saves results on the network volume, which is also an S3 bucket, so the volume can vouch for each file independently of how it was downloaded. RunPod's S3 gateway reports each file's MD5 (as its ETag), and the full SHA-256 on a whole-file download.
  - Pulled through the pod: the size and MD5 must match the volume.
  - Pulled with `--from-volume`: the size, MD5 and SHA-256 must all match.
  - A failed or mismatched download is tried again, up to 3 times. If the pod can't deliver a good copy (for example while it's restarting, which gives HTTP 502), the same file is fetched from the volume through S3 instead. A file that never matches isn't saved, and the next run retries it.
  - `SHA256SUMS` in the output folder lists every verified file, so `sha256sum -c SHA256SUMS` re-checks the folder locally at any time.
  - If the volume can't be reached (no `aws` CLI or credentials, or the S3 gateway is down), the pull stops. `--no-verify` downloads unchecked, and `--verify-existing` checks those files later.
- **vathopy GUI saves:** every pull (from the pod or `--from-volume`, including `batch.sh`'s) also downloads what the GUI on the pod saved, `favorites/` on the volume (PNGs and `looks/*.json`), into `~/Pictures/vathography/favorites/` (`--favourites-out`). It reads them through S3 with the same checksum checks, so it needs no GUI password or running GUI. Nothing is overwritten: a pod save named like an existing local look arrives as `looks/<name>-001.json`. That folder has its own `.pulled.json` and `SHA256SUMS`. `--no-favourites` skips it.
- **No repeats:** what's been pulled is recorded in `.pulled.json` in the output folder, per pod and per volume file (with its MD5), so reruns only fetch new results. ComfyUI reuses file names such as `depth_00001.exr` after the originals are deleted, so a volume file counts as pulled only if its contents match the record. `bench.py` runs are skipped unless `--include-bench`.
- **`--from-volume`:** reads `output/` on the volume through S3, for anything the pod's history no longer covers (a pod restart clears it). Files there only have ComfyUI's names (`depth_00007.exr`). Anything already pulled under its photo name is skipped. The volume is looked up by name (`--volume-name`, default `vathography-storage`); `--volume-id` and `--datacenter` skip that. It needs the `aws` CLI and `~/.api/runpod/s3-credentials`.
- **`--clear-inputs`:** deletes everything under `input/` on the volume: photos uploaded in the browser and `batch.sh`'s `input/batch-*/` folders. Uploads otherwise stay there for good.
  - It keeps ComfyUI's own `example.png` and its empty `_` placeholder files, and never touches models or `output/`.
  - A queued or running job still needs its photo, so it refuses while the pod's queue isn't empty.
  - It lists what it will delete, by folder, and asks first. `--yes` skips the question.
- **`--clear-volume`:** deletes `.exr` and `.png` files under `output/` on the volume, never models or inputs.
  - By default it only deletes files with a local copy whose checksum matches the volume right now.
  - Everything else (not pulled yet, `bench.py` runs, PNGs from other workflows) is listed and kept, unless you add `--include-unpulled`.
  - It shows exactly what it will delete and asks first. `--yes` skips the question.
  - Later pulls record jobs whose files are gone instead of retrying them.

**Keeping the volume tidy.** Results and uploads stay on the volume until you delete them. A routine after a session:

```bash
./pull.sh                                  # 1. download and verify everything new
./pull.sh --from-volume --list             # 2. anything left that isn't pulled? (e.g. from before a pod restart)
./pull.sh --clear-volume                   # 3. delete results that have a verified local copy
./pull.sh --clear-inputs                   # 4. delete the uploaded photos (waits for an empty queue)
```

`--clear-volume` only deletes results with a matching local copy, so step 3 is safe to run at any time; anything not yet pulled is kept and listed. `--clear-inputs` deletes every upload, so only run it once the photos aren't needed for reruns.

**From the volume through S3** (for anything `pull.sh` can no longer see):

`/workspace/output` is on the network volume, so it's reachable through the same S3 gateway as `populate-volume.sh`:

1. **List the outputs:**
   ```bash
   AWS_SHARED_CREDENTIALS_FILE=~/.api/runpod/s3-credentials \
     aws s3 ls s3://<volume-id>/output/vathography/ --profile runpods3 \
     --endpoint-url https://s3api-<dc>.runpod.io --region <dc>
   ```
2. **Download each file with `aws s3api get-object`**, using the same profile, endpoint and region flags:
   ```bash
   aws s3api get-object --bucket <volume-id> --key output/vathography/depth_00001_.exr depth_00001_.exr
   ```
   Not `aws s3 cp` or a presigned URL: RunPod's S3 gateway rejects `HeadObject` with a 403, and both of those depend on it (a known RunPod gateway quirk). `<dc>` is the datacenter ID in **lowercase**, e.g. `us-ks-2`.
3. **Clean it up locally:**
   ```bash
   vathopy prep depth_00001_.exr --kind near-bright -o depth.exr
   ```
   `--kind near-bright` matters: the default `log` assumes raw Marigold output, where far is larger. `prep` fills NaN and inf and clips at the 0.1/99.9 percentiles, which the add-on doesn't do itself.

You can also download single files from the ComfyUI UI (queue → image → save).

## Phase 1 measurements

What to measure (`vathography.md` §14.4), with results recorded in `AGENTS.md`:

- **Peak VRAM and time** at long edges of 1280, 2500 and 3750 px, using `bench.py` (it records both, and saves each EXR):
  ```bash
  python3 bench.py --url https://<pod-id>-8188.proxy.runpod.net \
    --image photo.jpg --long-edge 1280 2500 3750 --repeat 2 --out bench-out/
  ```
  The first run after boot includes loading the model from the volume, so `--repeat 2` gives a warm time.
- **Edge detail versus resolution:** use photos at least ~4000 px on the long edge, so every test size is a reduction. Compare the 2500 and 3750 px maps against the 1280 px map upscaled to the same size. Look at hair, foliage, railings and cables at 100% zoom.
- **Fit of the VRAM model:** check whether memory really grows linearly with pixels, as the `vathography.md` §14.3 table assumes.
- **Outside this image:** comparing ComfyUI INT8 against the reference 4-bit pipeline needs the reference repo in its own environment (`vathography.md` §6). It isn't in this image.

## Notes

- **Uploads over 100 MB fail with HTTP 413.** That's Cloudflare in front of RunPod's proxy, not ComfyUI. Save large photos as high-quality JPEG (quality 95) instead of PNG: a 16350 × 3828 panorama is 120 MB as PNG and 8 MB as JPEG.
- **Panoramas need a larger long edge.** The workflow sizes by the long edge, so the default 2496 turns a 4.3:1 panorama into 2496 × 576. Use about 4000–4200 px (4000 × 936), which is roughly the pixel count of a 2496 px regular photo.
- **Scripting the API through the proxy:** Cloudflare rejects Python urllib's default User-Agent (`error code: 1010`). Set a User-Agent header.
- **Proxy timeout:** RunPod's proxy drops connections after about 100s, which can cut the progress websocket during long runs. The job keeps running on the server, and refreshing the tab reconnects.
- **Serverless:** not set up. It's Phase 2 (`vathography.md` §14.5), built after Phase 1 fixes the resolution and GPU tier. `runpod/worker-comfyui` 5.11.0 ships ComfyUI 0.34.0, which predates Marigold V2 (v0.36.0), so it needs either a newer ComfyUI or its own handler. The pod tooling's lessons apply: results go to S3 rather than base64, the region is lowercase, downloads use `get-object`, and the template is pinned to a git-sha image tag.
- **Licenses:** the Marigold V2 code is Apache-2.0. Check the terms of the weights and the Qwen-Image-Edit base model before any commercial use or redistribution.

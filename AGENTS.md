# AGENTS.md: Vathography depth (Marigold V2) on RunPod (persistent Pod)

For an AI agent picking up this repo. `README.md` is the runbook. This file records the current state and the decisions behind it. The research behind it is in the `vathography.md` notes, which aren't public (§14 is the RunPod plan).

## Decisions

- **Persistent Pod, not serverless.** Phase 1 (pod exploration, `vathography.md` §14.4) comes before Phase 2 (serverless), and work has stayed on the pod since. Don't add serverless tooling unless asked.
- **ComfyUI v0.41.3 + torch 2.11.0+cu130.** Marigold V2 is native from v0.36.0 (Comfy-Org/ComfyUI#16232). cu130 is required for comfy-kitchen's CUDA backend, which runs the int8_convrot backbone (`comfy/quant_ops.py` disables it below CUDA 13). Pods on this (cu130) image are restricted to CUDA 13.0 hosts via `allowedCudaVersions`. comfy-kitchen's kernels need compute capability ≥ 8.0 (Ampere or newer).
- **cu128 variant (2026-10-20):** CUDA 13 hosts are scarce, so the image also builds against CUDA 12.8 (`BASE_IMAGE`/`CUDA_TAG`, tag suffix `-cu128`). The int8 backbone then runs on comfy-kitchen's eager (plain PyTorch) backend; depth maps match the cu130 build's within ~1%. `create-pod.py` accepts 12.8–13.0 hosts for `*-cu128` tags; cheap GPUs often pass no CUDA filter at all yet run it fine, so the last pod used `--allowed-cuda-versions ""`.
- **Local runs (2026-10-20):** the cu128 image runs on a 16GB card (`run-local.sh`): ComfyUI's dynamic VRAM loading holds part of the 20GB backbone in system RAM. ~24 s at 2496 px on an RTX 4060 Ti. No GGUF needed (`plan.md` Phase 2).
- **R2 as a model source (2026-10-20):** RunPod's S3 gateway can't put the 20.5GB backbone on a 30GB volume (`QuotaExceeded` counts parts plus the object), and `MODELS_S3` puts an all-volumes key on a pod. A private Cloudflare R2 bucket with a read-only token avoids both; tooling outside this repo fills volumes from it (README "Workaround"). Since 2026-10-28 the bucket also holds the see-through set (13 files, 37.2 GiB). Booting a pod straight from R2 is `plan.md` Phase 3, not built.
- **Custom nodes: ours only** (`custom_nodes/vathography`): Vathography Output Name names outputs after the photo; Vathography Tiled Depth (`tiled.py`) and the Vathography See-through Model loader (`see_through.py`) were written from scratch on 2026-10-28 for feature parity with the practice's own AGPL node pack, which was read but deliberately not used (the owner wants nothing from its code, and independence from the Blender add-on's needs). No third-party nodes (`vathography.md` §7 warns against community Marigold nodes).
- **Tiled depth design:** a global pass at up to 2496 px (Phase 1's best whole-photo size), never enlarged; 2048 px full-resolution tiles overlapping ≥512 px; a side shorter than a tile gets shorter tiles, so nothing is ever padded; each tile matched to the global map by a whole-tile Huber scale-and-offset fit, raised-cosine blending, and shapes above `detail_sigma` = 64 global-pass px taken from the global map. Modes full and draft (global only). Tried and rejected: a locally varying (guided-filter) fit (a seam and fringe shifts), split 4 (rings round fish a tile placed at another depth), blurring tiles before prediction for grainy film scans (below). Grainy black-and-white scans go through the plain workflow or `draft`, not tiled (README says so). `VATHOGRAPHY_TILED_DUMP=<dir>` saves raw predictions so merges can be retuned offline.
- **See-through model:** the nf4 Qwen-Image-Edit-2509 backbone (settings from its safetensors metadata, `unet_config`) dequantised per layer by bitsandbytes 0.48.2 (in the Dockerfile), its LoRA added unmerged; plain `ModelPatcher`, so it sits fully in VRAM (11.2GB).
- **Volume contents: depth, normals and albedo** (all three Marigold V2 predictions), about 26.4GB, plus the optional see-through set (13.5GB, `models-see-through.txt`, downloaded only with `MODEL_SETS=see-through`), which needs a 45GB volume. Only depth/Log-stage2 is packaged for ComfyUI by Comfy-Org.
- **Licence:** GPL-3.0-or-later (2026-10-28), like ComfyUI, whose modules the nodes import, and vathopy.
- **`vathography_*.json` is generated, never hand-edited.** `make_workflows.py` builds `vathography_depth_tiled.json` / `_see_through.json` from scratch, and the other three from the untouched official templates:
  - Resize by long edge in pixels (sizes are always given in pixels, never megapixels; default 2496 since 2026-10-18, was 1280), each side rounded to the nearest multiple of 16 (PrimitiveInt + GetImageSize + two ComfyMathExpression + ImageScale, no crop). That avoids the VAE's 8 px center crop and the DiT's circular padding, so the map is exactly the scaled photo's size.
  - 32-bit float EXR save, named `<photo>_<prediction>_00001.exr` by our node.
  - A Result preview node; Image Compare removed (it looked blank); every main-canvas node pinned.
  - `bench.py` and `batch.py` use the same sizing rule.
  - `start.sh` seeds workflows with `--ignore-existing`, so a running pod keeps old copies: push new ones with `POST /api/userdata/workflows%2F<name>?overwrite=true`.
- **No raw `.npy`.** The output is the post-processed map: per-image min/max normalized, near = bright, after the VAE decode clamped it to 0–1. `vathopy prep` needs `--kind near-bright` for these files.
- **Seeding:** from this machine (`populate-volume.sh`, conservative S3 multipart settings), or volume to volume (`copy-volume.py`, 2026-10-19). `copy-volume.py` runs a throwaway CPU pod on the target volume that pulls from the source volume's S3 gateway with `get-object`, checks each file against Hugging Face's SHA-256 and logs to `/workspace/seed.log`; the script deletes the pod when done (it holds the S3 key). It avoids the S3 gateway's upload problems below, since nothing is uploaded. CPU pods go through GraphQL `deployCpuPod`: REST `/v1/pods` rejects US-NE-1. Within one datacenter it needs no pod: S3 `copy-object` between the two volumes works server-side (~110 MB/s); a 20GB copy gets Cloudflare's 524 after ~2 min but lands anyway (~3 min), so the script polls for full size, then verifies (local copy's MD5/SHA-256, else download + SHA-256). `--push` runs the pod in the source datacenter and uploads (target with no pod supply): 8MB multipart (Cloudflare 413s anything over 100MB), verified by recomputing the multipart ETag from the source copy (the gateway ignores `x-amz-checksum-sha256`); files over `--max-size` (5GB, i.e. the backbone) are skipped for `populate-volume.sh`.
- **GPU: 48GB is plenty; 24GB would do** (measured: ~20 GiB of weights plus ≤1.8 GiB per job up to ~3500 px). Prefer cheaper, slower cards: depth jobs take seconds while the pod stays up for minutes.
- **The vathopy GUI runs on the pod** (port 8770), in its own venv so its numpy/OpenCV/OpenEXR can't affect ComfyUI. `sync-vathopy.sh` copies `vathopy` (https://github.com/dannen/vathopy; default checkout `../../vathopy` from this folder, or pass its path) into the gitignored `vathopy-src/` before each build and records the commit in `VATHOPY_COMMIT`. Image tags are `<this repo's sha>-<vathopy sha>[-cu128]`; `latest` / `latest-cu128` follow the newest. Pods are rolled out by the sha tag, never `latest`.
- **Your palettes on the pod:** `start.sh` creates `/workspace/palettes/` on the volume and starts the GUI with `vathopy --palettes /workspace/palettes gui …`. Upload palette files there through the S3 gateway (`put-object`, key `palettes/<file>`) after `vathopy check-palettes` passes locally; the GUI re-reads the folder on each page load and skips (and logs) anything that fails vathopy's checks. Looks saved with a user palette refer to it by name.
- **Untrusted files:** vathopy reads palettes and depth maps as data only and checks them before decoding (header-first size caps, regular files only, no pickles). That matters here because whatever reaches ComfyUI (uploads, any workflow's saves) lands in the folder the GUI reads, and ComfyUI's proxy URL is public; it's password-protected since 2026-10-18 (below), but the checks don't rely on that.
- **ComfyUI is behind the same password** (2026-10-18): it has no login of its own, so `start.sh` runs nginx on 8188 (basic auth from `COMFYUI_AUTH`, default `VATHOPY_GUI_AUTH`; websockets passed through) in front of ComfyUI on `127.0.0.1:8189`. No password: open with a warning; nginx failing: ComfyUI stays local-only (fail closed). `batch.py`/`pull.py`/`bench.py` send it via `podauth.py` (env or `~/.api/runpod/vathopy-gui.auth`), to `*.proxy.runpod.net` only.
- **Password on the GUI:** RunPod's proxy URLs are public. `VATHOPY_GUI_AUTH` (user:password) comes from the pod env; keep the credentials in a local file outside the repo (mode 600; `build.md` uses `~/.api/runpod/vathopy-gui.auth`). `create-pod.sh --update` replaces the whole env, so **always pass `--env "VATHOPY_GUI_AUTH=$(cat ~/.api/runpod/vathopy-gui.auth)"`** (see `build.md`).
- `models.txt` is the only list of the standard models (`start.sh` and `populate-volume.sh` both read it); optional sets are `models-<set>.txt`, which only `start.sh` reads, for each set in `MODEL_SETS`.

## Current state (2026-10-28)

- **No pod.**
  - **Last pod:** an A40 48GB in CA-MTL-1 ($0.59/hr) on the cu128 image `be6dda3-d52f5b4-cu128`, with a 50GB pod volume (no network volume in that datacenter), created with `--allowed-cuda-versions ""`. Verified: 401/200 on 8188 and 8770; 2496 px depth jobs ~13 s including upload; maps within ~0.5–0.9% of local runs. Deleted 2026-10-20; its pod volume went with it.
  - **Before that:** the L40S pod in US-MO-1 was deleted on 2026-10-19 after RunPod reported a critical error on its host, and no L40S was free to recreate it.
- **Images** (Docker Hub, `YOUR_DOCKERHUB_USER/vathography-comfyui`):
  - Newest: `2fb0d2f-08d6269` (cu130, = `latest`) and `2fb0d2f-08d6269-cu128` (= `latest-cu128`), built and pushed 2026-10-28: the tiled depth and see-through nodes, bitsandbytes, both new workflows, `pull.py` naming tiled/see-through results, and vathopy `08d6269` (add-on 1.0.96 colours and blend modes, local contrast, Duplicate, the depth map list's sort menu, Cut Reset). The cu128 one runs locally (`run-local.sh`); a local test build of the same nodes ran both workflows through the frontend and batch full/draft jobs; **neither has run on a pod yet**.
  - Before that: `3a869d4-bf51844` (+`-cu128`), 2026-10-28, the first with the new nodes; `fd3a831-ee7556c` (+`-cu128`), 2026-10-21. Verified on pods: `f600688-6374357-cu128` (RTX PRO 6000 MIG, US-NE-1, 2026-10-21), `be6dda3-d52f5b4-cu128` (above), `061a155` (cu130, 2026-10-18). Never run: `621c97b`, `2363fa3`.
- **Volumes**, 30GB each (`--volume-size`; models 26.38GB) except US-NE-1, named `vathography-storage-<dc suffix>`:
  - US-MO-1: seeded, by server-side `copy-object` from the original 50GB volume (since deleted with its old outputs and favorites).
  - US-NE-1: seeded by `copy-volume.py` (CPU pod); **grown to 45GB on 2026-10-28 and given the see-through set** from R2 (all 13 files = Hugging Face's SHA-256). The one volume for see-through pods (`--env MODEL_SETS=see-through`). Has RTX PRO 6000 MIG 24/48GB, which ran the cu128 image on 2026-10-21.
  - US-IL-1: seeded from R2 on 2026-10-20 (a GPU pod, 112 s, every file = Hugging Face's SHA-256). Two earlier US-IL-1 volumes, which `copy-volume.py --push` couldn't fill, were deleted.
  - `create-pod.sh` takes comma lists for `--datacenter` and `--gpu-type-id` and tries them in order, finding each datacenter's volume by `--volume-name` prefix (GraphQL fallback where REST rejects the datacenter).
- **Region:** North America only for now (owner's choice, 2026-10-19).
- **Local tools:**
  - `batch.sh DIR`: resizes locally (`--resize 2496|4200`, default 2496, sides snapped to ×16; `--enlarge` also enlarges smaller photos to it, plain predictions only), uploads (proxy or `--via-s3`), queues grouped by prediction, waits, then runs `pull.sh`. `--predictions depth-tiled depth-see-through` upload the photo at its own size for the tiled node; `--draft` runs only its global pass.
  - `pull.sh`: finds the running pod (cached 30 min) and downloads results to `~/Pictures/vathography/depth/`, plus the GUI's `favorites/` saves to `~/Pictures/vathography/favorites/` and its crops. Verified against the volume's S3 checksums (MD5, SHA-256), falls back to S3 if the pod can't deliver, never overwrites (`-001`), keeps MD5-aware records (ComfyUI reuses names after deletions). Also `--from-volume`, `--match`, `--verify-existing`, `--clear-volume`, `--clear-inputs`.
  - `bench.py`: time and peak VRAM per size.
  - `run-local.sh`: the image on this machine's GPU, folders and ports from `--config` (relative paths relative to the file, so a project folder can carry its own config) or `~/.config/vathography/run-local.conf`.
- **The US-CA-2 24GB test was cancelled** (2026-10-18) and its volume deleted. A5000, RTX 6000 Ada and 4090 had no CUDA 13 hosts there for an hour of retries.
- **Seeding gotcha (US-MO-1 S3 gateway):**
  - `CompleteMultipartUpload` failed on large files with `InvalidPart`/`NoSuchUpload` or a read timeout; in one case the object had been written in full anyway.
  - The 20.5GB backbone was verified by sampled ranged `get-object` comparisons (43 ranges) instead of a third upload.
  - A killed upload leaves its parts under `.s3compat_uploads/`; clear them with `aws s3api abort-multipart-upload` (retry with `--cli-read-timeout 600` if it times out).
- **ComfyUI's `lanczos` upscale goes through 8-bit PIL.** Harmless where the workflows use it (on the input photo), but never put a lanczos resize after the model.

## Open items

1. Run the newest image on a pod and check it: password 401/200 on 8188 and 8770; the tiled and see-through workflows (see-through on the US-NE-1 volume with `MODEL_SETS=see-through`); `batch.sh --predictions depth-tiled` end to end, including `pull.sh`'s names; the GUI's newer controls.
2. Consider making `--allowed-cuda-versions ""` the default for `*-cu128` tags (every cheap GPU failed the 12.8+ filter but ran the image).
3. R2 in `start.sh` (`MODELS_SOURCE=r2`), `plan.md` Phase 3, so a pod in any datacenter boots from the bucket.
4. Serverless depth endpoint, `plan.md` Phase 1 (not started; the owner: later).
5. Optional: `--cache-lru` in `start.sh`, so running another workflow doesn't drop the 20GB model (~2 min reload).
6. Optional: a 24GB card (A5000) for pods; README "GPU and VRAM" has the order. With the cu128 image the CUDA 13 constraint that blocked it is gone.
7. The 1280/2500/3750 comparison on a sharp, full-size digital original (only a film scan and reduced copies so far).
8. Whether to add the reference pipeline (Uniform-base, Disparity-base, INT8 vs 4-bit), per `vathography.md` §14.4.
9. The tiled node's out-of-memory retry has never triggered; test it on a 16GB card with see-through at full size.
10. Video depth (the add-on's video support is in progress upstream): per-frame depth with temporal consistency, then vathopy per frame. Not started.

## Tiled and see-through findings (2026-10-27/28, RTX 4060 Ti 16GB, local ComfyUI v0.41.3 cu128)

- **5616 × 3712 film scan:** 12 tiles in 5 min 18 s (the plain 2496 px workflow: 26 s). Edges and fine structure much crisper than the plain map, flat areas smooth, no seams at 6 fringe cycles. 2496 px tiles gave nearly the same map (r = 0.999) in 4 min 24 s. For comparison only, the AGPL pack took 6 min 2 s (20 tiles) and left a grain-like texture on flat areas.
- **5640 × 3726 grainy black-and-white negative (4000 dpi scan), 12 tiles: failed.** At full size a tile is mostly grain and soft background; the model reads it as flat haze and loses whole grainy objects, so 5 tiles had no fit and others got scales near 0 (hard rectangles, smears). Blurring each tile first (Gaussian σ 2 px) brought the fits back (scales 0.9–1.2), but the merged map still lost thin, low-contrast structures the plain 2496 px map keeps, and was softer: rejected, not in the code. The plain map averages the grain out.
- **5616 × 1200 strip (panorama case):** 4 tiles of 2048 × 1200, 64 s, sharp, nothing at the top or bottom edges.
- **See-through:** on glass facades and aquarium tanks, the base model stops at the glass while the see-through model shows the structure, rooms or fish behind it. Ours agrees with the pack's map at r = 0.99.
- **6048 × 4024 dark aquarium, see-through:** 12 tiles in 4 min 36 s; the draft (24 s) is nearly as good there, since tiles see mostly water and noise. The split of 4 global-pass px left rings round fish; 64 doesn't (re-merged offline from the dump).
- **Memory:** the see-through model sits fully in VRAM (11.2GB); a 2000 × 2496 global pass once ran out of memory on the 16GB card right after another job, then ran in a fresh process.

## Phase 1 findings (2026-10-18, L40S 48GB in US-MO-1, `bench.py`)

How these were measured:
- **Sizes:** the actual inference size in pixels (width × height). These runs used the old megapixel-based resize; workflows and `bench.py` now take the long edge in pixels instead.
- **Time:** ComfyUI's execution time, warm (model already loaded).
- **Peak VRAM:** device-level, sampled from `/system_stats` every 0.1–0.2s, so it's a lower bound.
- **Input:** the official sample photo (`ev_suv_gold_desert_pavilion.png`, 896×1152, portrait). Everything larger than 896×1168 ran on an upscaled photo, so these runs say nothing about real high-resolution edge detail.

| Inference size (px) | Long edge | Peak VRAM | Time (warm) | Depth quality |
|---|---|---|---|---|
| 896 × 1168 | 1168 | 20.0 GiB | 1.3–1.4 s | Coherent |
| 1280 × 1648 | 1648 | 20.2 GiB | 2.1 s | |
| 1808 × 2320 | 2320 | 20.3 GiB | 4.2–4.8 s | Coherent; same structure as 1168 |
| 2208 × 2848 | 2848 | 20.9 GiB | 6.8–7.8 s | |
| 2704 × 3488 | 3488 | 21.8 GiB | 13.3 s | |
| 3136 × 4016 | 4016 | 34.8 GiB | 20.9 s | |
| 3616 × 4640 | 4640 | 38.7 GiB | 31.8 s | **Degraded** (see below) |

Time and memory depend on total pixel count, not just the long edge. A 3:2 landscape photo has about the same pixel count as these portrait runs at a long edge about 8% larger (e.g. 3:2 at 2500 px ≈ the 2320 row).

- **Cold start:** the first job after boot took 127 s, almost all of it loading the 20.5GB backbone from the network volume. After that it stays loaded (19.1 GiB idle).
- **Switching models:** each first switch between depth, normals and albedo took 31–45 s (LoRA re-patch). Normals and albedo outputs looked right.
- **Memory is far below `vathography.md` §14.3.** That table extrapolated the README's 4-bit reference-pipeline figures (29 GB at 2048×2048). The native INT8 path is about 20 GiB flat up to a ~3500 px long edge, then jumps by 4000 px. On 48GB, everything up to 4640 px fits.
- **What breaks at a 4640 px long edge:**
  - The window and the desert background are no longer clearly far.
  - The floor/wall depth gradient flattens.
  - The sunlight stripe on the wall leaks into the depth as a false ridge.

  This matches the training-resolution concern in `vathography.md` §10 (the backbone was trained around 1250 × 835).
- **Time:** 1168→2320 px (4× the pixels) took ~3.2× as long, and 2320→4640 px ~7×, so attention cost starts to dominate past ~2500 px.
- **Long-edge resize checked (2026-10-18):** 1280, 1250 and 1256 px all ran. 1250 and 1256 leave an odd number of latent columns after the 8 px VAE trim, and the model handles it.
- **Resolution test (2026-10-18), grainy 35mm film scan, 5552 × 3692:**
  - Ran at 1280, 2500 and 3750 px long edge (1280 × 848, 2496 × 1656, 3744 × 2488): 1.3 / 5.6 / 12.5 s, 20.0 / 20.5 / 21.7 GiB.
  - Compared at 100% (lower sizes upscaled to 3744 wide):
    - **1280:** clean shapes, but edges are visibly stair-stepped once enlarged, and fine detail barely registers.
    - **2500:** the best balance: fine detail resolves, edges are clean, and out-of-focus areas stay well separated from the background.
    - **3750:** slightly finer detail, but out-of-focus areas lose contrast and separation, and banding/grain-like texture appears in flat areas.
  - So depth quality falls off between 2500 and 3750 px, at least where the source is soft or grainy.
  - **Film-scan borders must be cropped first:** an uncropped scan's black frame came out as the nearest surface and used up part of the tonal range.
- **Small digital photos (1600 px copies, run at 1280):** night scenes with lasers, flares and LEDs left no false shapes; dark areas invisible in the RGB were recovered; patterned clothing stayed out of the depth; a fisheye curve carried into the depth. Most middle depth values landed in a narrow band (the near foreground takes the top), so the look's ramp has to spread them.
- **Panorama (16350 × 3828 PNG, 120 MB):**
  - The browser upload failed with HTTP 413 (Cloudflare's 100 MB limit); as a JPEG at quality 95 it was 8 MB.
  - Ran at 2600 / 4000 / 5400 px long edge (2600 × 608, 4000 × 936, 5400 × 1264): 3.8 / 5.1 / 9.7 s, ~20–21 GiB.
  - 4000 was the cleanest; 5400 added jagged edges. Geometry is plausible but not faithful across a full cylindrical sweep.
- **VRAM conclusion (2026-10-18): 24GB is enough.**
  - The weights take ~20 GiB constant, and jobs add ~0.5 GiB at 2500 px and ≤1.8 GiB up to ~3500 px.
  - Recommended: RTX A5000 24GB ($0.27/hr) in US-CA-2, which needs a new volume there, since US-MO-1 has no 24GB cards with network-volume support.
  - The move is not done. README "GPU and VRAM" has the order: prove CUDA 13 on an A5000 host before seeding, then bench, then delete the old pod and volume.
  - A5000 speed at 2500 px is still unmeasured (estimated 20–30 s).
- **`batch.sh` (2026-10-18):**
  - `--resize 2496` (default; was 2500 until 2026-10-18) or `4200` sets the long edge, then each side is snapped to a multiple of 16 (3:2 at 2496 → 2496 × 1664). Wide photos are flagged at 2500. `--via-s3` uploads into the pod's volume `input/` through S3, with an MD5 check.
  - ComfyUI's default cache keeps only the latest job's nodes, so running any other workflow between batches drops the 20 GB model, and the next job reloads it (~2 min). `--cache-lru` in `start.sh` would avoid that; not done.
  - Resizes locally with Pillow before upload, queues grouped by prediction, then runs `pull.sh`.
  - Tested end to end on 4 photos, including a 120MB panorama: 12 s of GPU time, all verified.
- **Twilight photos (2026-10-18, ~1000–1200 px stills, run at their own size):** clean depth throughout (probe wobble 0.0007–0.0015). twilight4 (high view down a corridor) shows faint regular terracing on the featureless floor at high fringe cycles, probably from the model's 16 px patches. The probe doesn't detect it (it measures irregular wobble).
- **Looks that worked well on the test photos:** B/W fringes at 2–4.5 cycles with smoothing + anti-aliasing, the linear ramp at 4, light toning (Ice, Copper). Save them as reusable looks from `vathopy gui`.

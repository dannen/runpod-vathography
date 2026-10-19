# AGENTS.md: Vathography depth (Marigold V2) on RunPod (persistent Pod)

For an AI agent picking up this repo. `README.md` is the runbook. This file records the current state and the decisions behind it. The research behind it is in the `vathography.md` notes, which aren't public (§14 is the RunPod plan).

## Decisions

- **Persistent Pod, not serverless.** Phase 1 (pod exploration, `vathography.md` §14.4) comes before Phase 2 (serverless), and work has stayed on the pod since. Don't add serverless tooling unless asked.
- **ComfyUI v0.41.3 + torch 2.11.0+cu130.** Marigold V2 is native from v0.36.0 (Comfy-Org/ComfyUI#16232). cu130 is required for comfy-kitchen's CUDA backend, which runs the int8_convrot backbone (`comfy/quant_ops.py` disables it below CUDA 13). Pods are restricted to CUDA 13.0 hosts via `allowedCudaVersions`. comfy-kitchen's kernels need compute capability ≥ 8.0 (Ampere or newer).
- **One custom node, ours:** `custom_nodes/vathography` (Vathography Output Name) names outputs after the photo. No third-party nodes (`vathography.md` §7 warns against community Marigold nodes).
- **Volume contents: depth, normals and albedo** (all three Marigold V2 predictions), about 26.4GB. Only depth/Log-stage2 is packaged for ComfyUI.
- **`vathography_*.json` is generated, never hand-edited.** `make_workflows.py` builds them from the untouched official templates:
  - Resize by long edge in pixels (sizes are always given in pixels, never megapixels; default 1280), each side rounded to the nearest multiple of 16 (PrimitiveInt + GetImageSize + two ComfyMathExpression + ImageScale, no crop). That avoids the VAE's 8 px center crop and the DiT's circular padding, so the map is exactly the scaled photo's size.
  - 32-bit float EXR save, named `<photo>_<prediction>_00001.exr` by our node.
  - A Result preview node; Image Compare removed (it looked blank); every main-canvas node pinned.
  - `bench.py` and `batch.py` use the same sizing rule.
  - `start.sh` seeds workflows with `--ignore-existing`, so a running pod keeps old copies: push new ones with `POST /api/userdata/workflows%2F<name>?overwrite=true`.
- **No raw `.npy`.** The output is the post-processed map: per-image min/max normalized, near = bright, after the VAE decode clamped it to 0–1. `vathopy prep` needs `--kind near-bright` for these files.
- **Seeding is local** (`populate-volume.sh`, conservative S3 multipart settings), not pod-side.
- **GPU: 48GB is plenty; 24GB would do** (measured: ~20 GiB of weights plus ≤1.8 GiB per job up to ~3500 px). Prefer cheaper, slower cards: depth jobs take seconds while the pod stays up for minutes.
- **The vathopy GUI runs on the pod** (port 8770), in its own venv so its numpy/OpenCV/OpenEXR can't affect ComfyUI. `sync-vathopy.sh` copies `vathopy` (https://github.com/dannen/vathopy, checked out next to this repo as `../vathopy`) into the gitignored `vathopy-src/` before each build and records the commit in `VATHOPY_COMMIT`.
- **Password on the GUI:** RunPod's proxy URLs are public. `VATHOPY_GUI_AUTH` (user:password) comes from the pod env; keep the credentials in a local file outside the repo (mode 600; `build.md` uses `~/.api/runpod/vathopy-gui.auth`). `create-pod.sh --update` replaces the whole env, so **always pass `--env "VATHOPY_GUI_AUTH=$(cat ~/.api/runpod/vathopy-gui.auth)"`** (see `build.md`).
- `models.txt` is the only list of models (`start.sh` and `populate-volume.sh` both read it).

## Current state (2026-10-18)

- **Pod** `vathography-comfyui`, **L40S 48GB** ($1.09/hr) in US-MO-1. The A40 ($0.59) was the first choice but had no CUDA 13 hosts.
  - Image `YOUR_DOCKERHUB_USER/vathography-comfyui:ed23784` (vathopy `9769d79`), ports 8188/http, 8770/http, 22/tcp, env `VATHOPY_GUI_AUTH`.
  - vathopy has moved on since that image (creative focus, MIST, highlights, reach, lean/angle, lock, Download PNG): rebuild and roll (`build.md`) to bring them to the pod's GUI.
  - ComfyUI: `https://<pod-id>-8188.proxy.runpod.net` (no auth). `pull.sh` finds the pod ID. Verified: ComfyUI 0.41.3, torch cu130, `comfy_kitchen backend cuda: available: True`.
  - vathopy GUI: `https://<pod-id>-8770.proxy.runpod.net` (password). Reads `/workspace/output/vathography/`, saves to `/workspace/favorites/`. Verified through the proxy: 401 without the password, list, full render (213 ms), save and download.
  - Earlier images: `1752c0b`, `47d9975`, `d8333d8`, `gui-2026-10-18`, `c1eb935`.
- **Volume** `vathography-storage` (50GB, US-MO-1): the 10 models (26.38GB), plus `output/`, `input/` and `favorites/`. `input/` and `output/` were purged on 2026-10-18 after everything was pulled.
- **Local tools:**
  - `batch.sh DIR`: resizes locally (`--resize 2500|4200`, sides snapped to ×16), uploads (proxy or `--via-s3`), queues grouped by prediction, waits, then runs `pull.sh`.
  - `pull.sh`: finds the running pod (cached 30 min) and downloads results to `~/Pictures/vathography/depth/`, plus the GUI's `favorites/` saves to `~/Pictures/vathography/favorites/`. Verified against the volume's S3 checksums (MD5, SHA-256), falls back to S3 if the pod can't deliver, never overwrites (`-001`), keeps MD5-aware records (ComfyUI reuses names after deletions). Also `--from-volume`, `--match`, `--verify-existing`, `--clear-volume`, `--clear-inputs`.
  - `bench.py`: time and peak VRAM per size.
- **The US-CA-2 24GB test was cancelled** (2026-10-18) and its volume deleted. A5000, RTX 6000 Ada and 4090 had no CUDA 13 hosts there for an hour of retries.
- **Seeding gotcha (US-MO-1 S3 gateway):**
  - `CompleteMultipartUpload` failed on large files with `InvalidPart`/`NoSuchUpload` or a read timeout; in one case the object had been written in full anyway.
  - The 20.5GB backbone was verified by sampled ranged `get-object` comparisons (43 ranges) instead of a third upload.
  - A killed upload leaves its parts under `.s3compat_uploads/`; clear them with `aws s3api abort-multipart-upload` (retry with `--cli-read-timeout 600` if it times out).
- **ComfyUI's `lanczos` upscale goes through 8-bit PIL.** Harmless where the workflows use it (on the input photo), but never put a lanczos resize after the model.

## Open items

1. Optional: move to a 24GB card (A5000 in US-CA-2) if CUDA 13 hosts appear; README "GPU and VRAM" has the order (prove the host, seed, bench, then delete the old pod and volume). The first attempt was cancelled.
2. Optional: `--cache-lru` in `start.sh`, so running another workflow doesn't drop the 20GB model (~2 min reload).
3. The 1280/2500/3750 comparison on a sharp, full-size digital original (only a film scan and reduced copies so far).
4. Whether to add the reference pipeline (Uniform-base, Disparity-base, INT8 vs 4-bit), per `vathography.md` §14.4.

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
  - `--resize 2500` (default) or `4200` sets the long edge, then each side is snapped to a multiple of 16 (3:2 at 2500 → 2496 × 1664). Wide photos are flagged at 2500. `--via-s3` uploads into the pod's volume `input/` through S3, with an MD5 check.
  - ComfyUI's default cache keeps only the latest job's nodes, so running any other workflow between batches drops the 20 GB model, and the next job reloads it (~2 min). `--cache-lru` in `start.sh` would avoid that; not done.
  - Resizes locally with Pillow before upload, queues grouped by prediction, then runs `pull.sh`.
  - Tested end to end on 4 photos, including a 120MB panorama: 12 s of GPU time, all verified.
- **Twilight photos (2026-10-18, ~1000–1200 px stills, run at their own size):** clean depth throughout (probe wobble 0.0007–0.0015). twilight4 (high view down a corridor) shows faint regular terracing on the featureless floor at high fringe cycles, probably from the model's 16 px patches. The probe doesn't detect it (it measures irregular wobble).
- **Looks that worked well on the test photos:** B/W fringes at 2–4.5 cycles with smoothing + anti-aliasing, the linear ramp at 4, light toning (Ice, Copper). Save them as reusable looks from `vathopy gui`.

#!/usr/bin/env python3
"""
Run a folder of photos through Marigold V2 on the pod, then download the
results with pull.py (checksum-verified).

    ./batch.sh ~/Pictures/some-folder                  # depth for every photo, 2496 px long edge
    ./batch.sh DIR --resize 4200                       # 4200 px long edge (panoramas)
    ./batch.sh DIR --predictions depth normals albedo  # all three
    ./batch.sh DIR --predictions depth-tiled           # full-resolution tiled depth
    ./batch.sh DIR --predictions depth-see-through     # tiled, depth behind glass
    ./batch.sh DIR --predictions depth-tiled --draft   # quick look: the global pass alone
    ./batch.sh DIR --via-s3                            # upload through the volume's S3 gateway
    ./batch.sh DIR --dry-run                           # show sizes, upload nothing

Settings (from the Phase 1 tests in AGENTS.md):
  - Size: --resize 2496 (default, the same as the workflows) or 4200 sets
    the long edge in pixels. ~2500 gave the best depth on regular photos (it
    degraded by 3750); 4200 suits panoramas, where 2496 leaves too few pixels
    top to bottom (a 4.3:1 panorama at 2496 is 2496 x 576), so wide photos
    are flagged.
    Photos smaller than the target are never enlarged. Each side is then
    rounded to the nearest multiple of 16 (2496 at 3:2 -> 2496 x 1664), so
    the model neither crops nor pads and the depth map is exactly the
    uploaded size, covering the full frame.
  - The resize happens here, before upload (EXIF rotation applied, Lanczos,
    JPEG quality 95). Uploads are then a few MB instead of the original
    10-120 MB, which also stays under the 100 MB upload limit (Cloudflare
    and ComfyUI both cap uploads there). The pod's resize node gets the
    same long edge, so it doesn't resample again.
  - Upload route: ComfyUI's /upload/image through RunPod's proxy by default.
    --via-s3 writes each photo straight into the pod's volume (input/)
    through RunPod's S3 gateway instead, and checks the MD5 the gateway
    reports against the local file. That avoids the proxy and its limit,
    and needs the aws CLI and ~/.api/runpod/s3-credentials, like pull.py.
  - depth-tiled and depth-see-through use our Vathography Tiled Depth node:
    the photo goes up at its own size (upright, JPEG quality 95), and the
    node runs a global pass (up to 2496 px, never enlarged) plus
    full-resolution tiles, so --resize doesn't apply. --draft runs the
    global pass alone (about 20 s a photo) and names the maps
    <photo>_<prediction>-draft_NNNNN.exr. depth-see-through needs the
    see-through model files on the volume (models-see-through.txt).
  - Jobs are queued grouped by prediction (all depth, then normals, then
    albedo), so the model switch (30-45 s per LoRA) happens once each.
  - Output EXRs are saved on the volume as vathography/<photo>_<prediction>_NNNNN.exr
    (e.g. IMG_0001_depth_00001.exr), the same as the browser workflows.

Photos with a result already in the output folder (<photo>_<prediction>_*.exr)
are skipped; --force reruns them. Uploads go to input/batch-<time>/ on the
volume, so names can't collide with earlier runs.

The pod target is found and cached the same way as pull.py (30 minutes).
When every job has finished, pull.py runs to download and verify all new
results (--no-pull skips that). Python 3 + Pillow.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import uuid
from pathlib import Path

import bench
import pull

try:
    from PIL import Image, ImageOps
except ImportError:
    sys.exit("ERROR: batch.py needs Pillow (pip install Pillow).")

EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}
RESIZE_CHOICES = (2496, 4200)
WIDE = 2.0  # aspect ratio from which --resize 2496 gets a panorama warning
PREDICTION_ORDER = ["depth", "normals", "albedo", "depth-tiled", "depth-see-through"]
FULL_SIZE = ("depth-tiled", "depth-see-through")  # the tiled node takes the photo at its own size


def plan_size(w, h, long_edge):
    """Target (w, h) for a long edge in px, each side rounded to the nearest
    multiple of 16 (never above the photo's own size). Multiples of 16 avoid
    the VAE's crop and the model's circular padding, so the depth map is
    exactly this size and covers the full frame. The rounding stretches the
    shape by at most 8 px per side, which scaling the map back undoes."""
    scale = min(long_edge, max(w, h)) / max(w, h)

    def snap(v, limit):
        r = max(16, round(v / 16) * 16)
        return r if r <= limit else max(16, limit // 16 * 16)

    return snap(w * scale, w), snap(h * scale, h)


def prepare(path, size):
    """The photo upright, resized to `size`, as JPEG bytes."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if im.size != size:
            im = im.resize(size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=95, subsampling=0)
        return buf.getvalue()


def upload(url, data, name, subfolder):
    boundary = uuid.uuid4().hex
    parts = []
    for field, value in (("subfolder", subfolder), ("overwrite", "true"), ("type", "input")):
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n".encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                 "Content-Type: image/jpeg\r\n\r\n".encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    for attempt in range(3):
        try:
            out = json.loads(bench.request(url + "/upload/image", body,
                                           {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=300))
            return f"{out['subfolder']}/{out['name']}" if out.get("subfolder") else out["name"]
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code < 500:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("upload failed after 3 tries")


def upload_s3(vol, data, name, subfolder):
    """Write the photo into the volume's input/ folder; returns its LoadImage name."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_bytes(data)
        md5 = hashlib.md5(data).hexdigest()
        for attempt in range(3):
            etag = vol.put(f"input/{subfolder}/{name}", path)
            if etag == md5:
                return f"{subfolder}/{name}"
            print(f"    {name}: gateway reports {etag!r}, expected MD5 {md5}; try {attempt + 1}/3")
    raise RuntimeError("upload never matched its MD5")


def safe(stem):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "image"


def queue(url, image_name, long_edge, prediction, stem, draft=False):
    tag = f"{prediction}-draft" if draft and prediction in FULL_SIZE else prediction
    graph = bench.build_prompt(image_name, long_edge, prediction, f"vathography/{safe(stem)}_{tag}",
                               mode="draft" if draft else "full")
    graph["24"] = {"class_type": "PreviewImage", "inputs": {"images": ["22", 0]}}
    body = json.dumps({"prompt": graph}).encode()
    try:
        return json.loads(bench.request(url + "/prompt", body, {"Content-Type": "application/json"}))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"prompt rejected: HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")


def wait(url, pending, timeout):
    """Poll until every prompt has finished. {prompt_id: (ok, detail)}."""
    done, start = {}, time.time()
    while len(done) < len(pending) and time.time() - start < timeout:
        for pid, label in pending.items():
            if pid in done:
                continue
            try:
                h = json.loads(pull.fetch(f"{url}/history/{pid}", timeout=30))
            except Exception:
                continue  # proxy hiccup: the job keeps running regardless
            status = h.get(pid, {}).get("status", {})
            if status.get("completed") is None:
                continue
            msgs = status.get("messages", [])
            ts = {m[0]: m[1].get("timestamp") for m in msgs}
            if status.get("status_str") == "success":
                secs = (ts.get("execution_success", 0) - ts.get("execution_start", 0)) / 1000
                done[pid] = (True, f"{secs:.1f}s")
            else:
                err = [m[1] for m in msgs if m[0] == "execution_error"]
                done[pid] = (False, (err[0].get("exception_message", "") if err else "failed").strip()[:200])
            ok, detail = done[pid]
            print(f"  [{len(done)}/{len(pending)}] {'done' if ok else 'FAILED'}  {label}  ({detail})", flush=True)
        if len(done) < len(pending):
            time.sleep(2)
    return done


def main():
    ap = argparse.ArgumentParser(description="Run a folder of photos through Marigold V2 on the pod, then pull the results.")
    ap.add_argument("folder", type=Path, help="Folder of photos (jpg, png, tif, webp, bmp)")
    ap.add_argument("--predictions", nargs="+", choices=PREDICTION_ORDER, default=["depth"],
                    help="What to estimate (default: depth)")
    ap.add_argument("--resize", type=int, choices=RESIZE_CHOICES, default=2496,
                    help="Long edge in px for the uploads: 2496 (default, regular photos; the workflows' default "
                         "too) or 4200 (panoramas)")
    ap.add_argument("--draft", action="store_true",
                    help="depth-tiled / depth-see-through: the global pass alone, no tiles (a quick look; "
                         "saved as <photo>_<prediction>-draft_*.exr)")
    ap.add_argument("--via-s3", action="store_true",
                    help="Upload into the pod's volume through RunPod's S3 gateway instead of the web proxy")
    ap.add_argument("--recursive", action="store_true", help="Include photos in subfolders")
    ap.add_argument("--force", action="store_true", help="Rerun photos that already have a result in --out")
    ap.add_argument("--dry-run", action="store_true", help="Show what would run, at what size; upload nothing")
    ap.add_argument("--out", type=Path, default=pull.DEFAULT_OUT, help=f"Where results go (default {pull.DEFAULT_OUT})")
    ap.add_argument("--timeout", type=float, default=3600, help="Max seconds to wait for the queue (default 3600)")
    ap.add_argument("--no-pull", action="store_true", help="Don't run pull.py afterwards")
    ap.add_argument("--name", default="vathography-comfyui", help="Pod name (default vathography-comfyui)")
    ap.add_argument("--url", help="Pod proxy URL; skips the lookup")
    args = ap.parse_args()

    folder = args.folder.expanduser()
    if not folder.is_dir():
        sys.exit(f"ERROR: {folder} isn't a folder.")
    files = sorted(p for p in (folder.rglob("*") if args.recursive else folder.iterdir())
                   if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith("."))
    if not files:
        sys.exit(f"No photos in {folder}.")
    out = args.out.expanduser()
    preds = [p for p in PREDICTION_ORDER if p in args.predictions]

    # Plan: (photo, size, [predictions still needed]).
    plan = []
    for f in files:
        try:
            with Image.open(f) as im:
                w, h = ImageOps.exif_transpose(im).size
        except Exception as e:
            print(f"  skip {f.name}: can't read it ({e})")
            continue
        size = plan_size(w, h, args.resize)
        tag = lambda p: f"{p}-draft" if args.draft and p in FULL_SIZE else p
        todo = [p for p in preds if args.force or not list(out.glob(f"{f.stem}_{tag(p)}_*.exr"))]
        plan.append((f, (w, h), size, todo))
    # tiled jobs: the photo as it is (the node finds its own sizes)
    jobs = [(f, (w, h) if p in FULL_SIZE else size, p) for f, (w, h), size, todo in plan for p in todo]

    print(f"{len(plan)} photo(s) in {folder}, long edge {args.resize} px:")
    wide = 0
    for f, (w, h), (tw, th), todo in plan:
        what = ", ".join(f"{p} (full size)" if p in FULL_SIZE else p for p in todo) if todo else "already done (--force to rerun)"
        flag = ""
        if args.resize < max(RESIZE_CHOICES) and max(w, h) / min(w, h) >= WIDE and max(w, h) > args.resize:
            flag, wide = "  (wide: --resize 4200 suits panoramas)", wide + 1
        print(f"  {f.name:50s} {w} x {h}  ->  {tw} x {th}   {what}{flag}")
    if wide:
        print(f"{wide} wide photo(s) would run at only {args.resize} px across; consider --resize 4200 for those.")
    if not jobs:
        print("Nothing to run.")
        return
    if args.dry_run:
        print(f"Dry run: {len(jobs)} job(s) would be queued.")
        return

    if args.url:
        url = args.url.rstrip("/")
        pod_id = pull.urllib.parse.urlparse(url).hostname.split("-")[0]
    else:
        pod_id, url = pull.resolve_target(args.name, 30, False)
    subfolder = time.strftime("batch-%Y%m%d-%H%M%S")
    vol = None
    if args.via_s3:
        try:
            vol = pull.Volume(*pull.volume_for_pod(pod_id))
        except Exception as e:
            sys.exit(f"ERROR: --via-s3 can't reach the pod's volume: {e}")

    # Upload each photo once, then queue grouped by prediction.
    uploaded, failed = {}, []
    route = f"volume {vol.label} via S3" if vol else "the pod (web proxy)"
    print(f"Uploading {len({(f, size) for f, size, _ in jobs})} photo(s) to input/{subfolder}/ on {route} ...")
    for f, size, _ in jobs:
        if (f, size) in uploaded:
            continue
        try:
            data = prepare(f, size)
            name = f"{safe(f.stem)}.jpg"
            if any(g == f for g, _ in uploaded):                 # the same photo at a second size
                name = f"{safe(f.stem)}_{size[0]}px.jpg"
            uploaded[(f, size)] = upload_s3(vol, data, name, subfolder) if vol else upload(url, data, name, subfolder)
            print(f"  {f.name}  ({len(data) / 1e6:.1f} MB" + (", MD5 OK" if vol else "") + ")")
        except Exception as e:
            print(f"  FAILED upload {f.name}: {e}")
            failed.append(f.name)
    pending = {}
    for pred in preds:
        for f, size, p in jobs:
            if p != pred or (f, size) not in uploaded:
                continue
            try:
                pid = queue(url, uploaded[(f, size)], max(size), pred, f.stem, args.draft)
                pending[pid] = f"{f.name} {pred} {size[0]} x {size[1]}"
            except Exception as e:
                print(f"  FAILED to queue {f.name} {pred}: {e}")
                failed.append(f"{f.name} {pred}")
    print(f"Queued {len(pending)} job(s). Waiting (the first job after a pod start also loads the model, ~2 min) ...")

    t0 = time.time()
    done = wait(url, pending, args.timeout)
    ok = sum(1 for v in done.values() if v[0])
    unfinished = len(pending) - len(done)
    failed += [pending[p] for p, v in done.items() if not v[0]]
    print(f"{ok} of {len(pending)} job(s) succeeded in {time.time() - t0:.0f}s"
          + (f"; {unfinished} still running after --timeout" if unfinished else "") + ".")

    rc = 0
    if not args.no_pull and ok:
        print("\nDownloading results (pull.py) ...", flush=True)
        cmd = [sys.executable, str(Path(__file__).with_name("pull.py")), "--out", str(out)]
        if args.url:
            cmd += ["--url", url]
        rc = subprocess.call(cmd)
    if failed:
        print("\nFailed: " + "; ".join(failed))
    sys.exit(1 if (failed or unfinished or rc) else 0)


if __name__ == "__main__":
    main()

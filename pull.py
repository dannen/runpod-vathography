#!/usr/bin/env python3
"""
Download finished Marigold V2 results from the pod, and check every download
against the network volume's own checksums. Python 3 stdlib (+ aws CLI).

    ./pull.sh                      # pull everything new into ~/Pictures/vathography/depth
    ./pull.sh --out DIR            # somewhere else
    ./pull.sh --list               # show what's on the pod, download nothing
    ./pull.sh --refresh            # look the pod up again now, ignoring the cache
    ./pull.sh --from-volume        # read the volume's S3 bucket instead of the pod
    ./pull.sh --verify-existing    # re-check files already downloaded
    ./pull.sh --clear-volume       # delete results from the volume once safely pulled
    ./pull.sh --clear-inputs       # delete uploaded photos (input/) from the volume

Crops: the vathopy GUI's Crop section saves <map>-cropped.exr next to the depth maps on the pod
(output/vathography/), outside ComfyUI's history. Pulling from the pod fetches those too, named
after the local name of the map they were cut from (photo_depth_2496px-cropped.exr), with the
same checks and never-overwrite rule; a crop saved again with new contents arrives as -001, ...
They're listed from the volume, or with --no-verify from the pod's GUI (port 8770).
--from-volume pulls them like any other EXR under output/.

Every pull (from the pod or --from-volume) also downloads what the vathopy GUI saved on the
pod: favorites/ on the volume (PNGs and looks/*.json), into ~/Pictures/vathography/favorites
(--favourites-out), with the same checks and never-overwrite rule and its own .pulled.json and
SHA256SUMS there. It reads the volume through S3, so it needs no GUI password and no running
GUI. --no-favourites skips it.

Target: the pod is recreated from time to time, so its ID (and proxy URL)
isn't fixed. The running pod named --name (default vathography-comfyui) is
looked up through RunPod's REST API, and the answer is cached in
~/.cache/runpod-vathography/target.json for 30 minutes (--cache-minutes).
After that, or as soon as the cached pod stops answering, it's looked up
again. --url skips the lookup entirely.

Results: every successful job in the pod's ComfyUI history with an EXR
output. Each is saved as <photo>_<prediction>_<long edge>px.exr, plus
<same>_preview.png when the job has a Result preview (workflows from 7c5080c
on; previews live in ComfyUI's temp folder, which a restart clears). Runs
made by bench.py (prefix vathography/bench_) are skipped unless
--include-bench.

Checksums: ComfyUI writes its outputs onto the network volume, which is also
an S3 bucket, so the volume can vouch for every file independently of the
path it was downloaded through. RunPod's S3 gateway reports each file's MD5
as its ETag in the bucket listing (one call covers every file), and returns
the full-file SHA-256 with a whole-object `get-object --checksum-mode
ENABLED` (checked 2026-10-18: both matched local hashes exactly; a ranged
get returns the ETag but no SHA-256).
  - Pulled through the pod: size and MD5 must match the volume's listing.
  - Pulled with --from-volume: size, MD5 and SHA-256 must all match.
  - A failed or mismatched download is tried again (3 tries). If the pod
    can't deliver a good copy (e.g. it's restarting), the same file is
    fetched from the volume through S3 instead. A file that never matches
    is reported as failed and not recorded, so the next run retries it.
  - <out>/SHA256SUMS lists every verified file, so `sha256sum -c SHA256SUMS`
    re-checks the folder locally at any time.
  - If the volume can't be reached (no aws CLI or credentials, or the S3
    gateway is down) the run stops; --no-verify downloads unchecked, and
    --verify-existing can check those files later.

Already-pulled jobs are recorded per pod in <out>/.pulled.json, so rerunning
only downloads new ones. Every pull also records the volume key each file
came from (".outputs"), which is how --from-volume and --verify-existing
match volume files to photo-named local files. Nothing in the output folder
is ever overwritten: if a different file (different checksum) already has
the target name, the new one gets -001, -002, ... (photo_depth_1280px-001.exr);
an identical file is just recorded. Previews follow the same rule.

ComfyUI keeps history in memory, so a pod restart forgets it. The EXRs are
still on the volume under output/, and --from-volume reads them straight
from the bucket (no pod needed). There they only have ComfyUI's names
(vathography/depth_00007.exr -> depth_00007.exr); files already pulled under
their photo name are skipped.

Pulling from a pod checks against the volume attached to that pod (asked
from RunPod). --from-volume, --verify-existing and --clear-volume look the
volume up by name (--volume-name, default vathography-storage);
--volume-id/--datacenter skip either lookup. S3 access
needs the aws CLI and ~/.api/runpod/s3-credentials ([runpods3] profile).
Every call is `aws s3api list-objects-v2`/`get-object`: RunPod's S3 gateway
rejects HeadObject, which `aws s3 cp` relies on.

Clearing the volume: --clear-volume deletes .exr and .png files under
output/ on the volume (never models or inputs). By default it only deletes
files with a local copy that matches the volume's MD5 right now; everything
else (not pulled yet, bench runs, PNGs from other workflows) is listed and
kept unless --include-unpulled. It shows the list and asks first; --yes
skips the question. ComfyUI's history still names deleted files, so a later
pull records those jobs as gone instead of retrying them.

Clearing uploads: --clear-inputs deletes everything under input/ on the
volume: photos uploaded in the browser and batch.sh's input/batch-*/
folders. It keeps ComfyUI's own example.png and empty "_" placeholder files, and
never touches models or output/. A queued or running job still needs its
photo, so it refuses while the pod's queue isn't empty (when the pod is
running). It shows the list and asks first; --yes skips the question.

Cloudflare in front of RunPod's proxy rejects urllib's default User-Agent
(error 1010), so every request sets one, and every request has a timeout:
the proxy occasionally stalls a download indefinitely.
"""

import argparse
import base64
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import podauth

UA = {"User-Agent": "runpod-vathography-pull/1.0"}
REST_BASE = "https://rest.runpod.io/v1"
CACHE_FILE = Path.home() / ".cache" / "runpod-vathography" / "target.json"
DEFAULT_OUT = Path.home() / "Pictures" / "vathography" / "depth"
DEFAULT_FAV_OUT = Path.home() / "Pictures" / "vathography" / "favorites"
BENCH_PREFIX = "bench_"
TRIES = 3


# ── HTTP and the pod ─────────────────────────────────────────────────────────

def fetch(url, headers=None, timeout=60, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={**UA, **podauth.headers_for(url), **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 401 and podauth.is_pod_url(url):
                raise SystemExit(f"{url}: {podauth.HINT}") from None
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def api_key():
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        sys.exit("ERROR: RUNPOD_API_KEY not set (run through pull.sh).")
    return key


def proxy_url(pod_id):
    return f"https://{pod_id}-8188.proxy.runpod.net"


def alive(url):
    try:
        json.loads(fetch(url + "/system_stats", timeout=20, retries=1))
        return True
    except Exception:
        return False


def lookup_pod(name):
    pods = json.loads(fetch(f"{REST_BASE}/pods?" + urllib.parse.urlencode({"name": name}),
                            {"Authorization": f"Bearer {api_key()}"}, timeout=30))
    running = [p for p in pods if p.get("name") == name and p.get("desiredStatus") == "RUNNING"]
    if not running:
        states = ", ".join(f"{p['id']} ({p.get('desiredStatus')})" for p in pods if p.get("name") == name)
        sys.exit(f"ERROR: no running pod named {name!r}" + (f"; found: {states}" if states else "")
                 + ". For results already on the volume, use --from-volume.")
    if len(running) > 1:
        sys.exit(f"ERROR: {len(running)} running pods named {name!r}: "
                 + ", ".join(p["id"] for p in running) + ". Pass --url to pick one.")
    return running[0]["id"]


def resolve_target(name, cache_minutes, refresh):
    """(pod_id, url). Uses the cache while it's fresh and the pod answers."""
    if not refresh and CACHE_FILE.exists():
        try:
            c = json.loads(CACHE_FILE.read_text())
            age = time.time() - c["resolved_at"]
            if c.get("name") == name and age < cache_minutes * 60:
                if alive(c["url"]):
                    print(f"Target: pod {c['pod_id']} (cached {age / 60:.0f} min ago)")
                    return c["pod_id"], c["url"]
                print(f"Cached pod {c['pod_id']} isn't answering; looking it up again.")
        except (ValueError, KeyError):
            pass
    pod_id = lookup_pod(name)
    url = proxy_url(pod_id)
    if not alive(url):
        sys.exit(f"ERROR: pod {pod_id} is running but ComfyUI at {url} isn't answering yet "
                 "(still booting or loading?). Try again shortly.")
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"name": name, "pod_id": pod_id, "url": url, "resolved_at": time.time()}))
    print(f"Target: pod {pod_id} (looked up)")
    return pod_id, url


def jobs(url, include_bench):
    """Successful jobs with an EXR, oldest first."""
    history = json.loads(fetch(url + "/history", timeout=120))
    found = []
    for prompt_id, entry in history.items():
        status = entry.get("status", {})
        if status.get("status_str") != "success":
            continue
        graph = entry["prompt"][2]
        photos = [n["inputs"].get("image") for n in graph.values() if n.get("class_type") == "LoadImage"]
        preds = [n["inputs"].get("prediction") for n in graph.values() if n.get("class_type") == "MarigoldV2PostProcess"]
        exrs, previews = [], []
        for out in entry.get("outputs", {}).values():
            for files in out.values():
                for f in files if isinstance(files, list) else []:
                    if not isinstance(f, dict) or "filename" not in f:
                        continue
                    if f["filename"].endswith(".exr"):
                        exrs.append(f)
                    elif f.get("type") == "temp" and not f["filename"].startswith("comfy.compare."):
                        previews.append(f)
        if not exrs:
            continue
        if not include_bench and all(e["filename"].startswith(BENCH_PREFIX) for e in exrs):
            continue
        started = [m[1].get("timestamp", 0) for m in status.get("messages", []) if m[0] == "execution_start"]
        found.append({
            "prompt_id": prompt_id,
            "started": started[0] if started else 0,
            "photo": photos[0] if photos else "unknown",
            "prediction": preds[0] if preds else "depth",
            "exr": exrs[0],
            "preview": previews[0] if previews else None,
        })
    return sorted(found, key=lambda j: j["started"])


def view_url(url, f):
    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""),
                                "type": f.get("type", "output")})
    return f"{url}/view?{q}"


def output_key(f):
    """Where a ComfyUI output lives on the volume, e.g. output/vathography/depth_00007.exr."""
    sub = f.get("subfolder", "")
    return "output/" + (f"{sub}/" if sub else "") + f["filename"]


def target_name(job):
    meta = job["exr"].get("metadata", {})
    long_edge = max(meta.get("width") or 0, meta.get("height") or 0)
    stem = Path(job["photo"]).stem
    size = f"_{long_edge}px" if long_edge else ""
    return f"{stem}_{job['prediction']}{size}"


# ── The volume, through RunPod's S3 gateway ──────────────────────────────────

def volume_for_pod(pod_id):
    """(volume_id, datacenter) of the network volume attached to a pod."""
    headers = {"Authorization": f"Bearer {api_key()}"}
    pod = json.loads(fetch(f"{REST_BASE}/pods/{pod_id}", headers, timeout=30))
    vol_id = pod.get("networkVolumeId") or (pod.get("networkVolume") or {}).get("id")
    if not vol_id:
        raise RuntimeError(f"pod {pod_id} has no network volume attached")
    vol = json.loads(fetch(f"{REST_BASE}/networkvolumes/{vol_id}", headers, timeout=30))
    return vol_id, vol["dataCenterId"]


def lookup_volume(name):
    vols = json.loads(fetch(f"{REST_BASE}/networkvolumes", {"Authorization": f"Bearer {api_key()}"}, timeout=30))
    match = [v for v in vols if v.get("name") == name]
    if len(match) != 1:
        sys.exit(f"ERROR: expected one network volume named {name!r}, found {len(match)}. "
                 "Pass --volume-id and --datacenter.")
    return match[0]["id"], match[0]["dataCenterId"]


class Volume:
    """The network volume as an S3 bucket, via `aws s3api` only (no HeadObject)."""

    def __init__(self, volume_id, datacenter):
        if not shutil.which("aws"):
            raise RuntimeError("the aws CLI isn't installed")
        creds = Path.home() / ".api" / "runpod" / "s3-credentials"
        self.env = dict(os.environ)
        self.base = ["aws", "--endpoint-url", f"https://s3api-{datacenter.lower()}.runpod.io",
                     "--region", datacenter.lower()]
        if "AWS_ACCESS_KEY_ID" not in os.environ:
            if not creds.exists():
                raise RuntimeError(f"no S3 credentials (set AWS_* or create {creds})")
            self.env["AWS_SHARED_CREDENTIALS_FILE"] = str(creds)
            self.base += ["--profile", "runpods3"]
        self.bucket = volume_id
        self.label = f"{volume_id} ({datacenter})"

    def run(self, args, timeout):
        r = None
        for attempt in range(3):
            try:
                r = subprocess.run(self.base + args, env=self.env, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                r = None
            if r is not None and r.returncode == 0:
                return r.stdout
            time.sleep(2 * (attempt + 1))
        detail = r.stderr.strip() if r is not None else f"timed out after {timeout}s"
        raise RuntimeError(f"aws {' '.join(args[:2])} failed: {detail}")

    def list_exrs(self):
        """{key: {"size", "md5", "modified"}} for every .exr under output/."""
        return self.list_outputs((".exr",))

    def list_outputs(self, suffixes, prefix="output/"):
        """{key: {"size", "md5", "modified"}} for every file under `prefix` ending
        in one of `suffixes` (all files if suffixes is empty)."""
        out, token = {}, None
        while True:
            args = ["s3api", "list-objects-v2", "--bucket", self.bucket, "--prefix", prefix, "--output", "json"]
            if token:
                args += ["--continuation-token", token]
            page = json.loads(self.run(args, timeout=120) or "{}")
            for o in page.get("Contents", []):
                if not suffixes or o["Key"].lower().endswith(suffixes):
                    out[o["Key"]] = {"size": o["Size"], "md5": o.get("ETag", "").strip('"'), "modified": o["LastModified"]}
            token = page.get("NextContinuationToken")
            if not token:
                return out

    def get(self, key):
        """(bytes, base64 SHA-256 the gateway reports for the whole object, or None)."""
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "obj"
            meta = json.loads(self.run(["s3api", "get-object", "--bucket", self.bucket, "--key", key,
                                        "--checksum-mode", "ENABLED", str(dest)], timeout=300) or "{}")
            return dest.read_bytes(), meta.get("ChecksumSHA256")


    def put(self, key, path):
        """Upload a file; returns the ETag the gateway reports (the MD5 for a single PUT)."""
        meta = json.loads(self.run(["s3api", "put-object", "--bucket", self.bucket, "--key", key,
                                    "--body", str(path)], timeout=300) or "{}")
        return meta.get("ETag", "").strip('"')

    def delete(self, key):
        self.run(["s3api", "delete-object", "--bucket", self.bucket, "--key", key], timeout=60)


def open_volume(args, pod_id=None):
    """--volume-id/--datacenter if given, else the pod's own volume (when pulling
    from a pod), else the volume named --volume-name."""
    if args.volume_id and args.datacenter:
        return Volume(args.volume_id, args.datacenter)
    if pod_id:
        return Volume(*volume_for_pod(pod_id))
    return Volume(*lookup_volume(args.volume_name))


# ── Checksums ────────────────────────────────────────────────────────────────

def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def mismatch(data, ref, sha256_b64=None):
    """Why `data` doesn't match the volume's record of the file, or None if it does."""
    if len(data) != ref["size"]:
        return f"size {len(data)} != {ref['size']} on the volume"
    if "-" in ref["md5"] or len(ref["md5"]) != 32:
        if sha256_b64 is None:
            return f"the volume's ETag {ref['md5']!r} isn't an MD5, and there's no SHA-256 to check instead"
    elif hashlib.md5(data).hexdigest() != ref["md5"]:
        return f"MD5 {hashlib.md5(data).hexdigest()} != {ref['md5']} on the volume"
    if sha256_b64 is not None and base64.b64encode(hashlib.sha256(data).digest()).decode() != sha256_b64:
        return "SHA-256 doesn't match the volume's"
    return None


def write_sums(out, manifest):
    sums = manifest.get(".sha256", {})
    lines = [f"{h}  {name}\n" for name, h in sorted(sums.items()) if (out / name).exists()]
    (out / "SHA256SUMS").write_text("".join(lines))


# ── Saving ───────────────────────────────────────────────────────────────────

def free_path(out, base, data, ext=".exr"):
    """Where to write `data` for name `base` without overwriting anything: the
    plain name if it's free, an existing file with identical contents
    (returned with same=True; nothing to write), or the next free numbered
    name: base-001.exr, base-002.exr, ..."""
    n = 0
    while True:
        p = out / (f"{base}{ext}" if n == 0 else f"{base}-{n:03d}{ext}")
        if not p.exists():
            return p, False
        if p.stat().st_size == len(data) and p.read_bytes() == data:
            return p, True
        n += 1


def save(out, base, data, manifest, key, verified):
    path, same = free_path(out, base, data)
    if not same:
        path.write_bytes(data)
    print(f"  {'already here: ' if same else ''}{path.name}  ({len(data) / 1e6:.1f} MB, "
          + ("checksum OK" if verified else "NOT verified") + ")")
    manifest.setdefault(".outputs", {})[key] = path.name
    manifest.setdefault(".outputs_md5", {})[key] = hashlib.md5(data).hexdigest()
    if verified:
        manifest.setdefault(".sha256", {})[path.name] = sha256_hex(data)
    return path


def pulled_name(manifest, out, key, ref):
    """The local name `key` was pulled as, if that record is for the file that's
    on the volume now; else None. ComfyUI reuses names (depth_00001.exr) once
    the originals are deleted, so a key alone isn't enough: the recorded MD5
    must match, or for older records the local file must."""
    name = manifest.get(".outputs", {}).get(key)
    if not name:
        return None
    md5 = manifest.get(".outputs_md5", {}).get(key)
    if md5 is None:
        local = out / name
        if not local.exists():
            return name  # can't tell; keep the old behaviour
        md5 = hashlib.md5(local.read_bytes()).hexdigest()
    return name if ref is None or "-" in ref["md5"] or md5 == ref["md5"] else None


def download_checked(get, ref, label):
    """Run get() until the result matches `ref` (if given). (data, sha_b64) or None."""
    for attempt in range(1, TRIES + 1):
        try:
            data, sha_b64 = get()
        except Exception as e:
            print(f"  {label}: download failed ({e}), try {attempt}/{TRIES}")
            continue
        if ref is None:
            return data, sha_b64
        why = mismatch(data, ref, sha_b64)
        if why is None:
            return data, sha_b64
        print(f"  {label}: {why}, try {attempt}/{TRIES}")
    return None


# ── Modes ────────────────────────────────────────────────────────────────────

def pull_from_pod(args, out, manifest, manifest_path):
    if args.url:
        url = args.url.rstrip("/")
        pod_id = urllib.parse.urlparse(url).hostname.split("-")[0]
        print(f"Target: {url}")
    else:
        pod_id, url = resolve_target(args.name, args.cache_minutes, args.refresh)
    args.pod_id = pod_id  # for pull_favourites: the volume attached to this pod
    pulled = manifest.setdefault(pod_id, {})
    outputs = manifest.setdefault(".outputs", {})

    found = jobs(url, args.include_bench)
    new = [j for j in found if j["prompt_id"] not in pulled]
    print(f"{len(found)} result(s) on the pod, {len(new)} not pulled yet.")

    if args.list:
        for j in found:
            mark = f"pulled -> {pulled[j['prompt_id']]}" if j["prompt_id"] in pulled else "new"
            print(f"  {target_name(j):60s} {mark}")
        return pull_crops(args, url, pod_id, out, manifest, manifest_path)

    # Record which volume file each earlier pull came from (manifests written
    # before .outputs existed get filled in here).
    for j in found:
        if pulled.get(j["prompt_id"]):
            outputs.setdefault(output_key(j["exr"]), pulled[j["prompt_id"]])
    if not new:
        if out.exists():
            manifest_path.write_text(json.dumps(manifest, indent=1))
        return pull_crops(args, url, pod_id, out, manifest, manifest_path)

    refs = vol = None
    if not args.no_verify:
        try:
            vol = open_volume(args, pod_id)
            refs = vol.list_exrs()
            print(f"Checking against volume {vol.label}.")
        except (RuntimeError, SystemExit) as e:
            sys.exit(f"ERROR: can't reach the volume to verify downloads ({e}).\n"
                     "       Pass --no-verify to download unchecked, then --verify-existing later.")

    out.mkdir(parents=True, exist_ok=True)
    failed = gone = 0
    for j in new:
        base, key = target_name(j), output_key(j["exr"])
        ref = None
        if refs is not None:
            ref = refs.get(key)
            if ref is None:
                # Deleted from the volume (--clear-volume) before it was pulled:
                # record it so later runs don't keep retrying.
                print(f"  {base}: {key} is no longer on the volume; recorded as gone")
                pulled[j["prompt_id"]] = None
                manifest_path.write_text(json.dumps(manifest, indent=1))
                gone += 1
                continue
        got = download_checked(lambda: (fetch(view_url(url, j["exr"]), timeout=120), None), ref, base)
        if got is None and vol is not None:
            # The pod didn't deliver a good copy (restarting, proxy errors): the
            # same file is on the volume, so fetch it through S3 instead.
            print(f"  {base}: trying the volume through S3 instead")
            got = download_checked(lambda: vol.get(key), ref, base)
        if got is None:
            print(f"  FAILED {base}: no download matched the volume's checksums (see above); not saved")
            failed += 1
            continue
        path = save(out, base, got[0], manifest, key, verified=ref is not None)
        if j["preview"]:
            try:
                png = fetch(view_url(url, j["preview"]), timeout=60)
                ppath, same = free_path(out, f"{path.stem}_preview", png, ".png")
                if not same:
                    ppath.write_bytes(png)
            except Exception:
                print(f"    (preview no longer on the pod: {j['preview']['filename']})")
        pulled[j["prompt_id"]] = path.name
        manifest_path.write_text(json.dumps(manifest, indent=1))
    write_sums(out, manifest)
    print(f"Done: {len(new) - failed - gone} pulled into {out}"
          + (f", {gone} gone from the volume" if gone else "")
          + (f", {failed} failed (rerun to retry)" if failed else "") + ".")
    return max(1 if failed else 0, pull_crops(args, url, pod_id, out, manifest, manifest_path, vol, refs))


CROP_SUFFIX = "-cropped.exr"


def crop_name(manifest, key):
    """Local name for a crop: the local name of the map it was cut from plus -cropped
    (photo_depth_2496px-cropped), else the pod's own name."""
    stem = Path(key).name[: -len(CROP_SUFFIX)]
    src = manifest.get(".outputs", {}).get(str(Path(key).parent / f"{stem}.exr"))
    return (Path(src).stem if src else stem) + "-cropped"


def pull_crops(args, url, pod_id, out, manifest, manifest_path, vol=None, refs=None):
    """The vathopy GUI's crops (<map>-cropped.exr in output/vathography/); see the docstring."""
    if not args.no_verify and refs is None:
        try:
            vol = open_volume(args, pod_id)
            refs = vol.list_exrs()
        except (RuntimeError, SystemExit) as e:
            print(f"Crops: can't reach the volume to verify them ({e}); skipped (--no-verify pulls them unchecked).")
            return 0
    if refs is not None:
        keys = sorted(k for k in refs if k.startswith("output/vathography/") and k.endswith(CROP_SUFFIX))
    else:
        try:
            maps = json.loads(fetch(url.replace("-8188.", "-8770.") + "/api/maps", timeout=30))["maps"]
        except Exception as e:
            print(f"Crops: can't list them from the pod's GUI ({e}); skipped.")
            return 0
        keys = [f"output/vathography/{m}" for m in maps if m.endswith(CROP_SUFFIX)]
    md5s = manifest.setdefault(".outputs_md5", {})

    def pulled(k, data=None):
        """Already pulled: the recorded MD5 matches the volume's (or the downloaded data's)."""
        md5 = refs[k]["md5"] if refs is not None else (hashlib.md5(data).hexdigest() if data is not None else None)
        return md5 is not None and md5s.get(k) == md5 and (out / manifest[".outputs"].get(k, "")).is_file()

    new = [k for k in keys if refs is None or not pulled(k)]
    print(f"Crops: {len(keys)} on the pod" + (f", {len(new)} not pulled yet." if refs is not None else "."))
    if args.list:
        for k in keys:
            mark = "new" if refs is None or k in new else f"pulled -> {manifest['.outputs'].get(k)}"
            print(f"  {crop_name(manifest, k):60s} {mark}")
        return 0
    failed = done = same = 0
    for k in new:
        base = crop_name(manifest, k)
        ref = refs.get(k) if refs is not None else None
        f = {"filename": Path(k).name, "subfolder": "vathography", "type": "output"}
        got = download_checked(lambda: (fetch(view_url(url, f), timeout=120), None), ref, base)
        if got is None and vol is not None:
            print(f"  {base}: trying the volume through S3 instead")
            got = download_checked(lambda: vol.get(k), ref, base)
        if got is None:
            print(f"  FAILED {base}: not saved")
            failed += 1
            continue
        if refs is None and pulled(k, got[0]):
            same += 1                                  # unchanged since the last pull
            continue
        out.mkdir(parents=True, exist_ok=True)
        save(out, base, got[0], manifest, k, verified=ref is not None)
        manifest_path.write_text(json.dumps(manifest, indent=1))
        done += 1
    if done:
        write_sums(out, manifest)
    if new:
        print(f"Crops: {done} pulled" + (f", {same} unchanged" if same else "")
              + (f", {failed} failed (rerun to retry)" if failed else "") + ".")
    return 1 if failed else 0


def pull_from_volume(args, out, manifest, manifest_path):
    try:
        vol = open_volume(args)
        files = vol.list_exrs()
    except RuntimeError as e:
        sys.exit(f"ERROR: can't read the volume: {e}")
    print(f"Volume: {vol.label}")
    manifest.setdefault(".outputs", {})
    keys = sorted((k for k in files if (args.include_bench or not Path(k).name.startswith(BENCH_PREFIX))
                   and (not args.match or any(fnmatch.fnmatch(Path(k).name, m) for m in args.match))),
                  key=lambda k: files[k]["modified"])
    done = {k: pulled_name(manifest, out, k, files[k]) for k in keys}
    new = [k for k in keys if not done[k]]
    print(f"{len(keys)} result(s) on the volume, {len(new)} not pulled yet.")

    if args.list:
        for k in keys:
            mark = f"pulled -> {done[k]}" if done[k] else "new"
            print(f"  {k:50s} {files[k]['size'] / 1e6:6.1f} MB  {files[k]['modified'][:16]}  {mark}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for k in new:
        rel = k[len("output/"):]
        if rel.startswith("vathography/"):
            rel = rel[len("vathography/"):]
        base = rel[: -len(".exr")].replace("/", "_")
        ref = None if args.no_verify else files[k]
        got = download_checked(lambda: vol.get(k), ref, base)
        if got is None:
            print(f"  FAILED {k}: no download matched the volume's checksums (see above); not saved")
            failed += 1
            continue
        save(out, base, got[0], manifest, k, verified=ref is not None)
        manifest_path.write_text(json.dumps(manifest, indent=1))
    write_sums(out, manifest)
    print(f"Done: {len(new) - failed} pulled into {out}" + (f", {failed} failed (rerun to retry)" if failed else "") + ".")
    return 1 if failed else 0


def verify_existing(args, out, manifest, manifest_path):
    """Check every local file recorded in .outputs against the volume's MD5."""
    outputs = manifest.get(".outputs", {})
    if not outputs:
        sys.exit(f"Nothing recorded in {manifest_path} yet. Run a normal pull first "
                 "(it also records files that were already pulled).")
    try:
        vol = open_volume(args)
        files = vol.list_exrs()
    except RuntimeError as e:
        sys.exit(f"ERROR: can't read the volume: {e}")
    print(f"Checking {len(outputs)} file(s) in {out} against volume {vol.label}:")
    sums = manifest.setdefault(".sha256", {})
    bad = 0
    for key, name in sorted(outputs.items(), key=lambda kv: kv[1]):
        path = out / name
        if not path.exists():
            print(f"  MISSING   {name}")
            bad += 1
            continue
        if key not in files:
            print(f"  NOT ON VOLUME  {name}  ({key}; deleted from the volume?)")
            bad += 1
            continue
        data = path.read_bytes()
        why = mismatch(data, files[key])
        if why:
            print(f"  MISMATCH  {name}: {why}")
            sums.pop(name, None)
            bad += 1
        else:
            print(f"  OK        {name}")
            sums[name] = sha256_hex(data)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    write_sums(out, manifest)
    print(f"{len(outputs) - bad} OK, {bad} problem(s)." + ("" if not bad else
          " Delete a mismatched file and its entry in .pulled.json, then pull again."))
    return 1 if bad else 0


def clear_volume(args, out, manifest, manifest_path):
    """Delete .exr/.png results under output/ from the volume (see the docstring)."""
    try:
        vol = open_volume(args)
        files = vol.list_outputs((".exr", ".png"))
    except RuntimeError as e:
        sys.exit(f"ERROR: can't read the volume: {e}")
    outputs = manifest.get(".outputs", {})
    safe, kept = [], []
    for key in sorted(files, key=lambda k: files[k]["modified"]):
        name = outputs.get(key)
        local = out / name if name else None
        if local is not None and local.exists() and mismatch(local.read_bytes(), files[key]) is None:
            safe.append((key, f"pulled as {name}, checksum OK"))
        elif local is not None and local.exists():
            kept.append((key, f"local copy {name} does NOT match the volume"))
        elif name:
            kept.append((key, f"pulled as {name}, but that file is gone from {out}"))
        else:
            kept.append((key, "not pulled"))
    delete = safe + (kept if args.include_unpulled else [])
    keep = [] if args.include_unpulled else kept

    print(f"Volume {vol.label}: {len(files)} .exr/.png file(s) under output/.")
    for key, why in delete:
        print(f"  delete  {key:50s} {files[key]['size'] / 1e6:6.1f} MB  ({why})")
    for key, why in keep:
        print(f"  keep    {key:50s} {files[key]['size'] / 1e6:6.1f} MB  ({why})")
    if keep:
        print(f"{len(keep)} file(s) kept. --include-unpulled deletes them too.")
    if not delete:
        print("Nothing to delete.")
        return 0
    total = sum(files[k]["size"] for k, _ in delete) / 1e6
    if not args.yes:
        if not sys.stdin.isatty():
            sys.exit("ERROR: not a terminal, so I can't ask; pass --yes to delete.")
        if input(f"Delete these {len(delete)} file(s) ({total:.1f} MB) from the volume? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return 0
    failed = 0
    for key, _ in delete:
        try:
            vol.delete(key)
        except RuntimeError as e:
            print(f"  FAILED {key}: {e}")
            failed += 1
    left = vol.list_outputs((".exr", ".png"))
    still = [k for k, _ in delete if k in left]
    # Forget deleted files, so a later file ComfyUI saves under the same name
    # isn't taken for one that was already pulled.
    for key, _ in delete:
        if key not in left:
            manifest.get(".outputs", {}).pop(key, None)
            manifest.get(".outputs_md5", {}).pop(key, None)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"Deleted {len(delete) - len(still)} of {len(delete)} file(s); "
          f"{len(left)} .exr/.png file(s) left under output/." + (f" Still present: {', '.join(still)}" if still else ""))
    return 1 if (failed or still) else 0


KEEP_INPUTS = {"example.png"}  # ComfyUI's stock Load Image default


def clear_inputs(args, out, manifest, manifest_path):
    """Delete uploaded photos under input/ from the volume (see the docstring)."""
    try:
        vol = open_volume(args)
        files = vol.list_outputs((), prefix="input/")
    except RuntimeError as e:
        sys.exit(f"ERROR: can't read the volume: {e}")
    # ComfyUI's placeholders are empty "_..." files; photos can start with "_" too.
    keep = {k for k in files if Path(k).name in KEEP_INPUTS
            or (Path(k).name.startswith("_") and files[k]["size"] == 0)}
    delete = sorted(k for k in files if k not in keep)

    # A queued or running job still needs its photo.
    try:
        _, url = resolve_target(args.name, args.cache_minutes, args.refresh) if not args.url else (None, args.url.rstrip("/"))
        q = json.loads(fetch(url + "/queue", timeout=30))
        busy = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        if busy:
            sys.exit(f"ERROR: the pod has {busy} job(s) queued or running, which may need their photos. "
                     "Try again when the queue is empty.")
    except SystemExit as e:
        if e.code and "queued or running" in str(e.code):
            raise
        print("(No running pod to check for queued jobs; continuing.)")

    print(f"Volume {vol.label}: {len(files)} file(s) under input/.")
    groups = {}
    for k in delete:
        rel = k[len("input/"):]
        folder = rel.split("/")[0] + "/" if "/" in rel else "(uploaded in the browser)"
        n, size = groups.get(folder, (0, 0))
        groups[folder] = (n + 1, size + files[k]["size"])
    for folder, (n, size) in sorted(groups.items()):
        print(f"  delete  {folder:40s} {n:3d} file(s)  {size / 1e6:7.1f} MB")
    for k in sorted(keep):
        print(f"  keep    {k}")
    if not delete:
        print("Nothing to delete.")
        return 0
    total = sum(files[k]["size"] for k in delete) / 1e6
    if not args.yes:
        if not sys.stdin.isatty():
            sys.exit("ERROR: not a terminal, so I can't ask; pass --yes to delete.")
        if input(f"Delete these {len(delete)} file(s) ({total:.1f} MB) from input/? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing deleted.")
            return 0
    failed = 0
    for k in delete:
        try:
            vol.delete(k)
        except RuntimeError as e:
            print(f"  FAILED {k}: {e}")
            failed += 1
    left = vol.list_outputs((), prefix="input/")
    still = [k for k in delete if k in left]
    print(f"Deleted {len(delete) - len(still)} of {len(delete)} file(s) ({total:.1f} MB); "
          f"{len(left)} file(s) left under input/." + (f" Still present: {', '.join(still)}" if still else ""))
    return 1 if (failed or still) else 0


def pull_favourites(args, pod_id=None):
    """Download the vathopy GUI's saves (favorites/ on the volume) into --favourites-out."""
    out = args.favourites_out.expanduser()
    manifest_path = out / ".pulled.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    try:
        vol = open_volume(args, pod_id)
        files = vol.list_outputs((".png", ".json"), prefix="favorites/")
    except (RuntimeError, SystemExit) as e:
        print(f"GUI saves: can't read the volume ({e}); skipped.")
        return 1
    files = {k: v for k, v in files.items() if v["size"] > 0}
    keys = sorted(files, key=lambda k: files[k]["modified"])
    done = {k: pulled_name(manifest, out, k, files[k]) for k in keys}
    new = [k for k in keys if not done[k]]
    print(f"GUI saves (favorites/ on {vol.label}): {len(keys)}, {len(new)} not pulled yet.")
    if args.list:
        for k in keys:
            print(f"  {k:60s} {('pulled -> ' + done[k]) if done[k] else 'new'}")
        return 0
    if not new:
        return 0
    failed = 0
    for key in new:
        rel = key[len("favorites/"):]
        sub, name = os.path.split(rel)
        stem, ext = os.path.splitext(name)
        ref = None if args.no_verify else files[key]
        got = download_checked(lambda: vol.get(key), ref, rel)
        if got is None:
            print(f"  FAILED {rel}: no download matched the volume's checksums (see above); not saved")
            failed += 1
            continue
        data = got[0]
        (out / sub).mkdir(parents=True, exist_ok=True)
        path, same = free_path(out / sub, stem, data, ext)
        if not same:
            path.write_bytes(data)
        relname = str(path.relative_to(out))
        print(f"  {'already here: ' if same else ''}{relname}  ({len(data) / 1e6:.1f} MB, "
              + ("checksum OK" if ref else "NOT verified") + ")")
        manifest.setdefault(".outputs", {})[key] = relname
        manifest.setdefault(".outputs_md5", {})[key] = hashlib.md5(data).hexdigest()
        if ref:
            manifest.setdefault(".sha256", {})[relname] = sha256_hex(data)
        manifest_path.write_text(json.dumps(manifest, indent=1))
    write_sums(out, manifest)
    print(f"GUI saves: {len(new) - failed} pulled into {out}" + (f", {failed} failed (rerun to retry)" if failed else "") + ".")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser(description="Download finished Marigold V2 results from the pod, checksum-verified.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"Destination folder (default {DEFAULT_OUT})")
    ap.add_argument("--name", default="vathography-comfyui", help="Pod name to look up (default vathography-comfyui)")
    ap.add_argument("--url", help="Pod proxy URL; skips the lookup and the cache")
    ap.add_argument("--refresh", action="store_true", help="Look the pod up again now, ignoring the cache")
    ap.add_argument("--cache-minutes", type=float, default=30, help="How long a looked-up target is trusted (default 30)")
    ap.add_argument("--include-bench", action="store_true", help="Also pull runs made by bench.py")
    ap.add_argument("--list", action="store_true", help="List results and whether they're pulled; download nothing")
    ap.add_argument("--from-volume", action="store_true",
                    help="Read results from the network volume's S3 bucket instead of the pod's history")
    ap.add_argument("--verify-existing", action="store_true",
                    help="Check already-downloaded files against the volume's checksums; download nothing")
    ap.add_argument("--no-verify", action="store_true", help="Download without checking against the volume")
    ap.add_argument("--clear-volume", action="store_true",
                    help="Delete .exr/.png results under output/ from the volume (only verified-pulled ones unless --include-unpulled)")
    ap.add_argument("--include-unpulled", action="store_true",
                    help="With --clear-volume: also delete files that have no matching local copy")
    ap.add_argument("--clear-inputs", action="store_true",
                    help="Delete uploaded photos under input/ from the volume (keeps example.png)")
    ap.add_argument("--yes", action="store_true", help="With --clear-volume/--clear-inputs: don't ask before deleting")
    ap.add_argument("--match", nargs="+", metavar="PATTERN",
                    help="With --from-volume: only files whose name matches one of these patterns, e.g. 'IMG_*'")
    ap.add_argument("--favourites-out", type=Path, default=DEFAULT_FAV_OUT,
                    help=f"Where the vathopy GUI's saves go (default {DEFAULT_FAV_OUT})")
    ap.add_argument("--no-favourites", action="store_true", help="Don't download the vathopy GUI's saves")
    ap.add_argument("--volume-name", default="vathography-storage", help="Volume to look up (default vathography-storage)")
    ap.add_argument("--volume-id", help="Volume ID; with --datacenter, skips the lookup")
    ap.add_argument("--datacenter", help="Volume's datacenter, e.g. US-MO-1")
    args = ap.parse_args()

    out = args.out.expanduser()
    manifest_path = out / ".pulled.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.clear_inputs:
        sys.exit(clear_inputs(args, out, manifest, manifest_path))
    if args.clear_volume:
        sys.exit(clear_volume(args, out, manifest, manifest_path))
    if args.verify_existing:
        sys.exit(verify_existing(args, out, manifest, manifest_path))
    if args.from_volume:
        rc = pull_from_volume(args, out, manifest, manifest_path)
    else:
        rc = pull_from_pod(args, out, manifest, manifest_path)
    if not args.no_favourites:
        rc = max(rc, pull_favourites(args, getattr(args, "pod_id", None)))
    sys.exit(rc)


if __name__ == "__main__":
    main()

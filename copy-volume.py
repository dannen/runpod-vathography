#!/usr/bin/env python3
"""
Seed a new network volume from an already-seeded one, RunPod to RunPod, without
the models passing through this machine.

Same datacenter (--from and --to in one DC): no pod. Each file is copied
server-side with S3 `copy-object` (~110 MB/s on US-MO-1). Cloudflare gives up on
a long call (HTTP 524 after ~2 min, e.g. the 20.5GB backbone) but the copy
carries on, so the script then waits for the object to reach its full size.
Files already there at the right size aren't copied again. Then each file is
checked against Hugging Face's SHA-256 (--verify, default `full`): a file with a
local copy in ./models/ (from populate-volume.sh) is checked without a download,
by the local copy's SHA-256 and the target's ETag (a server-side copy's ETag is
the content's MD5) against the local MD5; any other file is downloaded once and
hashed. `--verify size` only compares sizes with the source. A file that fails
is deleted from the target, so a rerun copies it again.

  ./copy-volume.py --from US-MO-1:<old-volume-id> --to US-MO-1:<new-volume-id>

Pushing (--push): when the target's datacenter has no pod supply, the pod runs in
the SOURCE datacenter instead, with the source volume at /workspace, and uploads
each file to the target's S3 gateway. Uploads go in 8 MB parts (the gateway sits
behind Cloudflare, which refuses a request over 100 MB with HTTP 413, and it
ignores x-amz-checksum-sha256), so each file is verified by its ETag: the pod
computes the multipart ETag (MD5 of the parts' MD5s) from its own copy, whose
SHA-256 it has checked against Hugging Face's, and compares it with the one the
target lists. That also catches a CompleteMultipartUpload that reports failure
after the object landed. Files over --max-size (default 5 GB, which leaves out the
20.5GB backbone) are skipped: upload those with populate-volume.sh from a
machine that has them (it skips files already on the volume). The job logs to
push.log on the source volume.

  ./copy-volume.py --from US-NE-1:<volume-id> --to US-IL-1:<volume-id> --push

Different datacenters (or --pod): a throwaway pod (a 2 vCPU CPU pod by default, ~$0.06/hr; or the cheapest GPU pod
with --gpu, for datacenters with no CPU supply) starts with the TARGET volume at
/workspace and runs a short job: `get-object` each file in models.txt from the
SOURCE volume's S3 gateway into /workspace/models/, and check it against the
SHA-256 Hugging Face publishes for it (X-Linked-Etag), 3 tries per file. Files
already there with the right checksum are skipped, so a rerun resumes.
The job logs to /workspace/seed.log and leaves seed.done or seed.failed.

This script follows seed.log through the target's S3 gateway and deletes the pod
when the job ends (it holds your S3 key in its env). Ctrl-C stops following and
leaves the pod running: delete it yourself.

  ./copy-volume.py --from US-MO-1:<volume-id> --to US-NE-1:<volume-id>
  ./copy-volume.py --from US-MO-1:<volume-id> --to US-IL-1:<volume-id> --gpu "NVIDIA RTX A5000"

Both datacenters need RunPod's S3 API (the source for the copy, the target for
following the log). Measured 2026-10-19 (US-MO-1 -> US-NE-1): 42-50 MB/s on most
files; the source gateway dipped to 3 MB/s for a while and one attempt at the
20.5GB backbone failed (the retry ran at 30 MB/s). ~24 min and ~$0.03 in all.

Credentials: RUNPOD_API_KEY or ~/.api/runpod/claude.key (as create-pod.sh), and
AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or ~/.api/runpod/s3-credentials (as
populate-volume.sh). Python 3 stdlib + the aws CLI.
"""
import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "runpod-vathography-copy/1.0"}
CPU_INSTANCES = ["cpu3c-2-4", "cpu3g-2-8", "cpu5c-2-4", "cpu5g-2-8", "cpu3c-4-8", "cpu3g-4-16"]

POD_JOB = r"""set -u
L=/workspace/seed.log
exec >>"$L" 2>&1
rm -f /workspace/seed.done /workspace/seed.failed
echo "[seed] start __RUN__ $(date -u +%FT%TZ)"
pip install -q --root-user-action=ignore awscli || { echo "[seed] pip failed"; touch /workspace/seed.failed; sleep infinity; }
S3="aws --endpoint-url https://s3api-__SRC_DC_LC__.runpod.io --region __SRC_DC_LC__ --cli-read-timeout 600"
fail=0
while read -r key sha; do
  dest="/workspace/$key"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then echo "[seed] have $key"; continue; fi
  ok=0
  for attempt in 1 2 3; do
    t0=$(date +%s)
    if $S3 s3api get-object --bucket __SRC_VOL__ --key "$key" "$dest.partial" >/dev/null; then
      t1=$(date +%s); size=$(stat -c %s "$dest.partial"); dt=$(( t1 - t0 > 0 ? t1 - t0 : 1 ))
      if [ "$(sha256sum "$dest.partial" | cut -d' ' -f1)" = "$sha" ]; then
        mv "$dest.partial" "$dest"
        echo "[seed] ok $key $size bytes in $dt s ($(( size / dt / 1000000 )) MB/s), sha256 OK"
        ok=1; break
      fi
      echo "[seed] sha256 MISMATCH $key (attempt $attempt)"
    else
      echo "[seed] get-object failed $key (attempt $attempt)"
    fi
    rm -f "$dest.partial"; sleep 5
  done
  [ $ok = 1 ] || fail=1
done <<'LIST'
__LIST__
LIST
if [ $fail = 0 ]; then echo "[seed] ALL OK $(date -u +%FT%TZ)"; touch /workspace/seed.done
else echo "[seed] FAILED $(date -u +%FT%TZ)"; touch /workspace/seed.failed; fi
sleep infinity
"""


PUSH_PART = 8 << 20          # bytes per multipart part, as populate-volume.sh

PUSH_JOB = r"""
import base64, hashlib, json, os, subprocess, time
RUN, DC, VOL, MAX = "__RUN__", "__DST_DC_LC__", "__DST_VOL__", __MAX__
PART = __PART__
FILES = json.loads(base64.b64decode("__FILES__"))
LOG = "/workspace/push.log"
cfg = "/tmp/aws.cfg"
open(cfg, "w").write(f"[default]\nretry_mode = adaptive\nmax_attempts = 15\ns3 =\n"
                     f"    multipart_threshold = {PART}\n    multipart_chunksize = {PART}\n    max_concurrent_requests = 4\n")
os.environ["AWS_CONFIG_FILE"] = cfg
AWS = ["aws", "--endpoint-url", f"https://s3api-{DC}.runpod.io", "--region", DC, "--cli-read-timeout", "600"]

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[seed] {msg}\n")

def hashes(path):
    sha, parts, whole = hashlib.sha256(), [], hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(PART):
            sha.update(chunk); whole.update(chunk); parts.append(hashlib.md5(chunk).digest())
    etag = whole.hexdigest() if len(parts) <= 1 else hashlib.md5(b"".join(parts)).hexdigest() + f"-{len(parts)}"
    return sha.hexdigest(), etag

def target_etag(key):
    for _ in range(4):
        r = subprocess.run(AWS + ["s3api", "list-objects-v2", "--bucket", VOL, "--prefix", key, "--output", "json"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            for o in json.loads(r.stdout or "{}").get("Contents", []):
                if o["Key"] == key:
                    return o["Size"], o.get("ETag", "").strip('"')
            return None, None
        time.sleep(10)
    return None, None

log(f"start {RUN} {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
fail = skipped = 0
for key, want in FILES:
    path = "/workspace/" + key
    if not os.path.isfile(path):
        log(f"FAILED {key}: not on the source volume"); fail += 1; continue
    size = os.path.getsize(path)
    if size > MAX:
        log(f"skip {key} ({size / 1e9:.1f} GB > --max-size): upload it another way"); skipped += 1; continue
    sha, etag = hashes(path)
    if sha != want:
        log(f"FAILED {key}: the source copy's SHA-256 isn't Hugging Face's"); fail += 1; continue
    if target_etag(key) == (size, etag):
        log(f"have {key}"); continue
    ok = False
    for attempt in (1, 2, 3):
        t0 = time.time()
        r = subprocess.run(AWS + ["s3", "cp", "--only-show-errors", path, f"s3://{VOL}/{key}"], capture_output=True, text=True)
        dt = max(time.time() - t0, 1)
        got = target_etag(key)
        if got == (size, etag):
            note = "" if r.returncode == 0 else " (the upload call failed, but the object landed)"
            log(f"ok {key} {size} bytes in {dt:.0f} s ({size / dt / 1e6:.0f} MB/s), ETag OK{note}")
            ok = True; break
        log(f"attempt {attempt} {key}: target has {got}, want ({size}, {etag}); {(r.stderr.strip().splitlines() or [''])[-1][:150]}")
        time.sleep(10)
    if not ok:
        fail += 1
end = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
log((f"ALL OK {end}" if not fail else f"FAILED {end}") + (f" ({skipped} skipped)" if skipped else ""))
"""


def runpod_key():
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    f = Path.home() / ".api/runpod/claude.key"
    if f.is_file():
        return f.read_text().strip()
    sys.exit("ERROR: no API key (set RUNPOD_API_KEY or create ~/.api/runpod/claude.key)")


def s3_creds():
    ak, sk = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
    if ak and sk:
        return ak, sk
    f = Path.home() / ".api/runpod/s3-credentials"
    if not f.is_file():
        sys.exit("ERROR: no S3 credentials (set AWS_* or create ~/.api/runpod/s3-credentials)")
    t = f.read_text()
    m1, m2 = re.search(r"aws_access_key_id\s*=\s*(\S+)", t), re.search(r"aws_secret_access_key\s*=\s*(\S+)", t)
    if not (m1 and m2):
        sys.exit(f"ERROR: can't read the keys in {f}")
    return m1.group(1), m2.group(1)


def manifest():
    """[(volume key, Hugging Face SHA-256)] for every file in models.txt, smallest first
    (so the first lines of the log show the transfer speed)."""
    out = []
    for line in (HERE / "models.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) != 2 or line.lstrip().startswith("#"):
            continue
        subdir, url = parts
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:       # follows the redirect; the etag is on the first hop
                sha, size = r.headers.get("X-Linked-Etag"), int(r.headers.get("Content-Length") or 0)
        except urllib.error.HTTPError as e:
            sha, size = e.headers.get("X-Linked-Etag"), 0
        if not sha:
            sha = _first_hop_etag(url)
        sha = (sha or "").strip('"')
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            sys.exit(f"ERROR: no SHA-256 from Hugging Face for {url}")
        out.append((size, f"models/{subdir}/{url.rsplit('/', 1)[1]}", sha))
    return [(k, s) for _, k, s in sorted(out)]


def _first_hop_etag(url):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None
    opener = urllib.request.build_opener(NoRedirect)
    try:
        opener.open(urllib.request.Request(url, method="HEAD", headers=UA), timeout=60)
    except urllib.error.HTTPError as e:
        return e.headers.get("X-Linked-Etag")
    return None


def api(url, body, key, method="POST"):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}", **UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}


def create_pod(args, job, key, ak, sk):
    """Pod id, or None. CPU pods go through GraphQL: REST /v1/pods rejects some datacenters (US-NE-1)."""
    name = f"vathography-copy-{args.to_dc.lower()}"
    dc, vol = (args.src_dc, args.src_vol) if args.push else (args.to_dc, args.to_vol)
    if args.gpu:
        r = api("https://rest.runpod.io/v1/pods", {
            "name": name, "imageName": args.image, "cloudType": "SECURE", "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "dataCenterIds": [dc], "dataCenterPriority": "custom", "networkVolumeId": vol,
            "volumeInGb": 0, "volumeMountPath": "/workspace", "containerDiskInGb": 10, "ports": [],
            "env": {"AWS_ACCESS_KEY_ID": ak, "AWS_SECRET_ACCESS_KEY": sk}, "dockerStartCmd": ["bash", "-c", job]}, key)
        if r.get("id"):
            print(f"GPU pod {r['id']} ({args.gpu}, ${r.get('costPerHr')}/hr)")
            return r["id"]
        print(f"  {args.gpu}: {r.get('error', r)}")
        return None
    b64 = base64.b64encode(job.encode()).decode()
    for inst in ([args.cpu] if args.cpu else CPU_INSTANCES):
        inp = {"instanceId": inst, "cloudType": "SECURE", "dataCenterId": dc, "networkVolumeId": vol,
               "volumeMountPath": "/workspace", "containerDiskInGb": 10, "deployCost": 0.5, "imageName": args.image,
               "name": name, "startJupyter": False, "startSsh": False, "ports": "",
               "dockerArgs": f"bash -c 'echo {b64} | base64 -d | bash'",
               "env": [{"key": "AWS_ACCESS_KEY_ID", "value": ak}, {"key": "AWS_SECRET_ACCESS_KEY", "value": sk}]}
        r = api("https://api.runpod.io/graphql", {"query": "mutation($input: deployCpuPodInput!) "
                "{ deployCpuPod(input: $input) { id costPerHr } }", "variables": {"input": inp}}, key)
        pod = (r.get("data") or {}).get("deployCpuPod")
        if pod and pod.get("id"):
            print(f"CPU pod {pod['id']} ({inst}, ${pod.get('costPerHr')}/hr)")
            return pod["id"]
        msg = (r.get("errors") or [{}])[0].get("message") or r.get("error") or r
        print(f"  {inst}: {msg}")
    return None


def follow(args, ak, sk, run):
    """Print this run's [seed] lines until its job ends; True if it succeeded. seed.log is
    appended to across runs, so only the lines after this run's start line count."""
    env = {**os.environ, "AWS_ACCESS_KEY_ID": ak, "AWS_SECRET_ACCESS_KEY": sk}
    # A pulling pod logs on the target volume; a pushing one on the source volume, where it runs.
    dc, vol, name = (args.src_dc, args.src_vol, "push.log") if args.push else (args.to_dc, args.to_vol, "seed.log")
    dc = dc.lower()
    shown = 0
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "seed.log"
        while True:
            r = subprocess.run(["aws", "--endpoint-url", f"https://s3api-{dc}.runpod.io", "--region", dc, "s3api",
                                "get-object", "--bucket", vol, "--key", name, str(log)],
                               env=env, capture_output=True, text=True)
            if r.returncode == 0:
                lines = [l for l in log.read_text(errors="replace").splitlines() if l.startswith("[seed]")]
                starts = [i for i, l in enumerate(lines) if l.startswith(f"[seed] start {run} ")]
                lines = lines[starts[-1]:] if starts else []
                for l in lines[shown:]:
                    print(time.strftime("%H:%M:%S"), l, flush=True)
                shown = len(lines)
                if lines and (" ALL OK " in lines[-1] or " FAILED " in lines[-1] or "pip failed" in lines[-1]):
                    return " ALL OK " in lines[-1]
            time.sleep(30)


# ── Same datacenter: server-side copy ────────────────────────────────────────

def aws(dc, args, env, timeout=120):
    dc = dc.lower()
    try:
        return subprocess.run(["aws", "--endpoint-url", f"https://s3api-{dc}.runpod.io", "--region", dc,
                               "--cli-read-timeout", str(timeout)] + args,
                              env=env, capture_output=True, text=True, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 1, "", f"timed out after {timeout}s")


def listing(dc, vol, env, prefix="models/"):
    """{key: (size, etag)}; the gateway can be slow to list, so it's retried."""
    out, token = {}, None
    while True:
        cmd = ["s3api", "list-objects-v2", "--bucket", vol, "--prefix", prefix, "--output", "json"]
        if token:
            cmd += ["--continuation-token", token]
        for attempt in range(4):
            r = aws(dc, cmd, env, timeout=300)
            if r.returncode == 0:
                break
            time.sleep(10)
        else:
            sys.exit(f"ERROR: can't list {vol}: {r.stderr.strip()[:200]}")
        page = json.loads(r.stdout or "{}")
        for o in page.get("Contents", []):
            out[o["Key"]] = (o["Size"], o.get("ETag", "").strip('"'))
        token = page.get("NextContinuationToken")
        if not token:
            return out


def server_copy(args, files, env):
    """copy-object each file that isn't on the target at the source's size. True if all landed."""
    src, dst = listing(args.src_dc, args.src_vol, env), listing(args.to_dc, args.to_vol, env)
    ok = True
    for key, _ in files:
        if key not in src:
            print(f"  MISSING on the source: {key}")
            ok = False
            continue
        size = src[key][0]
        if dst.get(key, (None,))[0] == size:
            print(f"  have {key}")
            continue
        t0 = time.time()
        r = aws(args.to_dc, ["s3api", "copy-object", "--copy-source", f"{args.src_vol}/{key}",
                             "--bucket", args.to_vol, "--key", key], env, timeout=900)
        if r.returncode != 0:
            # Typically Cloudflare's 524 on a big file: the gateway finishes the copy anyway.
            print(f"  {key}: copy call failed ({(r.stderr.strip().splitlines() or ['?'])[-1][:120]}); waiting for it to land")
            deadline = time.time() + 1800
            while time.time() < deadline and listing(args.to_dc, args.to_vol, env, prefix=key).get(key, (None,))[0] != size:
                time.sleep(30)
            if listing(args.to_dc, args.to_vol, env, prefix=key).get(key, (None,))[0] != size:
                print(f"  FAILED {key}: not at full size after 30 min")
                ok = False
                continue
        dt = max(time.time() - t0, 1)
        print(f"  copied {key} ({size / 1e9:.2f} GB in {dt:.0f} s, {size / dt / 1e6:.0f} MB/s)")
    return ok


def file_hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def verify(args, files, env):
    """Check every file on the target; delete any that fail. True if all pass."""
    src, dst = listing(args.src_dc, args.src_vol, env), listing(args.to_dc, args.to_vol, env)
    ok = True
    for key, want in files:
        size, etag = dst.get(key, (None, ""))
        why = None
        if size is None or size != src.get(key, (None,))[0]:
            why = f"size {size} (source {src.get(key, (None,))[0]})"
        elif args.verify == "full":
            local = HERE / key
            if local.is_file() and local.stat().st_size == size:
                sha, md5 = file_hashes(local)
                if sha != want:
                    why = None
                    print(f"  (local copy of {key} doesn't match Hugging Face; downloading instead)")
                    local = None
                elif md5 != etag:
                    why = f"ETag {etag} != local MD5 {md5}"
                else:
                    print(f"  OK {key} (ETag = MD5 of the local copy, whose SHA-256 = Hugging Face's)")
                    continue
            else:
                local = None
            if local is None and why is None:
                with tempfile.TemporaryDirectory() as tmp:
                    dest = Path(tmp) / "f"
                    r = aws(args.to_dc, ["s3api", "get-object", "--bucket", args.to_vol, "--key", key, str(dest)],
                            env, timeout=1800)
                    if r.returncode != 0:
                        why = f"download failed: {r.stderr.strip()[:120]}"
                    elif file_hashes(dest)[0] != want:
                        why = "SHA-256 != Hugging Face's"
                    else:
                        print(f"  OK {key} (downloaded; SHA-256 = Hugging Face's)")
                        continue
        else:
            print(f"  OK {key} (size)")
            continue
        print(f"  FAILED {key}: {why}; deleting it from the target")
        aws(args.to_dc, ["s3api", "delete-object", "--bucket", args.to_vol, "--key", key], env)
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description="Seed a network volume from another one, RunPod to RunPod.")
    ap.add_argument("--from", dest="src", required=True, metavar="DC:VOLUME", help="the seeded volume, e.g. US-MO-1:abc123")
    ap.add_argument("--to", dest="dst", required=True, metavar="DC:VOLUME", help="the volume to seed")
    ap.add_argument("--pod", action="store_true", help="use a pod even when both volumes are in one datacenter")
    ap.add_argument("--push", action="store_true",
                    help="run the pod in the source datacenter and upload to the target (when the target has no pod supply)")
    ap.add_argument("--max-size", type=float, default=5.0, metavar="GB",
                    help="--push: skip files bigger than this (default 5, which skips the 20.5GB backbone)")
    ap.add_argument("--verify", choices=["full", "size"], default="full",
                    help="same-datacenter copies: check SHA-256s (default; downloads files with no local copy) or only sizes")
    ap.add_argument("--cpu", metavar="INSTANCE", help=f"CPU instance id (<flavor>-<vcpu>-<memGB>); default: try {', '.join(CPU_INSTANCES)}")
    ap.add_argument("--gpu", metavar="TYPE", help='use a GPU pod instead, e.g. "NVIDIA RTX A5000" (for datacenters with no CPU supply)')
    ap.add_argument("--image", default="python:3.12-slim", help="pod image (needs bash, pip, sha256sum, base64)")
    ap.add_argument("--keep-pod", action="store_true", help="don't delete the pod when the job ends")
    ap.add_argument("--no-follow", action="store_true", help="start the pod and exit (delete it yourself when seed.done appears)")
    args = ap.parse_args()
    try:
        args.src_dc, args.src_vol = args.src.split(":")
        args.to_dc, args.to_vol = args.dst.split(":")
    except ValueError:
        ap.error("--from and --to take DC:VOLUME")
    if args.cpu and args.gpu:
        ap.error("--cpu and --gpu are mutually exclusive")

    if args.src_vol == args.to_vol:
        ap.error("--from and --to are the same volume")

    ak, sk = s3_creds()
    files = manifest()
    print(f"{len(files)} files from {args.src} to {args.dst} (SHA-256 from Hugging Face)")
    if args.src_dc.upper() == args.to_dc.upper() and not (args.pod or args.push or args.cpu or args.gpu):
        env = {**os.environ, "AWS_ACCESS_KEY_ID": ak, "AWS_SECRET_ACCESS_KEY": sk}
        print("Same datacenter: server-side copy-object, no pod.")
        copied = server_copy(args, files, env)
        print(f"Verifying ({args.verify}):")
        ok = verify(args, files, env) and copied
        print("ALL OK" if ok else "FAILED: rerun to copy the failed files again")
        sys.exit(0 if ok else 1)

    key = runpod_key()
    run = f"run-{int(time.time())}"
    if args.push:
        py = (PUSH_JOB.replace("__RUN__", run).replace("__DST_DC_LC__", args.to_dc.lower()).replace("__DST_VOL__", args.to_vol)
              .replace("__MAX__", str(int(args.max_size * 1e9))).replace("__PART__", str(PUSH_PART))
              .replace("__FILES__", base64.b64encode(json.dumps(files).encode()).decode()))
        job = ("set -u\npip install -q --root-user-action=ignore awscli >/workspace/push-pip.log 2>&1 "
               "|| echo '[seed] pip failed' >> /workspace/push.log\n"
               f"echo {base64.b64encode(py.encode()).decode()} | base64 -d > /tmp/push.py && python3 /tmp/push.py\nsleep infinity\n")
    else:
        job = (POD_JOB.replace("__RUN__", run).replace("__SRC_DC_LC__", args.src_dc.lower()).replace("__SRC_VOL__", args.src_vol)
               .replace("__LIST__", "\n".join(f"{k} {s}" for k, s in files)))
    pod = create_pod(args, job, key, ak, sk)
    if not pod:
        sys.exit("No pod available. Try again later, another --cpu instance, or --gpu.")
    if args.no_follow:
        print(f"Not following. Delete pod {pod} when seed.done appears: it holds your S3 key.")
        return
    try:
        ok = follow(args, ak, sk, run)
    except KeyboardInterrupt:
        sys.exit(f"\nStopped following; pod {pod} is still running (and holds your S3 key): delete it when done.")
    if args.keep_pod:
        print(f"Job {'succeeded' if ok else 'FAILED'}; pod {pod} kept (--keep-pod).")
    else:
        r = api(f"https://rest.runpod.io/v1/pods/{pod}", None, key, method="DELETE")
        print(f"Job {'succeeded' if ok else 'FAILED'}; pod {pod} " + ("deleted." if "error" not in r else f"NOT deleted: {r['error']}"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

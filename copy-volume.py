#!/usr/bin/env python3
"""
Seed a new network volume from an already-seeded one, RunPod to RunPod, without
the models passing through this machine.

A throwaway pod (a 2 vCPU CPU pod by default, ~$0.06/hr; or the cheapest GPU pod
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
    if args.gpu:
        r = api("https://rest.runpod.io/v1/pods", {
            "name": name, "imageName": args.image, "cloudType": "SECURE", "gpuTypeIds": [args.gpu], "gpuCount": 1,
            "dataCenterIds": [args.to_dc], "dataCenterPriority": "custom", "networkVolumeId": args.to_vol,
            "volumeInGb": 0, "volumeMountPath": "/workspace", "containerDiskInGb": 10, "ports": [],
            "env": {"AWS_ACCESS_KEY_ID": ak, "AWS_SECRET_ACCESS_KEY": sk}, "dockerStartCmd": ["bash", "-c", job]}, key)
        if r.get("id"):
            print(f"GPU pod {r['id']} ({args.gpu}, ${r.get('costPerHr')}/hr)")
            return r["id"]
        print(f"  {args.gpu}: {r.get('error', r)}")
        return None
    b64 = base64.b64encode(job.encode()).decode()
    for inst in ([args.cpu] if args.cpu else CPU_INSTANCES):
        inp = {"instanceId": inst, "cloudType": "SECURE", "dataCenterId": args.to_dc, "networkVolumeId": args.to_vol,
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
    dc = args.to_dc.lower()
    shown = 0
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "seed.log"
        while True:
            r = subprocess.run(["aws", "--endpoint-url", f"https://s3api-{dc}.runpod.io", "--region", dc, "s3api",
                                "get-object", "--bucket", args.to_vol, "--key", "seed.log", str(log)],
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


def main():
    ap = argparse.ArgumentParser(description="Seed a network volume from another one, RunPod to RunPod.")
    ap.add_argument("--from", dest="src", required=True, metavar="DC:VOLUME", help="the seeded volume, e.g. US-MO-1:abc123")
    ap.add_argument("--to", dest="dst", required=True, metavar="DC:VOLUME", help="the volume to seed")
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

    key, (ak, sk) = runpod_key(), s3_creds()
    files = manifest()
    print(f"{len(files)} files from {args.src} to {args.dst} (SHA-256 from Hugging Face)")
    run = f"run-{int(time.time())}"
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

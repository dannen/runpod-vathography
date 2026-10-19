#!/usr/bin/env bash
# Seed the Vathography network volume from this machine via RunPod's S3
# gateway, bypassing the pod. Pods have been seen downloading from Hugging
# Face at well under 1MB/s, while a home connection is typically far faster.
#
# Usage: ./populate-volume.sh <datacenter-id> <volume-id> [--keep-local] [--dry-run]
#   e.g.: ./populate-volume.sh US-KS-2 <volume-id> --dry-run
#
# Reads models.txt (same manifest start.sh uses), then:
#   1. Lists the volume once and skips files already on it.
#   2. Gets each pending file's size up front (HTTP HEAD), and orders the
#      queue smallest to largest, for downloads and uploads alike.
#   3. Pipelines the queue: while file N uploads, file N+1 downloads in the
#      background, so the two directions overlap. Never more than one file
#      ahead, so peak local disk is the largest adjacent pair in flight
#      (~22GB here: a 1.7GB LoRA + the 20.5GB backbone at the end), not
#      the whole set.
#   4. Checks each download's size against the server's Content-Length and
#      each upload's size on the volume, then deletes the local copy
#      (--keep-local keeps it).
# --dry-run only reads (volume listing + HTTP HEAD requests) and prints the
# plan: what's on the volume, what's pending, sizes, and local disk headroom.
#
# Downloads resume from ./models/<subdir>/<file>.partial, so Ctrl-C and
# rerun is safe. Uploads restart from scratch on retry (aws s3 cp).
#
# Credentials: ~/.api/runpod/s3-credentials, or AWS_ACCESS_KEY_ID /
# AWS_SECRET_ACCESS_KEY already in the environment.
#
# The conservative multipart settings below are deliberate: more aggressive
# chunking/concurrency has tripped repeated Cloudflare 524 timeouts on the
# gateway. They're applied through a
# throwaway AWS_CONFIG_FILE rather than `aws configure set`, so your global
# ~/.aws/config isn't touched.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USAGE="usage: $0 <datacenter-id> <volume-id> [--keep-local] [--dry-run]"
DC="${1:?$USAGE}"
VOL="${2:?$USAGE}"
shift 2
KEEP_LOCAL=false
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --keep-local) KEEP_LOCAL=true ;;
        --dry-run)    DRY_RUN=true ;;
        *) echo "$USAGE"; exit 1 ;;
    esac
done

ENDPOINT="https://s3api-${DC,,}.runpod.io/"
LOCAL_ROOT="$SCRIPT_DIR/models"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/models.txt}"

if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    CREDS_FILE="$HOME/.api/runpod/s3-credentials"
    [[ -f "$CREDS_FILE" ]] || { echo "ERROR: no S3 credentials (set AWS_* or create $CREDS_FILE)"; exit 1; }
    AWS_ACCESS_KEY_ID=$(awk -F'=' '/aws_access_key_id/{gsub(/ /,"",$2); print $2}' "$CREDS_FILE")
    AWS_SECRET_ACCESS_KEY=$(awk -F'=' '/aws_secret_access_key/{gsub(/ /,"",$2); print $2}' "$CREDS_FILE")
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
fi

AWS_CONFIG_FILE=$(mktemp)
export AWS_CONFIG_FILE
cat > "$AWS_CONFIG_FILE" <<EOF
[default]
retry_mode = adaptive
max_attempts = 15
s3 =
    multipart_chunksize = 8MB
    max_concurrent_requests = 4
EOF

# On exit or Ctrl-C: stop a background download (its .partial stays for
# resume) and remove the temp aws config.
cleanup() { kill $(jobs -p) 2>/dev/null; rm -f "$AWS_CONFIG_FILE"; }
trap cleanup EXIT
trap 'echo; echo "Interrupted. Rerun to resume."; exit 130' INT TERM

s3() { aws "$@" --region "$DC" --endpoint-url "$ENDPOINT"; }
gb() { awk -v b="$1" 'BEGIN { printf "%.1fGB", b / 1e9 }'; }
# Final Content-Length after redirects (HF answers with a CDN redirect).
remote_size() { curl -sIL --fail "$1" | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n + 0 }'; }

# ---- 1. What's already on the volume (one listing, key -> size) ----
echo "Volume: s3://$VOL/models via $ENDPOINT"
if ! listing=$(s3 s3 ls "s3://$VOL/models/" --recursive 2>&1); then
    # An empty volume has no models/ prefix yet, which `ls` reports as exit 1
    # with no output; anything else is a real error.
    if [[ -n "$listing" ]]; then echo "ERROR: can't list volume: $listing"; exit 1; fi
fi
declare -A on_volume=()
while read -r _ _ size key; do
    [[ -n "${key:-}" ]] && on_volume["$key"]=$size
done <<< "$listing"

# ---- 2. Build the pending list from the manifest ----
subdirs=(); urls=(); files=(); sizes=()
total_pending=0
while read -r -u 3 subdir url; do
    [[ -z "$subdir" || "$subdir" == \#* ]] && continue
    file="$(basename "$url")"
    key="models/$subdir/$file"
    # Multipart objects only appear once CompleteMultipartUpload succeeds,
    # so "listed" means "fully uploaded".
    if [[ -n "${on_volume[$key]:-}" ]]; then
        echo "SKIP (on volume, $(gb "${on_volume[$key]}")): $subdir/$file"
        continue
    fi
    size=$(remote_size "$url")
    if (( size == 0 )); then
        echo "ERROR: can't get size (HEAD failed): $url"; exit 1
    fi
    subdirs+=("$subdir"); urls+=("$url"); files+=("$file"); sizes+=("$size")
    total_pending=$((total_pending + size))
done 3< "$MANIFEST"

n=${#files[@]}

# Smallest first, for both downloads and uploads (they share this order):
# the small files land on the volume within minutes, and a failure on one
# of the big ones late in the run doesn't hold everything else up. Stable
# sort, so equal sizes keep manifest order.
if (( n > 1 )); then
    sorted=$(for i in "${!files[@]}"; do printf '%s\t%s\n' "${sizes[$i]}" "$i"; done | sort -s -n -k1,1 | cut -f2)
    s_subdirs=(); s_urls=(); s_files=(); s_sizes=()
    for i in $sorted; do
        s_subdirs+=("${subdirs[$i]}"); s_urls+=("${urls[$i]}"); s_files+=("${files[$i]}"); s_sizes+=("${sizes[$i]}")
    done
    subdirs=("${s_subdirs[@]}"); urls=("${s_urls[@]}"); files=("${s_files[@]}"); sizes=("${s_sizes[@]}")
fi

echo
echo "Pending: $n file(s), $(gb $total_pending), smallest first"
for i in "${!files[@]}"; do
    local_path="$LOCAL_ROOT/${subdirs[$i]}/${files[$i]}"
    state="download"
    [[ -f "$local_path" ]] && state="local copy present, upload only"
    [[ -f "$local_path.partial" ]] && state="resume from $(gb "$(stat -c %s "$local_path.partial")")"
    printf "  %-8s %-24s %s  (%s)\n" "$(gb "${sizes[$i]}")" "${subdirs[$i]}/" "${files[$i]}" "$state"
done
(( n == 0 )) && { echo; echo "All models in models.txt are on the volume."; exit 0; }

# Peak local disk = the largest pair of adjacent files (current upload +
# next download).
peak=0
for ((i = 0; i < n; i++)); do
    pair=${sizes[$i]}
    (( i + 1 < n )) && pair=$((pair + sizes[i + 1]))
    (( pair > peak )) && peak=$pair
done
$KEEP_LOCAL && peak=$total_pending
mkdir -p "$LOCAL_ROOT"
free=$(( $(df -B1 --output=avail "$LOCAL_ROOT" | tail -1) ))
echo
echo "Local disk: need ~$(gb $peak) at peak, $(gb $free) free at $LOCAL_ROOT"
if (( free < peak )); then
    echo "ERROR: not enough local disk."; exit 1
fi

if $DRY_RUN; then
    echo; echo "Dry run: nothing downloaded or uploaded."; exit 0
fi

# ---- 3. Pipeline: download N+1 in the background while N uploads ----
download() {  # $1 = index. Quiet (no progress bar: it'd interleave with the upload's).
    local i=$1 url=${urls[$1]} want=${sizes[$1]}
    local local_path="$LOCAL_ROOT/${subdirs[$1]}/${files[$1]}"
    if [[ -f "$local_path" ]]; then
        local have; have=$(stat -c %s "$local_path")
        (( have == want )) && { echo "[download] already local: ${files[$i]}"; return 0; }
        echo "[download] local copy is $(gb "$have"), expected $(gb "$want"); re-downloading: ${files[$i]}"
        mv "$local_path" "$local_path.partial"
    fi
    mkdir -p "$(dirname "$local_path")"
    echo "[download] start: ${files[$i]} ($(gb "$want"))"
    local t0=$SECONDS
    if ! curl -L --fail -sS -C - -o "$local_path.partial" "$url"; then
        echo "[download] ERROR: ${files[$i]} (rerun to resume)"; return 1
    fi
    local have; have=$(stat -c %s "$local_path.partial")
    if (( have != want )); then
        echo "[download] ERROR: ${files[$i]} is $(gb "$have"), expected $(gb "$want") (rerun to resume)"; return 1
    fi
    mv "$local_path.partial" "$local_path"
    local secs=$(( SECONDS - t0 > 0 ? SECONDS - t0 : 1 ))
    echo "[download] done: ${files[$i]} in ${secs}s ($(awk -v b="$want" -v s="$secs" 'BEGIN { printf "%.1f", b / s / 1e6 }') MB/s)"
}

upload() {  # $1 = index
    local i=$1 want=${sizes[$1]}
    local local_path="$LOCAL_ROOT/${subdirs[$1]}/${files[$1]}"
    local s3path="s3://$VOL/models/${subdirs[$1]}/${files[$1]}"
    echo "[upload] start: ${subdirs[$i]}/${files[$i]} ($(gb "$want"))"
    local attempt=0 backoff=10 t0=$SECONDS
    until s3 s3 cp --only-show-errors "$local_path" "$s3path"; do
        attempt=$((attempt + 1))
        if (( attempt >= 5 )); then
            echo "[upload] ERROR: gave up after $attempt attempts: ${files[$i]} (local copy kept)"; return 1
        fi
        echo "[upload] attempt $attempt failed, retrying in ${backoff}s..."
        sleep "$backoff"
        backoff=$(( backoff * 2 > 60 ? 60 : backoff * 2 ))
    done
    local got
    got=$(s3 s3 ls "$s3path" | awk '{ print $3 }')
    if [[ "$got" != "$want" ]]; then
        echo "[upload] ERROR: ${files[$i]} is ${got:-missing} bytes on the volume, expected $want (local copy kept)"; return 1
    fi
    local secs=$(( SECONDS - t0 > 0 ? SECONDS - t0 : 1 ))
    echo "[upload] done: ${subdirs[$i]}/${files[$i]} in ${secs}s ($(awk -v b="$want" -v s="$secs" 'BEGIN { printf "%.1f", b / s / 1e6 }') MB/s)"
    $KEEP_LOCAL || rm -f "$local_path"
}

echo
failed=()
start=$SECONDS
prev=-1; prev_ok=false
for ((i = 0; i < n; i++)); do
    download "$i" &
    dl_pid=$!
    if (( prev >= 0 )) && $prev_ok; then
        upload "$prev" || failed+=("${files[$prev]}")
    fi
    if wait "$dl_pid"; then prev_ok=true; else prev_ok=false; failed+=("${files[$i]}"); fi
    prev=$i
done
if $prev_ok; then
    upload "$prev" || failed+=("${files[$prev]}")
fi

echo
echo "Finished in $(( (SECONDS - start) / 60 ))m. Volume contents:"
s3 s3 ls "s3://$VOL/models/" --recursive --human-readable

if (( ${#failed[@]} )); then
    echo; echo "FAILED (rerun to retry): ${failed[*]}"; exit 1
fi
echo; echo "All models in models.txt are on the volume."

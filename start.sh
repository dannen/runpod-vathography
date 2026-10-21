#!/usr/bin/env bash
# Entrypoint for the Vathography (Marigold V2) persistent pod container.
#
# The custom node and the vathopy GUI are baked into the image (see Dockerfile),
# so nothing here installs software:
# 1. Wait for the RunPod network volume to actually be mounted at
#    VOLUME_ROOT (an empty dir could just be the container's overlay fs,
#    which would silently put everything on ephemeral disk).
# 2. Serialize concurrent boots with a lock file on the volume.
# 3. Move ComfyUI's models/output/input/user dirs onto the volume and
#    replace them with symlinks, so every boot (even from a rebuilt image)
#    sees the same persistent content.
# 4. Download anything in models.txt that's missing from the volume. Every
#    boot checks, but existing files are skipped, so once the volume is
#    seeded this is a fast no-op. SKIP_MODEL_DOWNLOAD=true skips it
#    MODELS_S3=<DC>:<volume id> (plus AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY,
#    a RunPod S3 key) copies each missing file from that seeded network
#    volume's S3 gateway instead, falling back to Hugging Face if it fails.
#    entirely (e.g. while populate-volume.sh --s3 is still uploading).
# 5. Seed the bundled workflows into the Workflows sidebar.
# 6. Start the vathopy GUI (port 8770) in the background. With PUID set
#    (local runs with host folders mounted), hand new files to that user.
# 7. exec ComfyUI as PID 1 so RunPod's stop/restart signals land correctly,
#    behind nginx's password (the vathopy GUI's) when one is set.
set -euo pipefail

COMFY_ROOT=/opt/ComfyUI
VOLUME_ROOT=/workspace
LOCK_FILE="$VOLUME_ROOT/.boot.lock"

echo "[boot] waiting for $VOLUME_ROOT to mount..."
mounted=0
for i in $(seq 1 30); do
  if mountpoint -q "$VOLUME_ROOT"; then
    mounted=1
    break
  fi
  echo "[boot]   still waiting ($i/30)..."
  sleep 2
done

if [[ "$mounted" -ne 1 ]]; then
  echo "[FATAL] $VOLUME_ROOT never became a mountpoint." >&2
  echo "        Check that the pod was created with a networkVolumeId and" >&2
  echo "        that the volume's datacenter matches the pod's datacenter." >&2
  exit 1
fi

exec 9>"$LOCK_FILE"
if ! flock -w 300 9; then
  echo "[FATAL] could not acquire boot lock after 300s (another boot stuck?)" >&2
  exit 1
fi

echo "[boot] linking persistent directories into $VOLUME_ROOT"
for d in models output input user; do
  mkdir -p "$VOLUME_ROOT/$d"
  if [[ -L "$COMFY_ROOT/$d" ]]; then
    continue
  fi
  if [[ -d "$COMFY_ROOT/$d" ]]; then
    # Preserve ComfyUI's stub files under these dirs before switching.
    rsync -a --ignore-existing "$COMFY_ROOT/$d/" "$VOLUME_ROOT/$d/"
    rm -rf "$COMFY_ROOT/$d"
  fi
  ln -s "$VOLUME_ROOT/$d" "$COMFY_ROOT/$d"
done

if [[ "${SKIP_MODEL_DOWNLOAD:-false}" == "true" ]]; then
  echo "[boot] SKIP_MODEL_DOWNLOAD=true — not checking models.txt"
else
  echo "[boot] checking models against models.txt"
  # One file from a seeded network volume's S3 gateway (MODELS_S3=<DC>:<volume id>).
  # The gateway wants the datacenter in lower case. get-object, not "aws s3 cp":
  # the gateway rejects HeadObject.
  from_s3() {
    local dc="${MODELS_S3%%:*}" vol="${MODELS_S3#*:}"
    dc="${dc,,}"
    echo " • $(basename "$2" .partial): copying from $MODELS_S3"
    /opt/awscli/bin/aws --endpoint-url "https://s3api-$dc.runpod.io" --region "$dc" --cli-read-timeout 600 \
      s3api get-object --bucket "$vol" --key "$1" "$2" > /dev/null
  }
  while read -r -u 3 subdir url; do
    [[ -z "$subdir" || "$subdir" == \#* ]] && continue
    target="$VOLUME_ROOT/models/$subdir/$(basename "$url")"
    if [[ -f "$target" ]]; then
      echo " • $(basename "$target") exists - skip"
      continue
    fi
    mkdir -p "$(dirname "$target")"
    # .partial + mv so a killed transfer can't pass for a complete file.
    if [[ -n "${MODELS_S3:-}" && -n "${AWS_ACCESS_KEY_ID:-}" ]] && from_s3 "models/$subdir/$(basename "$url")" "${target}.partial"; then
      echo " • $(basename "$target") copied from $MODELS_S3"
    else
      echo " • downloading $(basename "$target")"
      curl -L --fail --progress-bar --show-error -o "${target}.partial" "$url"
    fi
    mv "${target}.partial" "$target"
  done 3< "$COMFY_ROOT/models.txt"
fi

echo "[boot] seeding bundled workflows"
mkdir -p "$VOLUME_ROOT/user/default/workflows"
rsync -a --ignore-existing /opt/vathography-workflows/ "$VOLUME_ROOT/user/default/workflows/"

flock -u 9

# Running locally with host folders mounted into /workspace (README "Local GPU"): ComfyUI and
# the GUI run as root, so hand what they write to the host user, PUID[:PGID], every few seconds.
# Unset on pods.
if [[ -n "${PUID:-}" ]]; then
  owner="${PUID}:${PGID:-$PUID}"
  echo "[boot] new files under $VOLUME_ROOT go to uid:gid $owner"
  (
    while true; do
      find "$VOLUME_ROOT" -uid 0 -print0 2>/dev/null | xargs -0 -r chown -h "$owner"
      sleep 5
    done
  ) &
fi

# vathopy GUI on 8770 (VATHOPY_GUI=off to skip): depth maps from ComfyUI's
# output/vathography/, saves to favorites/, your palettes from palettes/ on the volume. Set VATHOPY_GUI_AUTH
# (user:password) on the pod: RunPod's proxy URL is public. Restarted if it exits.
if [[ "${VATHOPY_GUI:-on}" != "off" && -x /opt/vathopy/bin/vathopy ]]; then
  mkdir -p "$VOLUME_ROOT/output/vathography" "$VOLUME_ROOT/favorites" "$VOLUME_ROOT/palettes"
  [[ -n "${VATHOPY_GUI_AUTH:-}" ]] || echo "[boot] WARNING: VATHOPY_GUI_AUTH not set; the vathopy GUI has no password"
  echo "[boot] starting vathopy gui on 0.0.0.0:8770 (vathopy $(cat /opt/vathopy/VATHOPY_COMMIT 2>/dev/null))"
  (
    while true; do
      /opt/vathopy/bin/vathopy --palettes "$VOLUME_ROOT/palettes" gui --host 0.0.0.0 --port 8770 \
        --depth-dir "$VOLUME_ROOT/output/vathography" --favourites "$VOLUME_ROOT/favorites" \
        >> /tmp/vathopy-gui.log 2>&1
      echo "[vathopy] gui exited ($?), restarting in 5s" >> /tmp/vathopy-gui.log
      sleep 5
    done
  ) &
fi

# ComfyUI has no login of its own, and RunPod's proxy URL is public. nginx on 8188 asks for the
# same user:password as the vathopy GUI (HTTP basic auth) and passes everything, websockets
# included, to ComfyUI on 127.0.0.1:8189, which nothing else can reach. COMFYUI_AUTH
# (user:password) defaults to VATHOPY_GUI_AUTH; COMFYUI_AUTH=off, or no password at all, leaves
# ComfyUI open on 8188 as before (with a warning), so the pod still boots.
COMFYUI_AUTH="${COMFYUI_AUTH:-${VATHOPY_GUI_AUTH:-}}"
cd "$COMFY_ROOT"
if [[ "$COMFYUI_AUTH" == *:* && -x /usr/sbin/nginx ]]; then
  (
    umask 077
    hash=$(printf '%s' "${COMFYUI_AUTH#*:}" | openssl passwd -apr1 -stdin)   # the password via stdin, not argv
    printf '%s:%s\n' "${COMFYUI_AUTH%%:*}" "$hash" > /etc/nginx/comfyui.htpasswd
  )
  chgrp www-data /etc/nginx/comfyui.htpasswd && chmod 640 /etc/nginx/comfyui.htpasswd   # nginx's workers read it
  cat > /etc/nginx/conf.d/comfyui.conf <<'NGINX'
map $http_upgrade $connection_upgrade { default upgrade; '' close; }
server {
    listen 8188 default_server;
    client_max_body_size 200m;
    location / {
        auth_basic "ComfyUI";
        auth_basic_user_file /etc/nginx/comfyui.htpasswd;
        proxy_pass http://127.0.0.1:8189;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header Authorization "";        # ComfyUI doesn't need the password
        proxy_buffering off;
        proxy_request_buffering off;
        proxy_read_timeout 1d;
        proxy_send_timeout 1d;
    }
}
NGINX
  if ! { nginx -t -q && nginx; }; then          # fail closed: never fall back to an open ComfyUI
    echo "[boot] ERROR: nginx didn't start; ComfyUI stays on 127.0.0.1 only, unreachable from outside"
  fi
  echo "[boot] starting ComfyUI on 127.0.0.1:8189 behind nginx on 0.0.0.0:8188 (password required)"
  exec ./venv/bin/python main.py --listen 127.0.0.1 --port 8189
fi
[[ "$COMFYUI_AUTH" == "off" ]] || echo "[boot] WARNING: no COMFYUI_AUTH/VATHOPY_GUI_AUTH (user:password); ComfyUI has no password"
echo "[boot] starting ComfyUI on 0.0.0.0:8188"
exec ./venv/bin/python main.py --listen 0.0.0.0 --port 8188

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
#    entirely (e.g. while populate-volume.sh --s3 is still uploading).
# 5. Seed the bundled workflows into the Workflows sidebar.
# 6. Start the vathopy GUI (port 8770) in the background.
# 7. exec ComfyUI as PID 1 so RunPod's stop/restart signals land correctly.
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
  while read -r -u 3 subdir url; do
    [[ -z "$subdir" || "$subdir" == \#* ]] && continue
    target="$VOLUME_ROOT/models/$subdir/$(basename "$url")"
    if [[ -f "$target" ]]; then
      echo " • $(basename "$target") exists - skip"
      continue
    fi
    echo " • downloading $(basename "$target")"
    mkdir -p "$(dirname "$target")"
    # .partial + mv so a killed transfer can't pass for a complete file.
    curl -L --fail --progress-bar --show-error -o "${target}.partial" "$url"
    mv "${target}.partial" "$target"
  done 3< "$COMFY_ROOT/models.txt"
fi

echo "[boot] seeding bundled workflows"
mkdir -p "$VOLUME_ROOT/user/default/workflows"
rsync -a --ignore-existing /opt/vathography-workflows/ "$VOLUME_ROOT/user/default/workflows/"

flock -u 9

# vathopy GUI on 8770 (VATHOPY_GUI=off to skip): depth maps from ComfyUI's
# output/vathography/, saves to favorites/ on the volume. Set VATHOPY_GUI_AUTH
# (user:password) on the pod: RunPod's proxy URL is public. Restarted if it exits.
if [[ "${VATHOPY_GUI:-on}" != "off" && -x /opt/vathopy/bin/vathopy ]]; then
  mkdir -p "$VOLUME_ROOT/output/vathography" "$VOLUME_ROOT/favorites"
  [[ -n "${VATHOPY_GUI_AUTH:-}" ]] || echo "[boot] WARNING: VATHOPY_GUI_AUTH not set; the vathopy GUI has no password"
  echo "[boot] starting vathopy gui on 0.0.0.0:8770 (vathopy $(cat /opt/vathopy/VATHOPY_COMMIT 2>/dev/null))"
  (
    while true; do
      /opt/vathopy/bin/vathopy gui --host 0.0.0.0 --port 8770 \
        --depth-dir "$VOLUME_ROOT/output/vathography" --favourites "$VOLUME_ROOT/favorites" \
        >> /tmp/vathopy-gui.log 2>&1
      echo "[vathopy] gui exited ($?), restarting in 5s" >> /tmp/vathopy-gui.log
      sleep 5
    done
  ) &
fi

cd "$COMFY_ROOT"
echo "[boot] starting ComfyUI on 0.0.0.0:8188"
exec ./venv/bin/python main.py --listen 0.0.0.0 --port 8188

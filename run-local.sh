#!/usr/bin/env bash
# Run the pod image on this machine's NVIDIA GPU (needs the NVIDIA Container Toolkit;
# 16GB of VRAM is enough with the cu128 image, see build.md). ComfyUI and the vathopy GUI
# listen on 127.0.0.1 only, so they run without a password.
#
#   ./run-local.sh                          # start (or restart) the container
#   ./run-local.sh --port 9188 --gui-port 9771   # on other local ports
#   ./run-local.sh stop                     # stop and remove it
#   ./run-local.sh logs                     # follow its log
#
# Folders (override with env vars):
#   MODELS  ./models (this repo; models.txt files missing there are downloaded on first start)
#   INPUT   ~/Pictures/vathography/input   photos uploaded to ComfyUI land here
#   OUTPUT  ~/Pictures/vathography/depth   ComfyUI writes its EXRs here; the GUI reads them
#   STATE   ~/.local/share/vathography-local   ComfyUI's user settings, the GUI's favorites/palettes
# Files the container writes there are handed to you (PUID/PGID), not left owned by root.
set -euo pipefail
cd "$(dirname "$0")"

NAME=${NAME:-vathography-local}
IMAGE=${IMAGE:-dannen/vathography-comfyui:latest-cu128}
MODELS=${MODELS:-$PWD/models}
INPUT=${INPUT:-$HOME/Pictures/vathography/input}
OUTPUT=${OUTPUT:-$HOME/Pictures/vathography/depth}
STATE=${STATE:-$HOME/.local/share/vathography-local}
COMFY_PORT=${COMFY_PORT:-8188}
GUI_PORT=${GUI_PORT:-8771}      # 8770 is left for a vathopy GUI run outside the container

usage() { echo "usage: $0 [start] [--port N] [--gui-port N] | stop | logs" >&2; exit 2; }
cmd=start
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|logs) cmd=$1 ;;
    --port) COMFY_PORT=${2:?--port needs a number}; shift ;;
    --gui-port) GUI_PORT=${2:?--gui-port needs a number}; shift ;;
    *) usage ;;
  esac
  shift
done
case "$cmd" in
  stop) docker rm -f "$NAME" >/dev/null && echo "stopped $NAME"; exit ;;
  logs) exec docker logs -f "$NAME" ;;
esac

# Create the folders as you, or Docker creates them owned by root.
mkdir -p "$MODELS" "$INPUT" "$OUTPUT" "$STATE"/{models,input,output/vathography}   # the last: mount points
# Check the ports before touching a running container (its own ports don't count: it's restarted).
ours=$(docker port "$NAME" 2>/dev/null | sed 's/.*://' || true)
for port in "$COMFY_PORT" "$GUI_PORT"; do
  if ss -ltn "sport = :$port" | grep -q LISTEN && ! grep -qx "$port" <<< "$ours"; then
    echo "ERROR: port $port is already in use; pick another with --port / --gui-port" >&2; exit 1
  fi
done
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  -v "$STATE:/workspace" -v "$MODELS:/workspace/models" \
  -v "$INPUT:/workspace/input" -v "$OUTPUT:/workspace/output/vathography" \
  -e COMFYUI_AUTH=off -e PUID="$(id -u)" -e PGID="$(id -g)" \
  -p "127.0.0.1:$COMFY_PORT:8188" -p "127.0.0.1:$GUI_PORT:8770" \
  "$IMAGE" >/dev/null

echo "Starting $NAME ($IMAGE) ..."
for _ in $(seq 1 600); do
  curl -sf -o /dev/null "http://127.0.0.1:$COMFY_PORT/system_stats" && break
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != true ]]; then
    docker logs "$NAME" 2>&1 | tail -20; echo "ERROR: $NAME stopped" >&2; exit 1
  fi
  sleep 3
done
echo "ComfyUI:     http://127.0.0.1:$COMFY_PORT"
echo "vathopy GUI: http://127.0.0.1:$GUI_PORT"
echo "Photos in:   $INPUT"
echo "Depth out:   $OUTPUT"

#!/usr/bin/env bash
# Run the pod image on this machine's NVIDIA GPU (needs the NVIDIA Container Toolkit;
# 16GB of VRAM is enough with the cu128 image, see build.md). ComfyUI and the vathopy GUI
# listen on 127.0.0.1 only, so they run without a password.
#
#   ./run-local.sh                          # start (or restart) the container
#   ./run-local.sh --port 9188 --gui-port 9771   # on other local ports
#   ./run-local.sh --config my.conf         # settings from a file (see below)
#   ./run-local.sh stop                     # stop and remove it
#   ./run-local.sh logs                     # follow its log
#
# Settings, each from (highest first) the command line, the environment, the config file,
# or the default:
#   MODELS      ./models (this repo; models.txt files missing there are downloaded on first start)
#   INPUT       ~/Pictures/vathography/input   photos uploaded to ComfyUI land here
#   OUTPUT      ~/Pictures/vathography/depth   ComfyUI writes its EXRs here; the GUI reads them
#   FAVORITES   ~/Pictures/vathography/favorites   the GUI's saves (PNGs + looks/*.json)
#   STATE       ~/.local/share/vathography-local   ComfyUI's user settings, the GUI's palettes
#   COMFY_PORT  8188 (--port)       GUI_PORT  8771 (--gui-port)
#   IMAGE       dannen/vathography-comfyui:latest-cu128       NAME  vathography-local
# Files the container writes there are handed to you (PUID/PGID), not left owned by root.
#
# Config file: --config FILE, else ~/.config/vathography/run-local.conf if it exists. KEY=value
# lines with the keys above; # comments; ~ and $HOME are expanded, and relative paths are
# relative to the file. It's read, not run. Example: run-local.conf.example.
set -euo pipefail

KEYS="NAME IMAGE MODELS INPUT OUTPUT FAVORITES STATE COMFY_PORT GUI_PORT"
usage() { echo "usage: $0 [start] [--config FILE] [--port N] [--gui-port N] | stop [--config FILE] | logs" >&2; exit 2; }
cmd=start config="" cli_comfy="" cli_gui=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|logs) cmd=$1 ;;
    --config) config=$(realpath "${2:?--config needs a file}"); shift ;;
    --port) cli_comfy=${2:?--port needs a number}; shift ;;
    --gui-port) cli_gui=${2:?--gui-port needs a number}; shift ;;
    *) usage ;;
  esac
  shift
done
[[ -n "$config" || ! -f "$HOME/.config/vathography/run-local.conf" ]] || config="$HOME/.config/vathography/run-local.conf"
[[ -z "$config" || -f "$config" ]] || { echo "ERROR: no config file $config" >&2; exit 1; }

# The config file fills in what the environment didn't set.
if [[ -n "$config" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"; line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" ]] && continue
    [[ "$line" == *=* ]] || { echo "ERROR: $config: not KEY=value: $line" >&2; exit 1; }
    key="${line%%=*}" val="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"; val="${val#"${val%%[![:space:]]*}"}"
    [[ " $KEYS " == *" $key "* ]] || { echo "ERROR: $config: unknown key $key (known: $KEYS)" >&2; exit 1; }
    val="${val#[\"\']}"; val="${val%[\"\']}"                         # optional quotes
    val="${val/#\~/$HOME}"; val="${val//\$HOME/$HOME}"; val="${val//\$\{HOME\}/$HOME}"
    if [[ "$key" =~ ^(MODELS|INPUT|OUTPUT|FAVORITES|STATE)$ && "$val" != /* ]]; then val="$(dirname "$config")/$val"; fi
    [[ -n "${!key:-}" ]] || printf -v "$key" '%s' "$val"
  done < "$config"
fi

cd "$(dirname "$0")"
NAME=${NAME:-vathography-local}
IMAGE=${IMAGE:-dannen/vathography-comfyui:latest-cu128}
MODELS=${MODELS:-$PWD/models}
INPUT=${INPUT:-$HOME/Pictures/vathography/input}
OUTPUT=${OUTPUT:-$HOME/Pictures/vathography/depth}
FAVORITES=${FAVORITES:-$HOME/Pictures/vathography/favorites}
STATE=${STATE:-$HOME/.local/share/vathography-local}
COMFY_PORT=${cli_comfy:-${COMFY_PORT:-8188}}
GUI_PORT=${cli_gui:-${GUI_PORT:-8771}}      # 8770 is left for a vathopy GUI run outside the container

case "$cmd" in
  stop) docker rm -f "$NAME" >/dev/null && echo "stopped $NAME"; exit ;;
  logs) exec docker logs -f "$NAME" ;;
esac

# Create the folders as you, or Docker creates them owned by root.
mkdir -p "$MODELS" "$INPUT" "$OUTPUT" "$FAVORITES" "$STATE"/{models,input,output/vathography,favorites}   # the last: mount points
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
  -v "$FAVORITES:/workspace/favorites" \
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
echo "GUI saves:   $FAVORITES"

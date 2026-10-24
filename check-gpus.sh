#!/usr/bin/env bash
# GPU stock and prices: runs tools/gpus.py from the comfyui-runpod repo
# (https://github.com/dannen/comfyui-runpod). It looks for a checkout next to
# this repo or one folder up; COMFYUI_RUNPOD=/path/to/checkout overrides.
#   ./check-gpus.sh                      >= 24GB, US/CA datacenters with network volumes
#   ./check-gpus.sh --vram 48 --all-regions
#   ./check-gpus.sh --cheapest [--top N] [--vram N] [--us]   cheapest in stock anywhere
#   ./check-gpus.sh --help
#
# API key resolution order:
#   1. RUNPOD_API_KEY env var (if already set)
#   2. ~/.api/runpod/claude.key
#   3. ~/.runpod/config.toml (runpodctl config)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=""
if [ -n "${COMFYUI_RUNPOD:-}" ]; then dirs=("$COMFYUI_RUNPOD"); else dirs=("$SCRIPT_DIR/../comfyui-runpod" "$SCRIPT_DIR/../../comfyui-runpod"); fi
for dir in "${dirs[@]}"; do
    [ -f "$dir/tools/gpus.py" ] && { GPUS="$dir/tools/gpus.py"; break; }
done
if [ -z "$GPUS" ]; then
    echo "ERROR: comfyui-runpod not found: git clone https://github.com/dannen/comfyui-runpod next to this repo, or set COMFYUI_RUNPOD"
    exit 1
fi

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    KEY_FILE="$HOME/.api/runpod/claude.key"
    TOML_FILE="$HOME/.runpod/config.toml"
    if [ -f "$KEY_FILE" ]; then
        RUNPOD_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")
    elif [ -f "$TOML_FILE" ]; then
        RUNPOD_API_KEY=$(sed -n 's/^[[:space:]]*api[Kk]ey[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$TOML_FILE" | head -1)
    fi
fi
[ -n "${RUNPOD_API_KEY:-}" ] || { echo "ERROR: No API key found. Set RUNPOD_API_KEY, or create ~/.api/runpod/claude.key"; exit 1; }

export RUNPOD_API_KEY
exec python3 "$GPUS" "$@"

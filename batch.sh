#!/usr/bin/env bash
# Run a folder of photos through Marigold V2 on the pod, then pull the results (see batch.py).
#
# Usage:
#   ./batch.sh DIR                                   # depth for every photo in DIR, 2496 px long edge
#   ./batch.sh DIR --resize 4200 --via-s3
#   ./batch.sh DIR --predictions depth normals albedo
#   ./batch.sh DIR --dry-run
#
# API key resolution order (same as the other runpod-* projects):
#   1. RUNPOD_API_KEY env var (if already set)
#   2. ~/.api/runpod/claude.key
#   3. ~/.runpod/config.toml (runpodctl config)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${RUNPOD_API_KEY:-}" ]; then
    KEY_FILE="$HOME/.api/runpod/claude.key"
    TOML_FILE="$HOME/.runpod/config.toml"

    if [ -f "$KEY_FILE" ]; then
        RUNPOD_API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")
    elif [ -f "$TOML_FILE" ]; then
        RUNPOD_API_KEY=$(python3 -c "
import re
with open('$TOML_FILE') as f:
    m = re.search(r'apikey\s*=\s*\"(.+?)\"', f.read())
    print(m.group(1) if m else '')
")
    else
        echo "ERROR: No API key found. Set RUNPOD_API_KEY, or create ~/.api/runpod/claude.key"
        exit 1
    fi
fi

if [ -z "$RUNPOD_API_KEY" ]; then
    echo "ERROR: Could not read API key."
    exit 1
fi

export RUNPOD_API_KEY
exec python3 "$SCRIPT_DIR/batch.py" "$@"

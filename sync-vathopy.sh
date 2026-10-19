#!/usr/bin/env bash
# Copy vathopy (from the vathography repo) into the build context as vathopy-src/,
# for the Dockerfile's vathopy GUI layer. Run before every docker build:
#   ./sync-vathopy.sh [path to the vathography repo, default ../../vathography]
# Records the source commit in vathopy-src/VATHOPY_COMMIT ("-dirty" if it had
# uncommitted changes). vathopy-src/ is gitignored: it's a build input, not source.
set -euo pipefail
SRC="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../vathography}"
DST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vathopy-src"
[[ -f "$SRC/vathopy/pyproject.toml" ]] || { echo "ERROR: no vathopy in $SRC"; exit 1; }
rsync -a --delete --exclude tests/ --exclude '__pycache__/' --exclude '*.egg-info/' \
      --exclude examples/ --exclude tools/ --exclude build/ --exclude .venv/ "$SRC/vathopy/" "$DST/"
commit=$(git -C "$SRC" rev-parse --short HEAD)
git -C "$SRC" diff --quiet -- vathopy || commit="$commit-dirty"
echo "$commit" > "$DST/VATHOPY_COMMIT"
echo "vathopy-src/ <- $SRC/vathopy ($commit)"

#!/usr/bin/env bash
# Assemble the Kaggle Dataset the scored notebook depends on.
#
# The scored run has no internet, so everything must be pre-attached:
#   - RDKit 2026.3.3 cp312 wheel (Kaggle's image has NO rdkit; probe confirmed)
#   - the prebuilt candidate pool (Kaggle gives only 4 cores, so building it
#     in-notebook would waste a large slice of the 9h budget)
#   - our casmi package source
set -euo pipefail

WORK=/mnt/casmi
STAGE="$WORK/kaggle_dataset"
export PATH="$HOME/.local/bin:$PATH"

rm -rf "$STAGE"
mkdir -p "$STAGE/src"

echo "=== rdkit wheel (cp312 manylinux, pinned to scoring version) ==="
cd "$WORK/repo"
.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
uv pip install -q --system-site-packages pip 2>/dev/null || true
# uv has no 'download'; use pip from the venv to fetch the wheel only.
.venv/bin/python -m pip download "rdkit==2026.3.3" \
    --no-deps --only-binary=:all: \
    --python-version 312 --implementation cp \
    --platform manylinux_2_28_x86_64 \
    -d "$STAGE" 2>&1 | tail -2

echo "=== pool ==="
cp "$WORK/data/processed/pool.npz" "$STAGE/pool.npz"

echo "=== code ==="
cp -r "$WORK/repo/src/casmi" "$STAGE/src/casmi"
find "$STAGE/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== contents ==="
du -sh "$STAGE"
ls -lh "$STAGE"

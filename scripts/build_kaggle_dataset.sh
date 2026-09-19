#!/usr/bin/env bash
# Assemble the Kaggle Dataset the scored notebook depends on.
#
# The scored run has no internet, so everything must be pre-attached:
#   - RDKit 2026.3.3 cp312 wheel (Kaggle's image has NO rdkit; probe confirmed)
#   - the prebuilt candidate pool (Kaggle gives only 4 cores, so building it
#     in-notebook would waste a large slice of the 9h budget)
#   - the trained FPNet checkpoint (Channel 4) — Kaggle's image already has
#     torch (2.10.0+cpu per the probe), so only the weights need attaching
#   - the fitted GBM reranker (Channel 4/5 feature fusion) — Kaggle's image
#     already has scikit-learn, so only the pickled ensemble needs attaching
#   - our casmi package source
set -euo pipefail

WORK=/mnt/casmi
STAGE="$WORK/kaggle_dataset"
CHECKPOINTS="$HOME/casmi_checkpoints"
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
cp "$CHECKPOINTS/pool.npz" "$STAGE/pool.npz"

echo "=== fpnet checkpoint (Channel 4) ==="
if [ -f "$CHECKPOINTS/fpnet_final.pt" ]; then
    cp "$CHECKPOINTS/fpnet_final.pt" "$STAGE/fpnet_final.pt"
else
    echo "WARNING: no FPNet checkpoint found at $CHECKPOINTS/fpnet_final.pt — Channel 4 will be skipped"
fi

echo "=== reranker (GBM ensemble) ==="
if [ -f "$WORK/checkpoints/ranker.pkl" ]; then
    cp "$WORK/checkpoints/ranker.pkl" "$STAGE/ranker.pkl"
else
    echo "WARNING: no ranker.pkl found at $WORK/checkpoints/ranker.pkl — reranker will be skipped"
fi

echo "=== code ==="
cp -r "$WORK/repo/src/casmi" "$STAGE/src/casmi"
find "$STAGE/src" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "=== contents ==="
du -sh "$STAGE"
ls -lh "$STAGE"

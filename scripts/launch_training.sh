#!/usr/bin/env bash
# Launch the full FPNet training run detached, so it survives SSH disconnect.
#
# Usage: launch_training.sh [MAX_STEPS] [BATCH_SIZE]
set -euo pipefail

MAX_STEPS="${1:-30000}"
BATCH="${2:-256}"

WORK=/mnt/casmi
REPO="$WORK/repo"
OUT="$WORK/checkpoints/fpnet"
LOG="$WORK/train.log"

mkdir -p "$OUT"
cd "$REPO"

if pgrep -f "train_fpnet.py" > /dev/null; then
    echo "ERROR: a training run is already active. Kill it first:"
    pgrep -af "train_fpnet.py"
    exit 1
fi

echo "launching: $MAX_STEPS steps, batch $BATCH -> $LOG"
setsid nohup .venv/bin/python scripts/train_fpnet.py \
    --train "$WORK/data/raw/train.parquet" \
    --split "$WORK/data/processed/split.npz" \
    --out "$OUT" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH" \
    --val-rows 5000 \
    --fp-workers 64 \
    > "$LOG" 2>&1 < /dev/null &

sleep 5
echo "pid $(pgrep -f train_fpnet.py | head -1)"
echo "started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

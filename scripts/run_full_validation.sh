#!/usr/bin/env bash
# Run the full step 6-8 validation sequence detached, so it survives SSH
# disconnect: baseline deltas (channel 1+2, +fragmentation, +fpnet) against
# the honest held-out split, then build the reranker's training data, fit it,
# and measure the full 4-channel+reranker MRR@25 on the same split.
#
# Usage: run_full_validation.sh
set -euo pipefail

WORK=/mnt/casmi
REPO="$WORK/repo"
DATA="$WORK/data/raw/train.parquet"
SPLIT="$HOME/casmi_checkpoints/split.npz"
POOL="$HOME/casmi_checkpoints/pool.npz"
FPNET="$HOME/casmi_checkpoints/fpnet_final.pt"
PROCESSED="$WORK/data/processed"
RUNS="$WORK/runs"
LOG="$WORK/full_validation.log"

mkdir -p "$PROCESSED" "$RUNS"
cd "$REPO"

if pgrep -f "run_full_validation" > /dev/null; then
    echo "ERROR: a validation run is already active."
    exit 1
fi

run_stage() {
    echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') STAGE: $1 ==="
}

{
    run_stage "1. Channel 1+2 baseline (reproduces prior 0.152-Kaggle-adjacent number, honest split)"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$SPLIT" --pool "$POOL" \
        --out "$RUNS/1_baseline_ch12"

    run_stage "2. + Channel 5 (fragmentation)"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$SPLIT" --pool "$POOL" \
        --out "$RUNS/2_plus_fragmentation" \
        --fragmentation

    run_stage "3. + Channel 4 (FPNet)"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$SPLIT" --pool "$POOL" \
        --out "$RUNS/3_plus_fpnet" \
        --fpnet-checkpoint "$FPNET" --fpnet-device cuda

    run_stage "4. + Channel 4 + Channel 5 together (pre-reranker, weighted fusion)"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$SPLIT" --pool "$POOL" \
        --out "$RUNS/4_plus_both" \
        --fragmentation --fpnet-checkpoint "$FPNET" --fpnet-device cuda

    run_stage "5. Build reranker training data (all 4 channels' features)"
    .venv/bin/python scripts/build_rank_train.py \
        --train "$DATA" --split "$SPLIT" --pool "$POOL" \
        --out "$PROCESSED/rank_train.npz" \
        --fragmentation --fpnet-checkpoint "$FPNET" --fpnet-device cuda

    run_stage "6. Train the GBM reranker (2 priors x 4 seeds per RankerConfig)"
    .venv/bin/python scripts/train_ranker.py \
        --data "$PROCESSED/rank_train.npz" \
        --out "$WORK/checkpoints/ranker.pkl"

    run_stage "7. Full 4-channel + GBM reranker MRR@25 (the number to compare against 0.339)"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$SPLIT" --pool "$POOL" \
        --out "$RUNS/5_full_reranked" \
        --fragmentation --fpnet-checkpoint "$FPNET" --fpnet-device cuda \
        --ranker "$WORK/checkpoints/ranker.pkl"

    run_stage "DONE"
    echo "finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
} > "$LOG" 2>&1 < /dev/null &

sleep 3
echo "pid $(pgrep -f run_baseline.py | head -1 || pgrep -f build_rank_train.py | head -1 || echo 'starting')"
echo "log: $LOG"
echo "started at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

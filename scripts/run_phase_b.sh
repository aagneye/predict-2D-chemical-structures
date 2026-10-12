#!/usr/bin/env bash
# Phase B on the Azure box, detached: everything the public LB still lacks
# after phase A, in one dependency-ordered sequence.
#
#   1. popularity_v2.npz  — fill PubChemLite pool rows' popularity from the tier
#   2. rank_train_v2.npz  — reranker rows with ALL channels (library, analog,
#                           fragmentation, FPNet v2, popularity, PubChem tier)
#   3. cv_ranker          — grouped 5-fold out-of-fold MRR@25 (the honest number)
#                           + the final ranker_v2.pkl
#   4. run_baseline       — full hold-out and the NP panel with the reranker
#   5. build_kaggle_dataset.sh — stage everything for the next submission
#
# Waits for phase A to finish (same pgrep guard), then runs on GPU 3.
# Usage: run_phase_b.sh
set -euo pipefail

WORK=/mnt/casmi
REPO="$WORK/repo"
DATA="$WORK/data/raw/train.parquet"
EXT="$WORK/external"
PROCESSED="$WORK/data/processed"
RUNS="$WORK/runs"
KEEP="$HOME/casmi_checkpoints"
FPNET="$KEEP/fpnet_v2/fpnet_final.pt"
LOG="$WORK/logs/phase_b.log"
export CUDA_VISIBLE_DEVICES=3

mkdir -p "$PROCESSED" "$RUNS" "$KEEP" "$WORK/logs"
cd "$REPO"

stage() { echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') STAGE: $1 ==="; }

{
    stage "0. waiting for phase A to finish"
    while pgrep -f "run_phase_a.sh|build_pool.py|build_split.py" > /dev/null 2>&1 \
          || pgrep -f "run_baseline.py" > /dev/null 2>&1; do
        sleep 120
    done
    echo "phase A idle at $(date -u)"

    stage "1. popularity_v2 (PubChemLite rows from the tier's pc_ik14)"
    .venv/bin/python scripts/extend_popularity.py \
        --pool "$KEEP/pool_v2.npz" --popularity "$KEEP/popularity.npz" \
        --tier-popularity "$EXT/pubchem_pop" --out "$PROCESSED/popularity_v2.npz"
    cp "$PROCESSED/popularity_v2.npz" "$KEEP/popularity_v2.npz"

    PUBCHEM="--pubchem-tier $EXT/pubchem_tier --pubchem-popularity $EXT/pubchem_pop"
    COMMON="--train $DATA --split $KEEP/split_v2.npz --pool $KEEP/pool_v2.npz
            --fragmentation --fpnet-checkpoint $FPNET --fpnet-device cuda
            --popularity $PROCESSED/popularity_v2.npz $PUBCHEM"

    stage "2. reranker rows: all channels incl. PubChem tier"
    # shellcheck disable=SC2086
    .venv/bin/python scripts/build_rank_train.py $COMMON --out "$PROCESSED/rank_train_v2.npz"
    cp "$PROCESSED/rank_train_v2.npz" "$KEEP/"

    stage "3. grouped 5-fold CV + final ranker"
    .venv/bin/python scripts/cv_ranker.py \
        --data "$PROCESSED/rank_train_v2.npz" --split "$KEEP/split_v2.npz" \
        --out "$KEEP/ranker.pkl" --report "$RUNS/b_cv_ranker/summary.json"

    stage "4a. full hold-out: all channels + reranker"
    # shellcheck disable=SC2086
    .venv/bin/python scripts/run_baseline.py $COMMON --ranker "$KEEP/ranker.pkl" \
        --out "$RUNS/b1_full_reranked"

    stage "4b. full hold-out: all channels + reranker + popularity blend mu=0.15"
    # shellcheck disable=SC2086
    .venv/bin/python scripts/run_baseline.py $COMMON --ranker "$KEEP/ranker.pkl" \
        --popularity-mu 0.15 --out "$RUNS/b2_full_reranked_mu015"

    stage "4c. NP panel: all channels + reranker"
    # shellcheck disable=SC2086
    .venv/bin/python scripts/run_baseline.py $COMMON --ranker "$KEEP/ranker.pkl" \
        --cohort-keys "$KEEP/split_v2.panel_keys.txt" --out "$RUNS/b1_panel_reranked"

    stage "5. stage the Kaggle dataset"
    cp "$PROCESSED/popularity_v2.npz" "$KEEP/popularity.npz"
    bash scripts/build_kaggle_dataset.sh

    stage "DONE"
} > "$LOG" 2>&1 < /dev/null &

sleep 2
echo "log: $LOG"

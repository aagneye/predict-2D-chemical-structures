#!/usr/bin/env bash
# Phase A on the Azure box, detached: rebuild the candidate pool with COCONUT
# and PubChemLite (scorer-keyed), convert the PubChem popularity table, build
# the honest split (2,000 sampled structures + the whole enveda-np-examples
# library forced into class 2 as a timsTOF natural-product panel), then
# measure the channel-1+2 baseline with and without the popularity prior on
# both the full hold-out and the panel.
#
# Everything lands under $WORK (ephemeral /mnt!) and the artifacts worth
# keeping are copied to $KEEP (persistent home dir) at the end.
#
# Usage: run_phase_a.sh [WORKERS]
set -euo pipefail

WORKERS="${1:-48}"
WORK=/mnt/casmi
REPO="$WORK/repo"
DATA="$WORK/data/raw/train.parquet"
EXT="$WORK/external"
PROCESSED="$WORK/data/processed"
RUNS="$WORK/runs"
KEEP="$HOME/casmi_checkpoints"
LOG="$WORK/logs/phase_a.log"

mkdir -p "$PROCESSED" "$RUNS" "$KEEP" "$WORK/logs"
cd "$REPO"

if pgrep -f "build_pool.py|run_baseline.py|build_split.py" > /dev/null 2>&1; then
    echo "ERROR: a phase-A job is already active."
    pgrep -af "build_pool.py|run_baseline.py|build_split.py"
    exit 1
fi

stage() { echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') STAGE: $1 ==="; }

{
    stage "1. pool = COCONUT + PubChemLite + train (scorer keys, $WORKERS workers)"
    .venv/bin/python scripts/build_pool.py \
        --coconut "$EXT/coconut/coconut_candidates.csv" \
        --extra-smiles "$EXT/pubchemlite/pubchemlite_candidates.csv" \
        --train "$DATA" \
        --out "$PROCESSED/pool_v2.npz" --workers "$WORKERS"

    stage "2. popularity prior -> npz"
    .venv/bin/python - <<'PY'
from casmi.priors import PopularityPrior
p = PopularityPrior.from_csv("/mnt/casmi/external/popularity/pool_popularity.csv")
p.save("/mnt/casmi/data/processed/popularity.npz")
print(f"[OK] popularity prior: {len(p):,} structures")
PY

    stage "3. split: 2000 sampled + enveda-np-examples panel as class 2, class 3 steered off-pool"
    .venv/bin/python scripts/build_split.py \
        --train "$DATA" --out "$PROCESSED/split_v2.npz" --n-holdout 2000 \
        --database-keys "$PROCESSED/pool_v2.npz" \
        --holdout-lib enveda-np-examples

    stage "4a. baseline ch1+2 on the full hold-out (new pool)"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$PROCESSED/split_v2.npz" --pool "$PROCESSED/pool_v2.npz" \
        --out "$RUNS/a1_ch12_poolv2"

    stage "4b. + popularity feature"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$PROCESSED/split_v2.npz" --pool "$PROCESSED/pool_v2.npz" \
        --popularity "$PROCESSED/popularity.npz" \
        --out "$RUNS/a2_ch12_poolv2_pop"

    stage "4c. + popularity feature + post-rank blend mu=0.15"
    .venv/bin/python scripts/run_baseline.py \
        --train "$DATA" --split "$PROCESSED/split_v2.npz" --pool "$PROCESSED/pool_v2.npz" \
        --popularity "$PROCESSED/popularity.npz" --popularity-mu 0.15 \
        --out "$RUNS/a3_ch12_poolv2_pop_mu015"

    stage "5. the same three on the enveda-np-examples panel only"
    for variant in "a1:" "a2:--popularity $PROCESSED/popularity.npz" \
                   "a3:--popularity $PROCESSED/popularity.npz --popularity-mu 0.15"; do
        name="${variant%%:*}"; flags="${variant#*:}"
        # shellcheck disable=SC2086
        .venv/bin/python scripts/run_baseline.py \
            --train "$DATA" --split "$PROCESSED/split_v2.npz" --pool "$PROCESSED/pool_v2.npz" \
            --cohort-keys "$PROCESSED/split_v2.panel_keys.txt" $flags \
            --out "$RUNS/${name}_panel"
    done

    stage "6. keep the artifacts somewhere persistent"
    cp "$PROCESSED/pool_v2.npz" "$PROCESSED/popularity.npz" "$PROCESSED/split_v2.npz" \
       "$PROCESSED/split_v2.panel_keys.txt" "$KEEP/"
    ls -la "$KEEP"

    stage "DONE"
} > "$LOG" 2>&1 < /dev/null &

sleep 3
echo "pid $(pgrep -f build_pool.py | head -1 || echo starting)"
echo "log: $LOG"

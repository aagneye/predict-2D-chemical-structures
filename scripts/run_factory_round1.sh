#!/usr/bin/env bash
# Factory round 1 on the VM, detached. Builds the LB emulator, calibrates it on the
# three configurations Kaggle already scored (v2 0.152, v7 0.186, v8 0.252) by
# re-running each one's *own code* (git worktrees at the submitted commits), trains
# the 33-feature reranker with Channel 8 (analog derivation) on test-like rows, and
# predicts v9's public score — all before any Kaggle submission.
#
# Usage: bash scripts/run_factory_round1.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

WORK=/mnt/casmi
REPO="$WORK/repo"
DATA="$WORK/data/raw/train.parquet"
EXT="$WORK/external"
KEEP="$HOME/casmi_checkpoints"
RUNS="$WORK/runs/factory1"
LOG="$WORK/logs/factory1.log"
FPNET="$KEEP/fpnet_v2/fpnet_final.pt"
PY="$REPO/.venv/bin/python"

mkdir -p "$RUNS" "$EXT" "$WORK/logs" "$WORK/wt"
cd "$REPO"

# GPU with the most free memory (the VM is shared).
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
export CUDA_VISIBLE_DEVICES="$GPU"

stage() { echo "=== $(date -u '+%Y-%m-%dT%H:%M:%SZ') $1 ==="; }

{
  stage "0. tests on the VM (background) — GPU $GPU"
  ( "$PY" -m pytest tests/ -q -p no:cacheprovider -W ignore > "$WORK/logs/pytest_vm.log" 2>&1;
    echo "PYTEST_EXIT=$?" >> "$WORK/logs/pytest_vm.log" ) &

  stage "1. external candidate tables + popularity v1"
  # COCONUT membership = the np_coconut flag of pool_popularity.csv; PubChemLite = the
  # ik14 column of its annotation table (the old candidate-CSV dataset is no longer public).
  mkdir -p "$EXT/popularity"
  [ -f "$EXT/popularity/pubchemlite_annotations.parquet" ] || kaggle datasets download dmitriigluzdov/casmi26-pubchem-popularity-prior -f pubchemlite_annotations.parquet -p "$EXT/popularity" -q
  [ -f "$EXT/popularity/pool_popularity.csv" ] || kaggle datasets download dmitriigluzdov/casmi26-pubchem-popularity-prior -f pool_popularity.csv -p "$EXT/popularity" --unzip -q
  [ -f "$KEEP/popularity_v1.npz" ] || "$PY" -c "from casmi.priors import PopularityPrior as P; P.from_csv('$EXT/popularity/pool_popularity.csv').save('$KEEP/popularity_v1.npz')"

  stage "2. external keys + emulator split (200 molecules per class)"
  [ -f "$KEEP/external_keys.txt" ] || "$PY" scripts/build_external_keys.py \
      --csv "$EXT/popularity/pool_popularity.csv:np_coconut" \
      --keys "$EXT/popularity/pubchemlite_annotations.parquet:ik14" \
      --out "$KEEP/external_keys.txt" --workers 48
  "$PY" scripts/make_emulator_split.py --split "$KEEP/split_v3.npz" --train "$DATA" \
      --external-keys "$KEEP/external_keys.txt" --per-class 200 --out "$KEEP/split_emu.npz"

  stage "3. v9 reranker rows (Channel 8 + test-like pool) — background"
  ( "$PY" scripts/build_rank_train.py --train "$DATA" --split "$KEEP/split_v3.npz" \
        --pool "$KEEP/pool_v2.npz" --realistic-pool "$KEEP/external_keys.txt" \
        --fragmentation --fpnet-checkpoint "$FPNET" --fpnet-device cuda \
        --popularity "$KEEP/popularity_v2.npz" \
        --pubchem-tier "$EXT/pubchem_tier" --pubchem-popularity "$EXT/pubchem_pop" \
        --derivation --out "$KEEP/rank_train_v3.npz" > "$WORK/logs/rank_train_v3.log" 2>&1 &&
    "$PY" scripts/cv_ranker.py --data "$KEEP/rank_train_v3.npz" --split "$KEEP/split_v3.npz" \
        --out "$KEEP/ranker_v9.pkl" --report "$RUNS/v9_cv/summary.json" > "$WORK/logs/cv_v9.log" 2>&1;
    echo "V9_DONE=$?" >> "$WORK/logs/cv_v9.log" ) &

  stage "4. calibration runs with each submission's own code"
  git worktree add -f "$WORK/wt/v2" 33dd6ea >/dev/null 2>&1 || true
  git worktree add -f "$WORK/wt/v7" 2c35b8b >/dev/null 2>&1 || true
  ( cd "$WORK/wt/v2" && PYTHONPATH=src "$PY" scripts/run_baseline.py --train "$DATA" \
        --split "$KEEP/split_emu.npz" --pool "$KEEP/pool.npz" --out "$RUNS/e_v2" \
        > "$WORK/logs/e_v2.log" 2>&1 ) &
  ( cd "$WORK/wt/v7" && PYTHONPATH=src "$PY" scripts/run_baseline.py --train "$DATA" \
        --split "$KEEP/split_emu.npz" --pool "$KEEP/pool_v2.npz" --out "$RUNS/e_v7" \
        --fragmentation --fpnet-checkpoint "$FPNET" --fpnet-device cuda \
        --popularity "$KEEP/popularity_v1.npz" --popularity-mu 0.15 \
        > "$WORK/logs/e_v7.log" 2>&1 ) &
  ( "$PY" scripts/run_baseline.py --train "$DATA" --split "$KEEP/split_emu.npz" \
        --pool "$KEEP/pool_v2.npz" --out "$RUNS/e_v8" \
        --fragmentation --fpnet-checkpoint "$FPNET" --fpnet-device cuda \
        --popularity "$KEEP/popularity_v2.npz" --popularity-mu 0.15 \
        --pubchem-tier "$EXT/pubchem_tier" --pubchem-popularity "$EXT/pubchem_pop" \
        --ranker "$KEEP/ranker.pkl" > "$WORK/logs/e_v8.log" 2>&1 ) &
  wait %3 %4 %5 2>/dev/null || wait

  stage "5. emulator fit on v2/v7/v8"
  "$PY" scripts/lb_emulator.py \
      --calibrate "$RUNS/e_v2/summary.json=0.152" \
      --calibrate "$RUNS/e_v7/summary.json=0.186" \
      --calibrate "$RUNS/e_v8/summary.json=0.252" | tee "$KEEP/emulator_round1.txt"

  stage "6. wait for the v9 reranker, then predict v9"
  while ! grep -q "V9_DONE" "$WORK/logs/cv_v9.log" 2>/dev/null; do sleep 120; done
  tail -12 "$WORK/logs/cv_v9.log"
  "$PY" scripts/lb_emulator.py \
      --calibrate "$RUNS/e_v2/summary.json=0.152" \
      --calibrate "$RUNS/e_v7/summary.json=0.186" \
      --calibrate "$RUNS/e_v8/summary.json=0.252" \
      --predict "$RUNS/v9_cv/summary.json" | tee "$KEEP/emulator_round1.txt"
  cp -r "$RUNS" "$KEEP/factory1_runs"
  stage "FACTORY1_DONE"
} > "$LOG" 2>&1 < /dev/null &

sleep 2
echo "log: $LOG  (GPU $GPU)"

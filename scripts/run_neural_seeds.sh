#!/usr/bin/env bash
# Train seed copies of the public spectrum-candidate scorer on the VM and evaluate them.
# Third-party training code, so every step runs in the no-network sandbox with ~/.kaggle
# hidden. Waits for `neural_seed_ensemble.py prepare` to finish if it is still running.
#
# Usage: scripts/run_neural_seeds.sh <seed> [<seed> ...]
# Env: N=/mnt/casmi/neural (bundle/, work/, runs/), GPU=<index> (default: most free memory)
# Output: $N/runs/seed<k>/ (holdout + refit checkpoints), $N/eval.json, and
#         ~/casmi_checkpoints/neural_seeds/seed<k>/final_model_seed<k>.pt for the dataset.
set -euo pipefail

N=${N:-/mnt/casmi/neural}
PY=/mnt/casmi/base0467/venv/bin/python
REPO=$(cd "$(dirname "$0")/.." && pwd)
V1=/mnt/casmi/base0467/input/datasets/sanpier/casmi-spectrum-model-v1/casmi_spectrum_model_v1
KEEP=~/casmi_checkpoints/neural_seeds
mkdir -p "$N/empty_home" "$KEEP"

sandbox() {
  unshare -rmn --propagation private bash -c \
    'mount --bind "$0" "$HOME/.kaggle" && shift && exec "$@"' "$N/empty_home" _ "$@"
}

while pgrep -f "neural_seed_ensemble.py prepare" > /dev/null; do sleep 60; done
[ -f "$N/work/spectra.parquet" ] || { echo "prepare did not finish"; exit 1; }

GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
             | sort -t, -k2 -nr | head -1 | cut -d, -f1)}
echo "$(date -u +%FT%TZ) training seeds $* on GPU $GPU"
for s in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU sandbox "$PY" -I "$REPO/scripts/neural_seed_ensemble.py" train \
      --bundle "$N/bundle" --work "$N/work" --out "$N/runs/seed$s" --seed "$s" \
      > "$N/runs/seed$s.log" 2>&1 &
done
wait
echo "$(date -u +%FT%TZ) trained"

MODELS=()
for s in "$@"; do
  MODELS+=("$N/runs/seed$s")
  mkdir -p "$KEEP/seed$s"
  cp "$N/runs/seed$s/final_model.pt" "$KEEP/seed$s/final_model_seed$s.pt"
done
# v1's own holdout model joins the comparison when our cache reproduces its split.
"$PY" -I -c "import pandas as pd, sys; a = pd.read_parquet(sys.argv[1]); \
b = pd.read_parquet(sys.argv[2]); sys.exit(0 if a[['key','split']].equals(b[['key','split']]) else 1)" \
    "$N/work/molecules.parquet" "$V1/molecules.parquet" \
  && { echo "split matches v1"; MODELS=("$V1" "${MODELS[@]}"); } || echo "split differs from v1"
CUDA_VISIBLE_DEVICES=$GPU sandbox "$PY" -I "$REPO/scripts/neural_seed_ensemble.py" eval \
    --bundle "$N/bundle" --work "$N/work" --models "${MODELS[@]}" --out "$N/eval.json"
cp "$N/eval.json" "$KEEP/eval.json"
echo "$(date -u +%FT%TZ) NEURAL_SEEDS_DONE"

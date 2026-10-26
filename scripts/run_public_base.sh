#!/usr/bin/env bash
# Run a public Kaggle notebook (e.g. the 0.467 base) on the VM the way Kaggle would, in a
# sandbox with NO network: an unprivileged user + mount + network namespace (unshare -rmn)
# where a Kaggle-shaped tree is bind-mounted over /kaggle and ~/.kaggle is hidden.
#
# Inputs live in /mnt/casmi/base0467/input/datasets/<owner>/<slug> (see download_files.sh);
# the Python 3.12 environment matching Kaggle's versions is /mnt/casmi/base0467/venv.
#
# Usage: scripts/run_public_base.sh <notebook.ipynb> <run-name> [comp-dir]
#   comp-dir: folder with train.parquet, test.parquet, sample_submission.csv
#             (default /mnt/casmi/data/raw = the visible competition files). Point it at a
#             folder with a held-out test.parquet to score the notebook offline.
# Env: GPU=<index> (default: the GPU with the most free memory);
#      FORCE_FULL=1 skips the visible-test fast path of notebooks that have one.
set -euo pipefail

NB=$(readlink -f "$1")
RUN="$2"
COMP=$(readlink -f "${3:-/mnt/casmi/data/raw}")
ROOT=/mnt/casmi/base0467
VENV="$ROOT/venv"
OUT="$ROOT/runs/$RUN"
TREE="$OUT/kaggle"
COMP_SLUG=enveda-CASMI26-molecule-id-mass-spectra
GPU=${GPU:-$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
             | sort -t, -k2 -nr | head -1 | cut -d, -f1)}

mkdir -p "$TREE/input/competitions/$COMP_SLUG" "$TREE/working" "$OUT/empty_home"
DATASETS=()
for dir in "$ROOT"/input/datasets/*/*; do
  rel=${dir#"$ROOT/input/"}
  mkdir -p "$TREE/input/$rel"
  DATASETS+=("$rel")
done
cp "$NB" "$OUT/input.ipynb"
echo "run $RUN | notebook $NB | comp $COMP | GPU $GPU | ${#DATASETS[@]} datasets"

# Everything below runs without network; loopback is needed for the Jupyter kernel.
unshare -rmn --propagation private env \
    TREE="$TREE" COMP="$COMP" ROOT="$ROOT" OUT="$OUT" VENV="$VENV" COMP_SLUG="$COMP_SLUG" \
    DATASETS="${DATASETS[*]}" REAL_HOME="$HOME" CUDA_VISIBLE_DEVICES="$GPU" \
    CASMI_FORCE_FULL="${FORCE_FULL:-}" PYTHONHASHSEED=0 bash -c '
  set -euo pipefail
  ip link set lo up
  mount --bind "$TREE" /kaggle
  mount --bind "$COMP" "/kaggle/input/competitions/$COMP_SLUG"
  for rel in $DATASETS; do mount --bind "$ROOT/input/$rel" "/kaggle/input/$rel"; done
  [ -d "$REAL_HOME/.kaggle" ] && mount --bind "$OUT/empty_home" "$REAL_HOME/.kaggle"
  cd /kaggle/working
  export HOME=/kaggle/working JUPYTER_RUNTIME_DIR=/kaggle/working/.jupyter_runtime
  "$VENV/bin/papermill" "$OUT/input.ipynb" "$OUT/output.ipynb" --cwd /kaggle/working \
      --log-output --no-progress-bar
' > "$OUT/run.log" 2>&1 \
  && echo "RUN_OK $RUN" | tee -a "$OUT/run.log" \
  || { echo "RUN_FAILED $RUN (see $OUT/run.log)" | tee -a "$OUT/run.log"; exit 1; }

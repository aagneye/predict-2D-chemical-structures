#!/usr/bin/env bash
# Provision the Azure 4x T4 box for CASMI training.
#
# Runs on /mnt (2.6 TB free) because the root disk is at 97%. Note that /mnt is
# Azure's ephemeral disk: its contents are LOST if the VM is deallocated, so
# checkpoints must be copied to ~/ or pulled down before shutting the VM off.
set -euo pipefail

WORK=/mnt/casmi
export PATH="$HOME/.local/bin:$PATH"

echo "=== workspace ==="
sudo mkdir -p "$WORK"
sudo chown "$(id -un):$(id -gn)" "$WORK"
df -h /mnt | tail -1

echo "=== uv ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
fi
uv --version

echo "=== repo ==="
cd "$WORK"
if [ -d repo/.git ]; then
    cd repo && git fetch -q origin && git reset -q --hard origin/main
else
    git clone -q https://github.com/aagneye/predict-2D-chemical-structures.git repo
    cd repo
fi
git log --oneline -n 1

echo "=== venv (uv fetches python 3.12; system only has 3.10) ==="
uv venv --python 3.12 --quiet
# Core deps first, then CUDA torch from the cu121 index (T4 = compute 7.5).
uv pip install -q -e ".[dev]"
uv pip install -q torch --index-url https://download.pytorch.org/whl/cu121

echo "=== verify ==="
.venv/bin/python - <<'PY'
import torch, rdkit, numpy, numba
print("torch      ", torch.__version__)
print("cuda avail ", torch.cuda.is_available())
print("gpu count  ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name} {p.total_memory/1e9:.1f} GB")
print("rdkit      ", rdkit.__version__)
print("numpy      ", numpy.__version__)
print("numba      ", numba.__version__)
PY

echo "=== done ==="

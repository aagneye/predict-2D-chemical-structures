#!/usr/bin/env bash
# Push a kernel folder to Kaggle, retrying while the weekly GPU quota is exhausted, then hand
# over to submit_kernel_when_done.sh. Runs on the VM; detach it with nohup.
#
# Usage: scripts/push_when_quota.sh <kernel-dir> <kernel-version> "<message>" [retry-minutes]
#   kernel-version: the version this push will create (1 for a new kernel).
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

DIR="$1"
VERSION="$2"
MESSAGE="$3"
RETRY_MINUTES="${4:-30}"
KERNEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['id'])" \
         "$DIR/kernel-metadata.json")

while true; do
  out=$(kaggle kernels push -p "$DIR" 2>&1 | grep -v -e outdated -e 'does not resolve')
  echo "$(date -u +%FT%TZ) $out"
  case "$out" in
    *quota*) sleep $((RETRY_MINUTES * 60));;
    *error*|*Error*) echo "PUSH FAILED"; exit 1;;
    *) break;;
  esac
done
# Give Kaggle a moment to register the new version before polling its status.
sleep 120
exec "$(dirname "$0")/submit_kernel_when_done.sh" "$KERNEL" "$VERSION" "$MESSAGE" 600

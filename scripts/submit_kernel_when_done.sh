#!/usr/bin/env bash
# Poll a Kaggle kernel and submit its submission.csv to the competition once it completes.
# Runs on the VM (kaggle CLI in ~/.local/bin); detach it with nohup so it survives the session.
#
# Usage: scripts/submit_kernel_when_done.sh <kernel-id> <kernel-version> "<message>" [max-minutes]
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"

KERNEL="$1"
VERSION="$2"
MESSAGE="$3"
MAX_MINUTES="${4:-480}"
COMPETITION=enveda-CASMI26-molecule-id-mass-spectra

for _ in $(seq 1 "$MAX_MINUTES"); do
  status=$(kaggle kernels status "$KERNEL" 2>&1 | grep -v outdated | tail -1)
  echo "$(date -u +%FT%TZ) $status"
  case "$status" in
    *COMPLETE*)
      kaggle competitions submit "$COMPETITION" -k "$KERNEL" -f submission.csv -v "$VERSION" \
          -m "$MESSAGE" 2>&1 | grep -v outdated | tail -3
      echo "SUBMITTED $KERNEL v$VERSION"
      exit 0;;
    *ERROR*|*CANCEL*)
      echo "KERNEL FAILED, not submitting"
      exit 1;;
  esac
  sleep 60
done
echo "timed out waiting for $KERNEL"
exit 2

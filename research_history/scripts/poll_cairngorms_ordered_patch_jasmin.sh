#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-ordered-patch"

ssh "${REMOTE_HOST}" \
  "squeue -u ameer096 -o '%.18i %.12P %.24j %.2t %.10M %.10l %R'; \
   printf '\ncompleted result files: '; \
   find '${REMOTE_ROOT}/data/interim/cairngorms_ordered_patch_results' -name '*.npz' -type f | wc -l; \
   printf 'latest logs:\n'; \
   find '${REMOTE_ROOT}/logs' -name '*.out' -type f -print0 | xargs -0 -r ls -1t 2>/dev/null | head -3"

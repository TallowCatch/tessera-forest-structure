#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm03"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ ! -f metadata/phase31_job_ids.txt ]]; then
  echo "Phase 31 has not been submitted"
  exit 0
fi
source metadata/phase31_job_ids.txt
echo "=== LOTUS jobs ==="
squeue -j "${train_job},${aggregate_job}" -o "%.18i %.28j %.2t %.10M %.10l %R" || true
echo "=== strict U-Net runs ==="
complete=$(find data/interim/phase31_cairngorms_v2_height_strict_results -name '*.npz' 2>/dev/null | wc -l)
echo "${complete}/15 complete"
errors=$(find logs -name 'phase31_*.err' -type f -size +0c 2>/dev/null | wc -l)
echo "non-empty error logs: ${errors}"
echo "=== recent training ==="
log=$(ls -1t logs/phase31_train_*.out 2>/dev/null | head -n 1 || true)
[[ -n "${log}" ]] && { echo "${log}"; tail -n 8 "${log}"; } || echo "waiting"
echo "=== final report ==="
if [[ -f outputs/reports/phase31_cairngorms_v2_height_strict.md ]]; then
  sed -n '1,120p' outputs/reports/phase31_cairngorms_v2_height_strict.md
else
  echo "not complete"
fi
REMOTE_SCRIPT

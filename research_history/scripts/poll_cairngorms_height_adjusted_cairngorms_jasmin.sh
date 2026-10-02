#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -u
cd "${ROOT}"
if [[ ! -f metadata/phase23_job_ids.txt ]]; then echo "Phase 23 has not reached LOTUS submission yet"; exit 0; fi
source metadata/phase23_job_ids.txt
echo "=== jobs ==="; cat metadata/phase23_job_ids.txt
echo "=== queue ==="
squeue -j "${prepare_job},${train_job},${aggregate_job}" -o '%.20i %.24j %.2t %.10M %.10l %R' 2>/dev/null || true
echo "=== preparation ==="
if [[ -f metadata/phase23_cairngorms_preparation_status.json ]]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase23_cairngorms_preparation_status.json'))
print(f"complete: {x['rows']:,} rows; {len(x['targets'])} adjusted targets")
print('folds:',x['fold_counts'])
PY
else
  echo "not complete"
  tail -n 8 "logs/phase23_prepare_${prepare_job}.out" 2>/dev/null || true
  tail -n 8 "logs/phase23_prepare_${prepare_job}.err" 2>/dev/null || true
fi
echo "=== model results ==="
completed=$(find data/interim/phase23_cairngorms_height_adjusted_results -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${completed}/45 complete"
latest=$(find logs -name "phase23_train_${train_job}_*.out" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -n "${latest}" ]]; then echo "latest: ${latest}"; tail -n 7 "${latest}"; fi
errors=$(find logs \( -name "phase23_prepare_${prepare_job}.err" -o -name "phase23_train_${train_job}_*.err" -o -name "phase23_aggregate_${aggregate_job}.err" \) -type f -size +0c 2>/dev/null | wc -l | tr -d ' ')
echo "non-empty error logs: ${errors}"
echo "=== final report ==="
if [[ -f metadata/phase23_cairngorms_result_freeze.json ]]; then cat metadata/phase23_cairngorms_result_freeze.json; else echo "not complete"; fi
REMOTE_SCRIPT

#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -u
cd "${ROOT}"
if [[ ! -f metadata/phase25_job_ids.txt ]]; then echo "Phase 25 has not been submitted"; exit 0; fi
source metadata/phase25_job_ids.txt
echo "=== jobs ==="; cat metadata/phase25_job_ids.txt
echo "=== queue ==="
squeue -j "${prepare_job},${train_job},${aggregate_job}" -o '%.20i %.24j %.2t %.10M %.10l %R' 2>/dev/null || true
echo "=== preparation ==="
if [[ -f metadata/phase25_cairngorms_corrected_lidar_preparation.json ]]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase25_cairngorms_corrected_lidar_preparation.json'))
print(f"complete: {x['rows_valid']:,}/{x['rows_source']:,} rows")
print('changed bands:', ', '.join(x['replacement_verification']['changed_bands']))
print('folds:', x['fold_counts'])
PY
else
  echo "not complete"
  tail -n 10 "logs/phase25_prepare_${prepare_job}.out" 2>/dev/null || true
  tail -n 10 "logs/phase25_prepare_${prepare_job}.err" 2>/dev/null || true
fi
echo "=== model results ==="
completed=$(find data/interim/phase25_cairngorms_corrected_lidar_results -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${completed}/45 complete"
latest=$(find logs -name "phase25_train_${train_job}_*.out" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -n "${latest}" ]]; then echo "latest: ${latest}"; tail -n 8 "${latest}"; fi
errors=$(find logs \( -name "phase25_prepare_${prepare_job}.err" -o -name "phase25_train_${train_job}_*.err" -o -name "phase25_aggregate_${aggregate_job}.err" \) -type f -size +0c 2>/dev/null | wc -l | tr -d ' ')
echo "non-empty error logs: ${errors}"
echo "=== final report ==="
if [[ -f metadata/phase25_cairngorms_corrected_lidar_result_freeze.json ]]; then
  cat metadata/phase25_cairngorms_corrected_lidar_result_freeze.json
else
  echo "not complete"
fi
REMOTE_SCRIPT


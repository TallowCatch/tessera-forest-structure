#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${REMOTE_HOST}" "ROOT='${REMOTE_ROOT}' bash -s" <<'REMOTE'
set -u
cd "${ROOT}"

prepare_job=""
train_job=""
aggregate_job=""
if [[ -f metadata/phase21_job_ids.txt ]]; then
  source metadata/phase21_job_ids.txt
  echo "=== job identifiers ==="
  cat metadata/phase21_job_ids.txt
else
  echo "Phase 21 has not been submitted"
  exit 0
fi

echo "=== LOTUS queue ==="
squeue -j "${prepare_job},${train_job},${aggregate_job}" \
  -o '%.18i %.22j %.2t %.10M %.10l %R' 2>/dev/null || true

echo "=== preparation ==="
if [[ -f metadata/phase21_cairngorms_preparation_status.json ]]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase21_cairngorms_preparation_status.json'))
print('complete:', x['rows_valid'], '/', x['rows_total'], 'valid rows')
print('test rows:', x['fold_test_rows'])
print('top-height agreement:', round(x['top_height_agreement']['spearman_r'], 3))
PY
else
  echo "not complete"
  tail -n 8 "logs/phase21_prepare_${prepare_job}.out" 2>/dev/null || true
  tail -n 8 "logs/phase21_prepare_${prepare_job}.err" 2>/dev/null || true
fi

echo "=== model runs ==="
completed=$(find data/interim/phase21_cairngorms_surface_results -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${completed}/45 complete"
latest=$(find logs -name "phase21_train_${train_job}_*.out" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -n "${latest}" ]]; then
  echo "latest log: ${latest}"
  tail -n 8 "${latest}"
fi

errors=0
errors=$((errors + $(find logs -name "phase21_prepare_${prepare_job}.err" -type f -size +0c 2>/dev/null | wc -l)))
errors=$((errors + $(find logs -name "phase21_train_${train_job}_*.err" -type f -size +0c 2>/dev/null | wc -l)))
errors=$((errors + $(find logs -name "phase21_aggregate_${aggregate_job}.err" -type f -size +0c 2>/dev/null | wc -l)))
echo "non-empty error logs: ${errors}"

echo "=== final result ==="
if [[ -f metadata/phase21_cairngorms_result_freeze.json ]]; then
  cat metadata/phase21_cairngorms_result_freeze.json
else
  echo "not complete"
fi
REMOTE

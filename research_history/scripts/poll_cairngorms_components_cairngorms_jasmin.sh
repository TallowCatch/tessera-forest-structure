#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${REMOTE_HOST}" "ROOT='${REMOTE_ROOT}' bash -s" <<'REMOTE'
set -u
cd "${ROOT}"

echo "=== remote acquisition ==="
if [[ -f metadata/phase20_orchestrator.pid ]] && kill -0 "$(cat metadata/phase20_orchestrator.pid)" 2>/dev/null; then
  echo "orchestrator running, PID $(cat metadata/phase20_orchestrator.pid)"
else
  echo "download/submission stage finished"
fi
declare -a names=(
  "chm_1m:data/raw/scotland_lidar/chm/Cairngorms_standard_CHM_DSM_DTM_1m_res.tif:6007617253"
  "slope_mask:data/raw/scotland_lidar/chm/Cairngorms_slope_mask_45_1m.tif:101932911"
  "slope_change_mask:data/raw/scotland_lidar/chm/Cairngorms_slope_of_slope_mask_84_5_1m.tif:114554275"
)
for item in "${names[@]}"; do
  IFS=: read -r name path expected <<< "${item}"
  actual=0
  [[ -f "${path}" ]] && actual=$(stat -c %s "${path}")
  [[ -f "${path}.part" ]] && actual=$(stat -c %s "${path}.part")
  percent=$(awk -v a="${actual}" -v e="${expected}" 'BEGIN {printf "%.1f", 100*a/e}')
  printf '%-18s %12s / %12s bytes (%s%%)\n' "${name}" "${actual}" "${expected}" "${percent}"
done
if [[ -f metadata/phase20_orchestrator.pid ]] && kill -0 "$(cat metadata/phase20_orchestrator.pid)" 2>/dev/null; then
  tail -n 8 logs/phase20_orchestrator.log 2>/dev/null || true
elif [[ -f metadata/phase20_download_status.txt ]]; then
  tail -n 8 metadata/phase20_download_status.txt
fi

echo "=== LOTUS jobs ==="
squeue -u ameer096 -o '%.18i %.22j %.2t %.10M %.10l %R' | grep -E 'JOBID|cairn_p20' || true
prepare_job=""
train_job=""
aggregate_job=""
if [[ -f metadata/phase20_job_ids.txt ]]; then
  source metadata/phase20_job_ids.txt
  cat metadata/phase20_job_ids.txt
fi

echo "=== preparation ==="
if [[ -f metadata/phase20_cairngorms_preparation_status.json ]]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase20_cairngorms_preparation_status.json'))
print('complete:', x['rows_valid'], '/', x['rows_total'], 'valid rows')
print('fold test rows:', x['fold_test_rows'])
print('target agreement:', {k: round(v['spearman_r'], 3) for k,v in x['supplied_recomputed_comparison'].items()})
PY
else
  echo "not complete"
  [[ -n "${prepare_job}" ]] && tail -n 8 "logs/phase20_prepare_${prepare_job}.out" 2>/dev/null || true
  [[ -n "${prepare_job}" ]] && tail -n 8 "logs/phase20_prepare_${prepare_job}.err" 2>/dev/null || true
fi

echo "=== model runs ==="
completed=$(find data/interim/phase20_cairngorms_component_results -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${completed}/45 complete"
latest=""
[[ -n "${train_job}" ]] && latest=$(find logs -name "phase20_train_${train_job}_*.out" -type f 2>/dev/null | sort | tail -n 1)
[[ -n "${latest}" ]] && { echo "latest log: ${latest}"; tail -n 6 "${latest}"; }
errors=0
if [[ -n "${prepare_job}" ]]; then
  errors=$((errors + $(find logs -name "phase20_prepare_${prepare_job}.err" -type f -size +0c 2>/dev/null | wc -l)))
fi
if [[ -n "${train_job}" ]]; then
  errors=$((errors + $(find logs -name "phase20_train_${train_job}_*.err" -type f -size +0c 2>/dev/null | wc -l)))
fi
if [[ -n "${aggregate_job}" ]]; then
  errors=$((errors + $(find logs -name "phase20_aggregate_${aggregate_job}.err" -type f -size +0c 2>/dev/null | wc -l)))
fi
echo "non-empty error logs: ${errors}"

echo "=== final report ==="
if [[ -f metadata/phase20_cairngorms_result_freeze.json ]]; then
  cat metadata/phase20_cairngorms_result_freeze.json
else
  echo "not complete"
fi
REMOTE

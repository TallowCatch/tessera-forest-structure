#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
LOCAL_LOG="${ROOT}/outputs/reports/cairngorms_dense_10m_deploy.log"
LOCAL_PID="${ROOT}/outputs/reports/cairngorms_dense_10m_deploy.pid"

echo "=== local deployment ==="
if [[ -f "${LOCAL_PID}" ]] && kill -0 "$(cat "${LOCAL_PID}")" 2>/dev/null; then
  echo "running PID $(cat "${LOCAL_PID}")"
  if [[ -f "${LOCAL_LOG}" ]]; then
    tail -n 8 "${LOCAL_LOG}"
  fi
else
  echo "upload/submission complete; computation is managed by JASMIN"
fi

echo "=== JASMIN jobs ==="
ssh "${REMOTE_HOST}" \
  "squeue -u ameer096 -o '%.18i %.22j %.2t %.10M %.10l %R' | grep -E 'JOBID|cairn_dense' || true"

echo "=== preparation ==="
ssh "${REMOTE_HOST}" \
  "if [[ -f '${REMOTE_ROOT}/metadata/cairngorms_dense_10m_preparation_status.json' ]]; then cat '${REMOTE_ROOT}/metadata/cairngorms_dense_10m_preparation_status.json'; else echo 'status file not created yet'; fi"

echo "=== completed model runs ==="
ssh "${REMOTE_HOST}" \
  "printf 'results '; find '${REMOTE_ROOT}/data/interim/cairngorms_dense_10m_results' -name '*.npz' 2>/dev/null | wc -l; printf 'expected 45\n'; printf 'current nonempty errors '; find '${REMOTE_ROOT}/logs' -maxdepth 1 -name '*.err' -size +0c 2>/dev/null | wc -l"

echo "=== final result ==="
ssh "${REMOTE_HOST}" \
  "if [[ -f '${REMOTE_ROOT}/metadata/cairngorms_dense_10m_result_freeze.json' ]]; then python - <<'PY'
import json
p='${REMOTE_ROOT}/metadata/cairngorms_dense_10m_result_freeze.json'
x=json.load(open(p))
print('complete')
print('selected_model', x['selected_model'])
for row in x['primary_metrics']:
    print(row['model'], 'RMSE', round(row['rmse'],6), 'R2', round(row['r2'],4), 'Spearman', round(row['spearman_r'],4))
PY
else echo 'not complete'; fi"

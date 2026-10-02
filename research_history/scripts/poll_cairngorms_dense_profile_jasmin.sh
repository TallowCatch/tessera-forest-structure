#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

echo "=== JASMIN profile jobs ==="
ssh "${REMOTE_HOST}" \
  "squeue -u ameer096 -o '%.18i %.22j %.2t %.10M %.10l %R' | grep -E 'JOBID|cairn_prof' || true"

echo "=== preparation ==="
ssh "${REMOTE_HOST}" \
  "if [[ -f '${REMOTE_ROOT}/metadata/cairngorms_dense_profile_preparation_status.json' ]]; then cat '${REMOTE_ROOT}/metadata/cairngorms_dense_profile_preparation_status.json'; else echo 'status file not created yet'; fi"

echo "=== completed model runs ==="
ssh "${REMOTE_HOST}" \
  "printf 'results '; find '${REMOTE_ROOT}/data/interim/cairngorms_dense_profile_results' -name '*.npz' 2>/dev/null | wc -l; printf 'expected 45\n'; printf 'nonempty profile errors '; find '${REMOTE_ROOT}/logs' -maxdepth 1 -name 'profile_*.err' -size +0c 2>/dev/null | wc -l"

echo "=== latest training output ==="
ssh "${REMOTE_HOST}" \
  "latest=\$(find '${REMOTE_ROOT}/logs' -maxdepth 1 -type f \( -name 'profile_scalar_*.out' -o -name 'profile_unet_*.out' \) -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-); if [[ -n \"\${latest}\" ]]; then echo \"\${latest##*/}\"; tail -n 8 \"\${latest}\"; else echo 'training has not started'; fi"

echo "=== final result ==="
ssh "${REMOTE_HOST}" \
  "if [[ -f '${REMOTE_ROOT}/metadata/cairngorms_dense_profile_result_freeze.json' ]]; then python - <<'PY'
import json
p='${REMOTE_ROOT}/metadata/cairngorms_dense_profile_result_freeze.json'
x=json.load(open(p))
print('complete')
print('selected_model', x['selected_model'])
for row in x['pooled_records']:
    print(row['model'], 'JS distance', round(row['mean_jensen_shannon_distance'], 5), 'profile RMSE', round(row['profile_rmse'], 5), 'entropy R2', round(row['entropy_r2'], 4), 'incremental R2 over height', round(row['incremental_r2_over_lidar_height'], 4))
PY
else echo 'not complete'; fi"

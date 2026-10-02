#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -u
cd "${ROOT}"
if [[ ! -f metadata/phase22_job_ids.txt ]]; then
  echo "Phase 22 has not reached LOTUS submission yet"
  exit 0
fi
source metadata/phase22_job_ids.txt
echo "=== jobs ==="
cat metadata/phase22_job_ids.txt
echo "=== queue ==="
squeue -j "${prepare_job},${scalar_job},${unet_job},${aggregate_job}" -o '%.20i %.24j %.2t %.10M %.10l %R' 2>/dev/null || true
echo "=== preparation ==="
if [[ -f metadata/phase22_cairngorms_preparation_status.json ]]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase22_cairngorms_preparation_status.json'))
print(f"complete: {x['rows_valid']:,}/{x['rows_source']:,} valid rows")
for key, value in x['fold_rows'].items(): print(key, value)
PY
else
  echo "not complete"
  tail -n 8 "logs/phase22_prepare_${prepare_job}.out" 2>/dev/null || true
  tail -n 8 "logs/phase22_prepare_${prepare_job}.err" 2>/dev/null || true
fi
echo "=== model results ==="
completed=$(find data/interim/phase22_cairngorms_spatial_unet_results -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${completed}/180 complete (150 scalar/CNN + 30 U-Net)"
for kind in scalar unet; do
  job_var="${kind}_job"
  job_id="${!job_var}"
  latest=$(find logs -name "phase22_${kind}_${job_id}_*.out" -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)
  if [[ -n "${latest}" ]]; then
    echo "latest ${kind}: ${latest}"
    tail -n 6 "${latest}"
  fi
done
errors=$(find logs \( -name "phase22_prepare_${prepare_job}.err" -o -name "phase22_scalar_${scalar_job}_*.err" -o -name "phase22_unet_${unet_job}_*.err" -o -name "phase22_aggregate_${aggregate_job}.err" \) -type f -size +0c 2>/dev/null | wc -l | tr -d ' ')
echo "non-empty error logs: ${errors}"
echo "=== final report ==="
if [[ -f metadata/phase22_cairngorms_result_freeze.json ]]; then
  cat metadata/phase22_cairngorms_result_freeze.json
else
  echo "not complete"
fi
REMOTE_SCRIPT

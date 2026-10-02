#!/usr/bin/env bash
set -euo pipefail

HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ ! -f metadata/phase26_ahn4_job_ids.txt ]]; then
  echo "Phase 26 replication has not been submitted"
  exit 0
fi
source metadata/phase26_ahn4_job_ids.txt
echo "=== LOTUS jobs ==="
squeue -j "${prepare_job},${tessera_job},${conventional_job},${evaluate_job}" \
  -o "%.18i %.24j %.2t %.10M %.10l %R" || true
echo "=== accounting ==="
sacct -j "${prepare_job},${tessera_job},${conventional_job},${evaluate_job}" \
  --format=JobID,JobName%24,State,Elapsed,ExitCode -n -X || true
for stage in prepare tessera conventional evaluate; do
  job_variable="${stage}_job"
  job="${!job_variable}"
  echo "=== ${stage} log (${job}) ==="
  tail -n 12 "logs/phase26_${stage}_${job}.out" 2>/dev/null || echo "not started"
  if [[ -s "logs/phase26_${stage}_${job}.err" ]]; then
    echo "--- ${stage} errors ---"
    tail -n 12 "logs/phase26_${stage}_${job}.err"
  fi
done
echo "=== artifacts ==="
for path in \
  metadata/phase26_ahn4_design_freeze.json \
  data/processed/phase26_ahn4_tessera.parquet \
  data/processed/phase26_ahn4_conventional.parquet \
  metadata/phase26_ahn4_result_freeze.json; do
  if [[ -f "${path}" ]]; then stat -c '%n %s bytes' "${path}"; else echo "missing ${path}"; fi
done
REMOTE_SCRIPT

#!/usr/bin/env bash
set -euo pipefail

HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ ! -f metadata/phase27_job_ids.txt ]]; then
  echo "Phase 27 has not been submitted"
  exit 0
fi
source metadata/phase27_job_ids.txt
jobs="${cairngorms_acquire_job},${savelsbos_acquire_job},${cairngorms_train_job},${savelsbos_evaluate_job},${cairngorms_aggregate_job}"
echo "=== LOTUS jobs ==="
squeue -j "${jobs}" -o "%.18i %.30j %.2t %.10M %.10l %R" || true
echo "=== accounting ==="
sacct -j "${jobs}" --format=JobID,JobName%30,State,Elapsed,ExitCode -n -X || true
for pair in \
  "cairngorms_acquire:${cairngorms_acquire_job}" \
  "savelsbos_acquire:${savelsbos_acquire_job}" \
  "savelsbos_evaluate:${savelsbos_evaluate_job}" \
  "cairngorms_aggregate:${cairngorms_aggregate_job}"; do
  stage=${pair%%:*}; job=${pair##*:}
  echo "=== ${stage} ==="
  tail -n 10 "logs/phase27_${stage}_${job}.out" 2>/dev/null || echo "not started"
  if [[ -s "logs/phase27_${stage}_${job}.err" ]]; then
    echo "--- errors ---"
    tail -n 10 "logs/phase27_${stage}_${job}.err"
  fi
done
echo "=== Cairngorms model runs ==="
complete=$(find data/interim/phase27_cairngorms_v2_results -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${complete}/30 complete"
find logs -maxdepth 1 -name "phase27_cairngorms_train_${cairngorms_train_job}_*.out" -type f -print0 2>/dev/null \
  | xargs -0 tail -n 1 2>/dev/null | tail -n 8 || true
echo "=== Final artifacts ==="
for path in \
  metadata/phase27_cairngorms_v2_result_freeze.json \
  metadata/phase27_savelsbos_v2_result_freeze.json \
  outputs/tables/phase27_cairngorms_v1_v2_comparison.csv \
  outputs/tables/phase27_savelsbos_v1_v2_comparison.csv; do
  if [[ -f "${path}" ]]; then stat -c '%n %s bytes' "${path}"; else echo "missing ${path}"; fi
done
REMOTE_SCRIPT

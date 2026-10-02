#!/usr/bin/env bash
set -euo pipefail

HOST="jasmin-sci-vm03"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ ! -f metadata/phase28_job_ids.txt ]]; then
  echo "Phase 28 has not been submitted"
  exit 0
fi
source metadata/phase28_job_ids.txt
jobs="${acquire_job},${prepare_job},${train_job},${aggregate_job}"
echo "=== LOTUS jobs ==="
squeue -j "${jobs}" -o "%.18i %.28j %.2t %.10M %.10l %R" || true
echo "=== accounting ==="
sacct -j "${jobs}" --format=JobID,JobName%28,State,Elapsed,ExitCode -n -X || true
echo "=== dense v2 acquisition ==="
complete=$(find data/interim/phase28_cairngorms_v2_dense/tiles -maxdepth 1 -name 'grid_*.json' 2>/dev/null | wc -l | tr -d ' ')
expected=$(python - <<'PY'
import json
from pathlib import Path
p = Path("metadata/phase28_cairngorms_v2_dense_acquisition.json")
print(len(json.loads(p.read_text())["tiles"]) if p.exists() else 17)
PY
)
echo "${complete}/${expected} tiles complete"
tail -n 12 "logs/phase28_acquire_${acquire_job}.out" 2>/dev/null || echo "not started"
if [[ -s "logs/phase28_acquire_${acquire_job}.err" ]]; then
  echo "--- acquisition errors ---"
  tail -n 12 "logs/phase28_acquire_${acquire_job}.err"
fi
echo "=== preparation ==="
tail -n 8 "logs/phase28_prepare_${prepare_job}.out" 2>/dev/null || echo "not started"
if [[ -s "logs/phase28_prepare_${prepare_job}.err" ]]; then
  echo "--- preparation errors ---"
  tail -n 10 "logs/phase28_prepare_${prepare_job}.err"
fi
echo "=== U-Net runs ==="
complete=$(find data/interim/phase28_cairngorms_v2_unet_results -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo "${complete}/15 complete"
find logs -maxdepth 1 -name "phase28_train_${train_job}_*.out" -type f -print0 2>/dev/null \
  | xargs -0 tail -n 1 2>/dev/null | tail -n 8 || true
errors=$(find logs -maxdepth 1 -name "phase28_train_${train_job}_*.err" -type f -size +0c 2>/dev/null | wc -l | tr -d ' ')
echo "non-empty training error logs: ${errors}"
echo "=== final result ==="
tail -n 20 "logs/phase28_aggregate_${aggregate_job}.out" 2>/dev/null || echo "not started"
if [[ -f metadata/phase28_cairngorms_v2_unet_result_freeze.json ]]; then
  python - <<'PY'
import json
d=json.load(open("metadata/phase28_cairngorms_v2_unet_result_freeze.json"))
print("decision:", d["decision"])
print("primary:", d["primary_result"])
PY
else
  echo "not complete"
fi
REMOTE_SCRIPT

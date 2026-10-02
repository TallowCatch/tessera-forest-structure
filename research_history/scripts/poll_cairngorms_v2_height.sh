#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm03"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ ! -f metadata/phase30_job_ids.txt ]]; then
  echo "Phase 30 has not been submitted"
  exit 0
fi
source metadata/phase30_job_ids.txt
echo "=== LOTUS jobs ==="
squeue -j "${prepare_job},${train_job},${aggregate_job}" -o "%.18i %.28j %.2t %.10M %.10l %R" || true
echo "=== preparation ==="
if [[ -f metadata/phase30_cairngorms_v2_height_preparation.json ]]; then
  python - <<'PY'
import json
p=json.load(open('metadata/phase30_cairngorms_v2_height_preparation.json'))
print(f"complete: {p['rows']:,} rows; targets={', '.join(p['targets'])}")
PY
else
  log=$(ls -1t logs/phase30_prepare_*.out 2>/dev/null | head -n 1 || true)
  [[ -n "${log}" ]] && tail -n 8 "${log}" || echo "waiting"
fi
echo "=== model runs ==="
mlp=$(find data/interim/phase30_cairngorms_v2_height_results -name 'tessera_v2_mlp_5x5_*.npz' 2>/dev/null | wc -l)
unet=$(find data/interim/phase30_cairngorms_v2_height_results -name 'tessera_v2_unet_audited_*.npz' 2>/dev/null | wc -l)
echo "V2 MLP ${mlp}/15; audited V2 U-Net ${unet}/15; total $((mlp + unet))/30"
errors=$(find logs -name 'phase30_*.err' -type f -size +0c 2>/dev/null | wc -l)
echo "non-empty error logs: ${errors}"
echo "=== recent training ==="
log=$(ls -1t logs/phase30_train_*.out 2>/dev/null | head -n 1 || true)
[[ -n "${log}" ]] && { echo "${log}"; tail -n 8 "${log}"; } || echo "waiting"
echo "=== final report ==="
if [[ -f outputs/reports/phase30_cairngorms_v2_height.md ]]; then
  sed -n '1,100p' outputs/reports/phase30_cairngorms_v2_height.md
else
  echo "not complete"
fi
REMOTE_SCRIPT

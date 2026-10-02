#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ ! -f metadata/phase29_job_ids.txt ]]; then
  echo "Phase 29 has not been submitted"
  exit 0
fi
source metadata/phase29_job_ids.txt
echo "=== LOTUS jobs ==="
squeue -j "${prepare_job},${train_job},${aggregate_job}" -o "%.18i %.28j %.2t %.10M %.10l %R" || true
echo "=== preparation ==="
if [[ -f metadata/phase29_cairngorms_v2_surface_preparation.json ]]; then
  python - <<'PY'
import json
p=json.load(open('metadata/phase29_cairngorms_v2_surface_preparation.json'))
print(f"complete: {p['rows']:,} rows; minimum valid patch cells={p['valid_patch_cells_minimum']}")
PY
else
  log=$(ls -1t logs/phase29_prepare_*.out 2>/dev/null | head -n 1 || true)
  [[ -n "${log}" ]] && tail -n 8 "${log}" || echo "waiting"
fi
echo "=== model runs ==="
raw=$(find data/interim/phase29_cairngorms_v2_surface_raw_results -name '*.npz' 2>/dev/null | wc -l)
adjusted=$(find data/interim/phase29_cairngorms_v2_surface_adjusted_results -name '*.npz' 2>/dev/null | wc -l)
echo "raw ${raw}/30; height-adjusted ${adjusted}/30; total $((raw + adjusted))/60"
errors=$(find logs -name 'phase29_*.err' -type f -size +0c 2>/dev/null | wc -l)
echo "non-empty error logs: ${errors}"
echo "=== final report ==="
if [[ -f outputs/reports/phase29_cairngorms_v2_surface.md ]]; then
  sed -n '1,80p' outputs/reports/phase29_cairngorms_v2_surface.md
else
  echo "not complete"
fi
REMOTE_SCRIPT


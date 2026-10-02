#!/bin/zsh
set -euo pipefail

ROOT="/Users/ameerfiras/ICCS/tessera-gedi-fhd"
PYTHON="/Users/ameerfiras/miniforge3/envs/tessera-crop-label-audit/bin/python"
STATUS="$ROOT/outputs/reports/phase13_background.status"

cd "$ROOT"
mkdir -p outputs/reports
trap 'printf "FAILED %s\n" "$(date -u +%FT%TZ)" > "$STATUS"' ERR
printf "RUNNING acquisition %s\n" "$(date -u +%FT%TZ)" > "$STATUS"

if [[ ! -f metadata/phase13_2025_target_free_gedi_freeze.json ]]; then
  "$PYTHON" scripts/acquire_acquisition_replication_2025_extension.py
fi
"$PYTHON" - <<'PY'
import json
from pathlib import Path

path = Path("metadata/phase13_2025_target_free_gedi_freeze.json")
if not path.exists():
    raise SystemExit("2025 acquisition ended without a freeze")
freeze = json.loads(path.read_text())
if not freeze["freeze_basis"]["gate"]["passed"]:
    raise SystemExit("Phase 13 combined unique-forest gate failed")
print("2025 extension passed:", freeze["freeze_basis"]["viable_2025_sites"], flush=True)
PY

printf "RUNNING predictors %s\n" "$(date -u +%FT%TZ)" > "$STATUS"
if [[ ! -f metadata/phase13_combined_target_free_freeze.json ]]; then
  "$PYTHON" scripts/freeze_acquisition_replication_combined_cohort.py
fi
if [[ ! -f metadata/phase13_tessera_alignment_freeze.json ]]; then
  "$PYTHON" scripts/build_acquisition_replication_tessera_alignment.py
fi
if [[ ! -f metadata/phase13_conventional_freeze.json ]]; then
  "$PYTHON" scripts/build_acquisition_replication_conventional_predictors.py
fi

printf "RUNNING evaluation %s\n" "$(date -u +%FT%TZ)" > "$STATUS"
if [[ ! -f metadata/phase13_prediction_freeze.json ]]; then
  "$PYTHON" scripts/freeze_acquisition_replication_track_predictions.py
fi
if [[ ! -f metadata/phase13_track_replication_result_freeze.json ]]; then
  "$PYTHON" scripts/evaluate_acquisition_replication_track_replication.py
fi

if "$PYTHON" - <<'PY'
import json
from pathlib import Path

freeze = json.loads(Path("metadata/phase13_track_replication_result_freeze.json").read_text())
raise SystemExit(0 if freeze["freeze_basis"]["primary_gate"]["passed"] else 1)
PY
then
  if [[ ! -f metadata/phase13_regional_map_freeze.json ]]; then
    printf "RUNNING map %s\n" "$(date -u +%FT%TZ)" > "$STATUS"
    "$PYTHON" scripts/build_acquisition_replication_regional_map.py
  fi
fi

"$PYTHON" -m pytest tests/test_acquisition_replication_target_free_acquisition.py tests/test_acquisition_replication_track_replication.py -q
printf "COMPLETE %s\n" "$(date -u +%FT%TZ)" > "$STATUS"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="uk.ac.cam.tessera-gedi-fhd.phase19-cairngorms"
DOMAIN="gui/$(id -u)"
LOG_PATH="${ROOT}/outputs/reports/phase19_cairngorms_background.log"
ERROR_PATH="${ROOT}/outputs/reports/phase19_cairngorms_background.error.log"
REPORT_PATH="${ROOT}/outputs/reports/phase19_cairngorms_diagnostics_results.md"

printf '=== process ===\n'
if launchctl print "${DOMAIN}/${LABEL}" >/tmp/phase19_launchctl.txt 2>/dev/null; then
  awk '
    /^[[:space:]]*state = / ||
    /^[[:space:]]*pid = / ||
    /^[[:space:]]*last exit code = / {gsub(/^[[:space:]]+/, ""); print}
  ' /tmp/phase19_launchctl.txt
else
  printf 'LaunchAgent is not loaded.\n'
fi
rm -f /tmp/phase19_launchctl.txt

printf '\n=== checkpoints ===\n'
for stage in profiles patch9 context conventional arrays; do
  case "${stage}" in
    profiles) path="${ROOT}/metadata/phase19_cairngorms_vertical_profile_manifest.json" ;;
    patch9) path="${ROOT}/metadata/phase19_cairngorms_patch9_manifest.json" ;;
    context) path="${ROOT}/metadata/phase19_cairngorms_context_manifest.json" ;;
    conventional) path="${ROOT}/metadata/phase19_cairngorms_conventional_manifest.json" ;;
    arrays) path="${ROOT}/data/interim/phase19_cairngorms_arrays/manifest.json" ;;
  esac
  if [[ -f "${path}" ]]; then
    printf '%-24s complete\n' "${stage}:"
  else
    printf '%-24s pending\n' "${stage}:"
  fi
done
printf 'TESSERA 90 m tiles: %s complete\n' "$(
  find "${ROOT}/data/interim/phase19_cairngorms_patch9_tiles" \
    -type f -name 'grid_*.json' 2>/dev/null | wc -l | tr -d ' '
)"
printf 'Conventional chunks: %s / 9\n' "$(
  find "${ROOT}/data/interim/phase19_cairngorms_conventional_chunks" \
    -type f -name 'rows_*.parquet' 2>/dev/null | wc -l | tr -d ' '
)"
printf 'Neural seed results: %s / 150\n' "$(
  find "${ROOT}/data/interim/phase19_cairngorms_neural_results" \
    -type f -name 'fold_*.npz' 2>/dev/null | wc -l | tr -d ' '
)"
printf 'Completed evaluation folds: %s / 5\n' "$(
  find "${ROOT}/data/interim/phase19_cairngorms_evaluation_folds" \
    -type f -name 'fold_*_manifest.json' 2>/dev/null | wc -l | tr -d ' '
)"
if [[ -f "${REPORT_PATH}" ]]; then
  printf 'Final report: complete\n'
else
  printf 'Final report: pending\n'
fi

printf '\n=== storage ===\n'
df -h "${ROOT}" | tail -n 1

printf '\n=== latest log ===\n'
if [[ -f "${LOG_PATH}" ]]; then
  tail -n 28 "${LOG_PATH}"
else
  printf 'No standard-output log yet.\n'
fi
if [[ -s "${ERROR_PATH}" ]]; then
  printf '\n=== latest errors ===\n'
  tail -n 16 "${ERROR_PATH}"
fi

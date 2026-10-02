#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="uk.ac.cam.tessera-gedi-fhd.phase18-cairngorms"
DOMAIN="gui/$(id -u)"
LOG_PATH="${ROOT}/outputs/reports/phase18_cairngorms_background.log"
ERROR_PATH="${ROOT}/outputs/reports/phase18_cairngorms_background.error.log"
REPORT_PATH="${ROOT}/outputs/reports/phase18_cairngorms_spatial_heads_results.md"

printf '=== process ===\n'
if launchctl print "${DOMAIN}/${LABEL}" >/tmp/phase18_launchctl.txt 2>/dev/null; then
  awk '
    /^[[:space:]]*state = / ||
    /^[[:space:]]*pid = / ||
    /^[[:space:]]*last exit code = / {gsub(/^[[:space:]]+/, ""); print}
  ' /tmp/phase18_launchctl.txt
else
  printf 'LaunchAgent is not loaded.\n'
fi
rm -f /tmp/phase18_launchctl.txt

printf '\n=== checkpoints ===\n'
printf 'TESSERA patch tiles: %s / 15\n' "$(
  find "${ROOT}/data/interim/phase18_cairngorms_patch_tiles" \
    -type f -name 'grid_*.json' 2>/dev/null | wc -l | tr -d ' '
)"
if [[ -f "${ROOT}/metadata/phase18_cairngorms_patch_manifest.json" ]]; then
  printf 'Ordered patch array: complete\n'
else
  printf 'Ordered patch array: pending\n'
fi
if [[ -f "${ROOT}/metadata/phase18_cairngorms_conventional_manifest.json" ]]; then
  printf 'Sentinel/terrain predictors: complete\n'
else
  printf 'Sentinel/terrain predictors: pending\n'
fi
printf 'Neural seed results: %s / 20\n' "$(
  find "${ROOT}/data/interim/phase18_cairngorms_neural_results" \
    -type f -name 'fold_*.npz' 2>/dev/null | wc -l | tr -d ' '
)"
printf 'Completed evaluation folds: %s / 5\n' "$(
  find "${ROOT}/data/interim/phase18_cairngorms_evaluation_folds" \
    -type f -name 'fold_*_manifest.json' 2>/dev/null | wc -l | tr -d ' '
)"
if [[ -f "${REPORT_PATH}" ]]; then
  printf 'Final report: complete\n'
else
  printf 'Final report: pending\n'
fi

printf '\n=== latest log ===\n'
if [[ -f "${LOG_PATH}" ]]; then
  tail -n 24 "${LOG_PATH}"
else
  printf 'No standard-output log yet.\n'
fi
if [[ -s "${ERROR_PATH}" ]]; then
  ERROR_MODIFIED="$(stat -f '%m' "${ERROR_PATH}")"
  LOG_MODIFIED="$(
    if [[ -f "${LOG_PATH}" ]]; then
      stat -f '%m' "${LOG_PATH}"
    else
      printf '0'
    fi
  )"
  if (( ERROR_MODIFIED < LOG_MODIFIED )); then
    printf '\n=== historical error from before latest progress ===\n'
    printf 'No newer error has been recorded.\n'
    tail -n 6 "${ERROR_PATH}"
  else
    printf '\n=== latest errors ===\n'
    tail -n 16 "${ERROR_PATH}"
  fi
fi

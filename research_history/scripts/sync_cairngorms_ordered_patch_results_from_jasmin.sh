#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-ordered-patch"

mkdir -p \
  "${ROOT}/data/interim/cairngorms_ordered_patch_results" \
  "${ROOT}/data/processed" \
  "${ROOT}/outputs/tables" \
  "${ROOT}/metadata" \
  "${ROOT}/outputs/reports/cairngorms_ordered_patch_logs"

rsync -a \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/interim/cairngorms_ordered_patch_results/" \
  "${ROOT}/data/interim/cairngorms_ordered_patch_results/"
rsync -a \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/processed/cairngorms_ordered_patch_oof_predictions.parquet" \
  "${ROOT}/data/processed/"
rsync -a \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/tables/cairngorms_ordered_patch_"'*.csv' \
  "${ROOT}/outputs/tables/"
rsync -a \
  "${REMOTE_HOST}:${REMOTE_ROOT}/metadata/cairngorms_ordered_patch_result_freeze.json" \
  "${ROOT}/metadata/"
rsync -a \
  "${REMOTE_HOST}:${REMOTE_ROOT}/logs/" \
  "${ROOT}/outputs/reports/cairngorms_ordered_patch_logs/"

printf 'Synchronized completed ordered-patch results into %s\n' "${ROOT}"

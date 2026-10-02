#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

echo "=== LOTUS jobs ==="
ssh "${HOST}" "cd '${REMOTE}'; source metadata/phase33_job_ids.txt; squeue -j \"\${train_job},\${aggregate_job}\" -o '%.18i %.26j %.2t %.10M %.10l %R'"

echo "=== model runs ==="
ssh "${HOST}" "find '${REMOTE}/data/interim/phase33_cairngorms_flip_augmentation_results' -maxdepth 1 -name '*.npz' | wc -l"

echo "=== latest training output ==="
ssh "${HOST}" "find '${REMOTE}/logs' -maxdepth 1 -name 'phase33_train_*.out' -type f -print0 | xargs -0 ls -t 2>/dev/null | head -n 1 | xargs -I{} tail -n 12 '{}'" || true

echo "=== non-empty errors ==="
ssh "${HOST}" "find '${REMOTE}/logs' -maxdepth 1 -name 'phase33_*.err' -type f -size +0c -print | wc -l"

if ssh "${HOST}" "test -f '${REMOTE}/metadata/phase33_cairngorms_flip_augmentation_result_freeze.json'"; then
  mkdir -p "${ROOT}/data/processed" "${ROOT}/outputs/tables" "${ROOT}/outputs/figures" "${ROOT}/outputs/reports" "${ROOT}/metadata"
  rsync -a "${HOST}:${REMOTE}/data/processed/phase33_cairngorms_flip_augmentation_predictions.parquet" "${ROOT}/data/processed/"
  rsync -a "${HOST}:${REMOTE}/outputs/tables/phase33_cairngorms_flip_augmentation_"*.csv "${ROOT}/outputs/tables/"
  rsync -a "${HOST}:${REMOTE}/outputs/figures/phase33_cairngorms_flip_augmentation.pdf" "${ROOT}/outputs/figures/"
  rsync -a "${HOST}:${REMOTE}/outputs/reports/phase33_cairngorms_flip_augmentation.md" "${ROOT}/outputs/reports/"
  rsync -a "${HOST}:${REMOTE}/metadata/phase33_cairngorms_flip_augmentation_result_freeze.json" "${ROOT}/metadata/"
  echo "=== final report ==="
  cat "${ROOT}/outputs/reports/phase33_cairngorms_flip_augmentation.md"
else
  echo "=== final report ==="
  echo "not complete"
fi


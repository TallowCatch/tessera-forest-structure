#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-ordered-patch"

ssh "${REMOTE_HOST}" \
  "umask 002; mkdir -p '${REMOTE_ROOT}'/{configs,docs,scripts/jasmin,data/interim/phase18_cairngorms_neural,data/processed,outputs/tables,metadata,logs}"

rsync -a "${ROOT}/configs/cairngorms_ordered_patch.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/configs/"
rsync -a "${ROOT}/docs/cairngorms_ordered_patch_design.md" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/docs/"
rsync -a \
  "${ROOT}/scripts/cairngorms_ordered_patch_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_ordered_patch.py" \
  "${ROOT}/scripts/freeze_cairngorms_ordered_patch_inputs.py" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/jasmin/"

for name in \
  patch_quantized.npy patch_scales.npy patch_valid.npy row_id.npy manifest.json \
  fold_0.npz fold_1.npz fold_2.npz fold_3.npz fold_4.npz; do
  rsync -a \
    "${ROOT}/data/interim/phase18_cairngorms_neural/${name}" \
    "${REMOTE_HOST}:${REMOTE_ROOT}/data/interim/phase18_cairngorms_neural/"
done

rsync -a \
  "${ROOT}/data/processed/phase18_cairngorms_oof_predictions.parquet" \
  "${ROOT}/data/processed/phase17_cairngorms_50m_targets.parquet" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/processed/"
rsync -a "${ROOT}/outputs/tables/phase18_cairngorms_pooled_metrics.csv" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/tables/"
rsync -a \
  "${ROOT}/metadata/phase18_cairngorms_patch_manifest.json" \
  "${ROOT}/metadata/phase19_cairngorms_result_freeze.json" \
  "${ROOT}/metadata/cairngorms_ordered_patch_input_freeze.json" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/metadata/"

ssh "${REMOTE_HOST}" "chmod -R g+rwX '${REMOTE_ROOT}'"
printf 'Deployed ordered-patch experiment to %s:%s\n' \
  "${REMOTE_HOST}" "${REMOTE_ROOT}"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,data/interim/phase33_cairngorms_flip_augmentation_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"
rsync -a "${ROOT}/configs/reflection_augmentation_cairngorms_flip_augmentation.yaml" "${HOST}:${REMOTE}/configs/"
rsync -a "${ROOT}/docs/phase33_cairngorms_flip_augmentation_design.md" "${HOST}:${REMOTE}/docs/"
rsync -a \
  "${ROOT}/scripts/reflection_augmentation_flip_mlp_common.py" \
  "${ROOT}/scripts/reflection_augmentation_flip_augmented_worker.py" \
  "${ROOT}/scripts/aggregate_reflection_augmentation_cairngorms_flip_augmentation.py" \
  "${ROOT}/scripts/cairngorms_spatial_unet_spatial_unet_worker.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/reflection_augmentation_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
python -m py_compile \
  scripts/reflection_augmentation_flip_mlp_common.py \
  scripts/reflection_augmentation_flip_augmented_worker.py \
  scripts/aggregate_reflection_augmentation_cairngorms_flip_augmentation.py
test -f metadata/phase29_cairngorms_v2_surface_result_freeze.json
test -f data/interim/phase29_cairngorms_v2_surface_raw/patch_quantized.npy
test -f data/interim/phase29_cairngorms_v2_surface_adjusted/tessera.npy
if [[ -f metadata/phase33_job_ids.txt ]]; then
  source metadata/phase33_job_ids.txt
  active=$(squeue -h -j "${train_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 33 already has ${active} active job records"
    cat metadata/phase33_job_ids.txt
    exit 0
  fi
  if [[ -f metadata/phase33_cairngorms_flip_augmentation_result_freeze.json ]]; then
    echo "Phase 33 is already complete"
    cat metadata/phase33_job_ids.txt
    exit 0
  fi
fi
train_job=$(sbatch --parsable scripts/jasmin/reflection_augmentation_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/reflection_augmentation_aggregate.sbatch)
printf 'train_job=%s\naggregate_job=%s\n' \
  "${train_job}" "${aggregate_job}" > metadata/phase33_job_ids.txt
chmod -R g+rwX configs docs scripts data metadata logs outputs
cat metadata/phase33_job_ids.txt
REMOTE_SCRIPT

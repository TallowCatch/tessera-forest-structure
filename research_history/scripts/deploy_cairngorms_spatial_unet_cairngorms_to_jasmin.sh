#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,data/interim/phase22_cairngorms_spatial_unet,data/interim/phase22_cairngorms_spatial_unet_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"

rsync -a "${ROOT}/configs/cairngorms_spatial_unet.yaml" "${HOST}:${REMOTE}/configs/"
rsync -a "${ROOT}/docs/phase22_cairngorms_spatial_unet_design.md" "${HOST}:${REMOTE}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_components.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${ROOT}/scripts/prepare_cairngorms_spatial_unet.py" \
  "${ROOT}/scripts/cairngorms_spatial_unet_spatial_unet_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_spatial_unet.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/cairngorms_spatial_unet_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"
rsync -a "${ROOT}/data/processed/phase21_cairngorms_surface_targets.parquet" "${HOST}:${REMOTE}/data/processed/"
rsync -a \
  "${ROOT}/data/interim/phase18_cairngorms_neural/patch_quantized.npy" \
  "${ROOT}/data/interim/phase18_cairngorms_neural/patch_scales.npy" \
  "${ROOT}/data/interim/phase18_cairngorms_neural/patch_valid.npy" \
  "${ROOT}/data/interim/phase18_cairngorms_neural/row_id.npy" \
  "${HOST}:${REMOTE}/data/interim/phase18_cairngorms_neural/"
rsync -a \
  "${ROOT}/data/interim/phase19_cairngorms_arrays/conventional_5.npy" \
  "${ROOT}/data/interim/phase19_cairngorms_arrays/row_id.npy" \
  "${HOST}:${REMOTE}/data/interim/phase19_cairngorms_arrays/"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
python -m py_compile \
  scripts/prepare_cairngorms_spatial_unet.py \
  scripts/cairngorms_spatial_unet_spatial_unet_worker.py \
  scripts/aggregate_cairngorms_spatial_unet.py

if [[ -f metadata/phase22_job_ids.txt ]]; then
  source metadata/phase22_job_ids.txt
  active=$(squeue -h -j "${prepare_job:-0},${scalar_job:-0},${unet_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 22 already has ${active} active job records"
    cat metadata/phase22_job_ids.txt
    exit 0
  fi
fi

prepare_job=$(sbatch --parsable scripts/jasmin/cairngorms_spatial_unet_prepare.sbatch)
scalar_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_spatial_unet_scalar_array.sbatch)
unet_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_spatial_unet_unet_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${scalar_job}:${unet_job}" scripts/jasmin/cairngorms_spatial_unet_aggregate.sbatch)
printf 'prepare_job=%s\nscalar_job=%s\nunet_job=%s\naggregate_job=%s\n' \
  "${prepare_job}" "${scalar_job}" "${unet_job}" "${aggregate_job}" > metadata/phase22_job_ids.txt
chmod -R g+rwX configs docs scripts data metadata logs outputs
cat metadata/phase22_job_ids.txt
REMOTE_SCRIPT

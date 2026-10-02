#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${REMOTE_HOST}" \
  "umask 002; mkdir -p '${REMOTE_ROOT}'/{configs,docs,scripts/jasmin,data/raw/scotland_lidar/chm,data/interim/phase18_cairngorms_neural,data/interim/phase19_cairngorms_arrays,data/interim/phase21_cairngorms_surface,data/interim/phase21_cairngorms_surface_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"

rsync -a \
  "${ROOT}/configs/cairngorms_surface.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/configs/"
rsync -a \
  "${ROOT}/docs/phase21_cairngorms_surface_design.md" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_components.py" \
  "${ROOT}/scripts/cairngorms_components_components_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_components.py" \
  "${ROOT}/scripts/prepare_cairngorms_surface.py" \
  "${ROOT}/scripts/cairngorms_surface_surface_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_surface.py" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -a \
  "${ROOT}/scripts/jasmin/cairngorms_surface_prepare.sbatch" \
  "${ROOT}/scripts/jasmin/cairngorms_surface_train_array.sbatch" \
  "${ROOT}/scripts/jasmin/cairngorms_surface_aggregate.sbatch" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/jasmin/"
rsync -a \
  "${ROOT}/data/processed/phase20_cairngorms_component_targets.parquet" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/processed/"
rsync -a \
  "${ROOT}/data/interim/phase18_cairngorms_neural/summary.npy" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/interim/phase18_cairngorms_neural/"
rsync -a \
  "${ROOT}/data/interim/phase19_cairngorms_arrays/conventional_5.npy" \
  "${ROOT}/data/interim/phase19_cairngorms_arrays/row_id.npy" \
  "${ROOT}/data/interim/phase19_cairngorms_arrays/common_valid.npy" \
  "${ROOT}/data/interim/phase19_cairngorms_arrays/fold_"*.npz \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/interim/phase19_cairngorms_arrays/"
rsync -a \
  "${ROOT}/data/raw/scotland_lidar/chm/CHM_canopy_metrics_50m_new.tif" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/raw/scotland_lidar/chm/"

ssh "${REMOTE_HOST}" "ROOT='${REMOTE_ROOT}' bash -s" <<'REMOTE'
set -euo pipefail
cd "${ROOT}"

if [[ -f metadata/phase21_job_ids.txt ]]; then
  source metadata/phase21_job_ids.txt
  active=$(squeue -h -j "${prepare_job:-0},${train_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 21 already has ${active} active LOTUS job records"
    cat metadata/phase21_job_ids.txt
    exit 0
  fi
fi

prepare_job=$(sbatch --parsable scripts/jasmin/cairngorms_surface_prepare.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_surface_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/cairngorms_surface_aggregate.sbatch)
printf 'prepare_job=%s\ntrain_job=%s\naggregate_job=%s\n' \
  "${prepare_job}" "${train_job}" "${aggregate_job}" \
  > metadata/phase21_job_ids.txt
chmod -R g+rwX configs docs scripts data metadata logs outputs
cat metadata/phase21_job_ids.txt
REMOTE

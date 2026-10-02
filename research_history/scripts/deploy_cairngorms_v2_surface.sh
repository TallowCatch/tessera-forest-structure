#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,data/interim/phase29_cairngorms_v2_surface_raw,data/interim/phase29_cairngorms_v2_surface_adjusted,data/interim/phase29_cairngorms_v2_surface_raw_results,data/interim/phase29_cairngorms_v2_surface_adjusted_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"
rsync -a "${ROOT}/configs/cairngorms_v2_surface.yaml" "${HOST}:${REMOTE}/configs/"
rsync -a "${ROOT}/docs/phase29_cairngorms_v2_surface_design.md" "${HOST}:${REMOTE}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_v2_surface.py" \
  "${ROOT}/scripts/cairngorms_v2_surface_v2_surface_raw_worker.py" \
  "${ROOT}/scripts/cairngorms_v2_surface_v2_surface_adjusted_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_v2_surface.py" \
  "${ROOT}/scripts/cairngorms_spatial_unet_spatial_unet_worker.py" \
  "${ROOT}/scripts/cairngorms_components_components_worker.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/cairngorms_v2_surface_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
python -m py_compile \
  scripts/prepare_cairngorms_v2_surface.py \
  scripts/cairngorms_v2_surface_v2_surface_raw_worker.py \
  scripts/cairngorms_v2_surface_v2_surface_adjusted_worker.py \
  scripts/aggregate_cairngorms_v2_surface.py
if [[ -f metadata/phase29_job_ids.txt ]]; then
  source metadata/phase29_job_ids.txt
  active=$(squeue -h -j "${prepare_job:-0},${train_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 29 already has ${active} active job records"
    cat metadata/phase29_job_ids.txt
    exit 0
  fi
  if [[ -f metadata/phase29_cairngorms_v2_surface_result_freeze.json ]]; then
    echo "Phase 29 is already complete"
    cat metadata/phase29_job_ids.txt
    exit 0
  fi
fi
prepare_job=$(sbatch --parsable scripts/jasmin/cairngorms_v2_surface_prepare.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_v2_surface_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/cairngorms_v2_surface_aggregate.sbatch)
printf 'prepare_job=%s\ntrain_job=%s\naggregate_job=%s\n' \
  "${prepare_job}" "${train_job}" "${aggregate_job}" > metadata/phase29_job_ids.txt
chmod -R g+rwX configs docs scripts data metadata logs outputs
cat metadata/phase29_job_ids.txt
REMOTE_SCRIPT

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${REMOTE_HOST}" \
  "umask 002; mkdir -p '${REMOTE_ROOT}'/{configs,docs,scripts/jasmin,data/raw/scotland_lidar/chm,data/interim/phase18_cairngorms_neural,data/interim/phase19_cairngorms_arrays,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"

rsync -a \
  "${ROOT}/configs/cairngorms_components.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/configs/"
rsync -a \
  "${ROOT}/docs/phase20_cairngorms_components_design.md" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_components.py" \
  "${ROOT}/scripts/cairngorms_components_components_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_components.py" \
  "${ROOT}/scripts/cairngorms_components_download_and_submit_remote.sh" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -a \
  "${ROOT}/scripts/jasmin/cairngorms_components_prepare.sbatch" \
  "${ROOT}/scripts/jasmin/cairngorms_components_train_array.sbatch" \
  "${ROOT}/scripts/jasmin/cairngorms_components_aggregate.sbatch" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/jasmin/"
rsync -a \
  "${ROOT}/data/processed/phase17_cairngorms_50m_targets.parquet" \
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

ssh "${REMOTE_HOST}" "set -e; cd '${REMOTE_ROOT}'; chmod -R g+rwX configs docs scripts data metadata logs outputs; if [[ -f metadata/phase20_orchestrator.pid ]] && kill -0 \$(cat metadata/phase20_orchestrator.pid) 2>/dev/null; then echo 'Phase 20 orchestrator already running'; else nohup bash scripts/cairngorms_components_download_and_submit_remote.sh > logs/phase20_orchestrator.log 2>&1 < /dev/null & echo \$! > metadata/phase20_orchestrator.pid; echo \"started Phase 20 orchestrator PID \$(cat metadata/phase20_orchestrator.pid)\"; fi"

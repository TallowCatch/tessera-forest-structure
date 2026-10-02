#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${REMOTE_HOST}" \
  "umask 002; mkdir -p '${REMOTE_ROOT}'/{configs,docs,scripts/jasmin,data/raw/scotland_lidar,data/interim,data/processed,outputs/tables,outputs/maps,metadata,logs}"

rsync -a --progress \
  "${ROOT}/data/raw/scotland_lidar/LiDAR_metrics_2.tif" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/data/raw/scotland_lidar/"
rsync -a \
  "${ROOT}/configs/cairngorms_dense_10m.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/configs/"
rsync -a \
  "${ROOT}/docs/cairngorms_dense_10m_design.md" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_dense_10m.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_dense_10m.py" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
rsync -a \
  "${ROOT}/scripts/jasmin/cairngorms_dense_"*.sbatch \
  "${ROOT}/scripts/jasmin/submit_cairngorms_dense_10m.sh" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/jasmin/"
rsync -a \
  "${ROOT}/metadata/cairngorms_dense_10m_tessera_tile_manifest.json" \
  "${ROOT}/metadata/cairngorms_dense_10m_protocol_freeze.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/metadata/"

ssh "${REMOTE_HOST}" \
  "chmod -R g+rwX '${REMOTE_ROOT}'; cd '${REMOTE_ROOT}'; bash scripts/jasmin/submit_cairngorms_dense_10m.sh"

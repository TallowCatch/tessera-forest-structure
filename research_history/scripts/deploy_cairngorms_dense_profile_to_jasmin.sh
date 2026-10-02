#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${REMOTE_HOST}" \
  "umask 002; test -f '${REMOTE_ROOT}/data/interim/cairngorms_dense_10m/manifest.json'; test -f '${REMOTE_ROOT}/data/raw/scotland_lidar/LiDAR_metrics_2.tif'; mkdir -p '${REMOTE_ROOT}'/{configs,docs,scripts/jasmin,metadata,logs,data/interim,data/processed,outputs/tables,outputs/maps}"

scp \
  "${ROOT}/configs/cairngorms_dense_profile.yaml" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/configs/"
scp \
  "${ROOT}/docs/cairngorms_dense_profile_design.md" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/docs/"
scp \
  "${ROOT}/scripts/prepare_cairngorms_dense_10m.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${ROOT}/scripts/prepare_cairngorms_dense_profile.py" \
  "${ROOT}/scripts/cairngorms_dense_profile_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_dense_profile.py" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/"
scp \
  "${ROOT}/scripts/jasmin/cairngorms_profile_"*.sbatch \
  "${ROOT}/scripts/jasmin/submit_cairngorms_dense_profile.sh" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/scripts/jasmin/"

ssh "${REMOTE_HOST}" \
  "chmod -R g+rwX '${REMOTE_ROOT}'; cd '${REMOTE_ROOT}'; bash scripts/jasmin/submit_cairngorms_dense_profile.sh"

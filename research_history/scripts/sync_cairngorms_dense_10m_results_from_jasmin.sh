#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

mkdir -p \
  "${ROOT}/outputs/tables" \
  "${ROOT}/outputs/maps/cairngorms_dense_10m" \
  "${ROOT}/outputs/reports/cairngorms_dense_10m_logs" \
  "${ROOT}/metadata"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/tables/cairngorms_dense_10m_"* \
  "${ROOT}/outputs/tables/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/maps/cairngorms_dense_10m/" \
  "${ROOT}/outputs/maps/cairngorms_dense_10m/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/metadata/cairngorms_dense_10m_"* \
  "${ROOT}/metadata/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/logs/" \
  "${ROOT}/outputs/reports/cairngorms_dense_10m_logs/"

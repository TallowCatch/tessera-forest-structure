#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

mkdir -p "${ROOT}/outputs/tables" "${ROOT}/outputs/maps/cairngorms_dense_profile" "${ROOT}/metadata"
scp \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/tables/cairngorms_dense_profile_*.csv" \
  "${ROOT}/outputs/tables/"
scp \
  "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/maps/cairngorms_dense_profile/*.tif" \
  "${ROOT}/outputs/maps/cairngorms_dense_profile/"
scp \
  "${REMOTE_HOST}:${REMOTE_ROOT}/metadata/cairngorms_dense_profile_*.json" \
  "${ROOT}/metadata/"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REMOTE_HOST="jasmin-sci-vm01"
REMOTE_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

mkdir -p "${ROOT}/outputs/tables" "${ROOT}/outputs/figures" "${ROOT}/outputs/reports" "${ROOT}/metadata" "${ROOT}/data/processed"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/tables/phase20_"* "${ROOT}/outputs/tables/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/figures/phase20_"* "${ROOT}/outputs/figures/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/outputs/reports/phase20_"* "${ROOT}/outputs/reports/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/metadata/phase20_"* "${ROOT}/metadata/"
rsync -a "${REMOTE_HOST}:${REMOTE_ROOT}/data/processed/phase20_cairngorms_component_targets.parquet" "${REMOTE_HOST}:${REMOTE_ROOT}/data/processed/phase20_cairngorms_component_predictions.parquet" "${ROOT}/data/processed/"
echo "Phase 20 results synchronized to ${ROOT}"

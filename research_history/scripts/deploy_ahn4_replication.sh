#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,data/raw/ahn4,data/processed,data/interim,docs,logs,metadata,outputs/tables,outputs/reports,scripts/jasmin}"

rsync -a "${ROOT}/configs/ahn4_replication_ahn4_deciduous.yaml" "${HOST}:${REMOTE}/configs/"
rsync -a "${ROOT}/docs/phase26_ahn4_deciduous_replication_plan.md" "${HOST}:${REMOTE}/docs/"
rsync -a "${ROOT}/metadata/phase26_ahn4_selection_freeze.json" "${HOST}:${REMOTE}/metadata/"
rsync -a "${ROOT}/data/processed/phase26_ahn4_selected_site.gpkg" "${HOST}:${REMOTE}/data/processed/"
rsync -a "${ROOT}/data/raw/ahn4/ahn4_10m_flighttime.tif" "${HOST}:${REMOTE}/data/raw/ahn4/"
rsync -a "${ROOT}/data/raw/ahn4/ahn4_10m_mask_building_road_water.tif" "${HOST}:${REMOTE}/data/raw/ahn4/"
rsync -a "${ROOT}/data/raw/ahn4/ahn4_10m_mask_powerline.tif" "${HOST}:${REMOTE}/data/raw/ahn4/"
rsync -a "${ROOT}/data/raw/ahn4/ahn4_10m_NA_mask.tif" "${HOST}:${REMOTE}/data/raw/ahn4/"
rsync -a \
  "${ROOT}/scripts/prepare_ahn4_replication_ahn4_cohort.py" \
  "${ROOT}/scripts/acquire_ahn4_replication_ahn4_tessera.py" \
  "${ROOT}/scripts/acquire_ahn4_replication_ahn4_conventional.py" \
  "${ROOT}/scripts/evaluate_ahn4_replication_ahn4.py" \
  "${ROOT}/scripts/build_conventional_predictors.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/ahn4_replication_prepare.sbatch" \
  "${ROOT}/scripts/jasmin/ahn4_replication_tessera.sbatch" \
  "${ROOT}/scripts/jasmin/ahn4_replication_conventional.sbatch" \
  "${ROOT}/scripts/jasmin/ahn4_replication_evaluate.sbatch" \
  "${HOST}:${REMOTE}/scripts/jasmin/"

ssh "${HOST}" "module load jaspy/3.12/v20250704; source '${ENVIRONMENT}/bin/activate'; python -m pip install --quiet planetary-computer==1.0.0"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ -f metadata/phase26_ahn4_job_ids.txt ]]; then
  source metadata/phase26_ahn4_job_ids.txt
  if squeue -h -j "${prepare_job:-0},${tessera_job:-0},${conventional_job:-0},${evaluate_job:-0}" 2>/dev/null | grep -q .; then
    echo "Phase 26 replication is already active"
    cat metadata/phase26_ahn4_job_ids.txt
    exit 0
  fi
fi
prepare_job=$(sbatch --parsable scripts/jasmin/ahn4_replication_prepare.sbatch)
printf 'prepare_job=%s\n' "${prepare_job}" > metadata/phase26_ahn4_job_ids.txt
tessera_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/ahn4_replication_tessera.sbatch)
printf 'tessera_job=%s\n' "${tessera_job}" >> metadata/phase26_ahn4_job_ids.txt
conventional_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/ahn4_replication_conventional.sbatch)
printf 'conventional_job=%s\n' "${conventional_job}" >> metadata/phase26_ahn4_job_ids.txt
evaluate_job=$(sbatch --parsable --dependency="afterok:${tessera_job}:${conventional_job}" scripts/jasmin/ahn4_replication_evaluate.sbatch)
cat >> metadata/phase26_ahn4_job_ids.txt <<EOF
evaluate_job=${evaluate_job}
EOF
chmod -R g+rwX configs data docs logs metadata outputs scripts
cat metadata/phase26_ahn4_job_ids.txt
REMOTE_SCRIPT

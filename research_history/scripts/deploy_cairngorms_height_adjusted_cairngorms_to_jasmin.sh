#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,data/interim/phase23_cairngorms_height_adjusted,data/interim/phase23_cairngorms_height_adjusted_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"
rsync -a "${ROOT}/configs/cairngorms_height_adjusted.yaml" "${HOST}:${REMOTE}/configs/"
rsync -a "${ROOT}/docs/phase23_cairngorms_height_adjusted_design.md" "${HOST}:${REMOTE}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_components.py" \
  "${ROOT}/scripts/cairngorms_components_components_worker.py" \
  "${ROOT}/scripts/prepare_cairngorms_height_adjusted.py" \
  "${ROOT}/scripts/cairngorms_height_adjusted_height_adjusted_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_height_adjusted.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/cairngorms_height_adjusted_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"
ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
python -m py_compile scripts/prepare_cairngorms_height_adjusted.py scripts/cairngorms_height_adjusted_height_adjusted_worker.py scripts/aggregate_cairngorms_height_adjusted.py
if [[ -f metadata/phase23_job_ids.txt ]]; then
  source metadata/phase23_job_ids.txt
  active=$(squeue -h -j "${prepare_job:-0},${train_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 23 already has ${active} active job records"
    cat metadata/phase23_job_ids.txt
    exit 0
  fi
fi
prepare_job=$(sbatch --parsable scripts/jasmin/cairngorms_height_adjusted_prepare.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_height_adjusted_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/cairngorms_height_adjusted_aggregate.sbatch)
printf 'prepare_job=%s\ntrain_job=%s\naggregate_job=%s\n' "$prepare_job" "$train_job" "$aggregate_job" > metadata/phase23_job_ids.txt
chmod -R g+rwX configs docs scripts data metadata logs outputs
cat metadata/phase23_job_ids.txt
REMOTE_SCRIPT

#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm03"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,tests,data/interim/phase30_cairngorms_v2_height,data/interim/phase30_cairngorms_v2_height_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"
rsync -a \
  "${ROOT}/configs/cairngorms_v2_height_unet.yaml" \
  "${HOST}:${REMOTE}/configs/"
rsync -a \
  "${ROOT}/docs/phase30_cairngorms_v2_height_unet_design.md" \
  "${HOST}:${REMOTE}/docs/"
rsync -a \
  "${ROOT}/scripts/prepare_cairngorms_v2_height.py" \
  "${ROOT}/scripts/cairngorms_v2_height_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_v2_height.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/cairngorms_v2_height_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"
rsync -a \
  "${ROOT}/tests/test_cairngorms_v2_height.py" \
  "${HOST}:${REMOTE}/tests/"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
python -m py_compile \
  scripts/prepare_cairngorms_v2_height.py \
  scripts/cairngorms_v2_height_worker.py \
  scripts/aggregate_cairngorms_v2_height.py
python -m pytest -q tests/test_cairngorms_v2_height.py
for required in \
  data/processed/phase22_cairngorms_surface_targets.parquet \
  data/interim/phase29_cairngorms_v2_surface_raw/embedding_int8.npy \
  data/interim/phase29_cairngorms_v2_surface_raw/folds/balanced_fold_0.npz \
  data/interim/phase29_cairngorms_v2_surface_adjusted/tessera.npy \
  metadata/phase28_cairngorms_v2_dense_acquisition.json; do
  [[ -e "${required}" ]] || { echo "Missing frozen input: ${required}" >&2; exit 1; }
done
if [[ -f metadata/phase30_job_ids.txt ]]; then
  source metadata/phase30_job_ids.txt
  active=$(squeue -h -j "${prepare_job:-0},${train_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 30 already has ${active} active job records"
    cat metadata/phase30_job_ids.txt
    exit 0
  fi
  if [[ -f metadata/phase30_cairngorms_v2_height_result_freeze.json ]]; then
    echo "Phase 30 is already complete"
    cat metadata/phase30_job_ids.txt
    exit 0
  fi
fi
prepare_job=$(sbatch --parsable scripts/jasmin/cairngorms_v2_height_prepare.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_v2_height_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/cairngorms_v2_height_aggregate.sbatch)
printf 'prepare_job=%s\ntrain_job=%s\naggregate_job=%s\n' \
  "${prepare_job}" "${train_job}" "${aggregate_job}" > metadata/phase30_job_ids.txt
chmod -R g+rwX configs docs scripts tests data metadata logs outputs
cat metadata/phase30_job_ids.txt
REMOTE_SCRIPT

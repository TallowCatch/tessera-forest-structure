#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm03"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,tests,data/interim/phase31_cairngorms_v2_height_strict_results,data/processed,outputs/tables,outputs/figures,outputs/reports,metadata,logs}"
rsync -a \
  "${ROOT}/configs/cairngorms_v2_height_strict_unet.yaml" \
  "${HOST}:${REMOTE}/configs/"
rsync -a \
  "${ROOT}/docs/phase31_cairngorms_v2_height_strict_unet_design.md" \
  "${HOST}:${REMOTE}/docs/"
rsync -a \
  "${ROOT}/scripts/cairngorms_v2_height_strict_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_v2_height_strict.py" \
  "${ROOT}/scripts/cairngorms_v2_height_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_v2_height.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/cairngorms_v2_height_strict_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"
rsync -a \
  "${ROOT}/tests/test_cairngorms_v2_height_strict.py" \
  "${HOST}:${REMOTE}/tests/"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
python -m py_compile \
  scripts/cairngorms_v2_height_strict_worker.py \
  scripts/aggregate_cairngorms_v2_height_strict.py
python -m pytest -q tests/test_cairngorms_v2_height_strict.py
for required in \
  data/interim/phase30_cairngorms_v2_height/targets.npy \
  data/interim/phase30_cairngorms_v2_height/folds/fold_0.npz \
  data/interim/phase30_cairngorms_v2_height_results/tessera_v2_mlp_5x5_fold_0_seed_20260804.npz \
  data/processed/phase30_cairngorms_v2_height_predictions.parquet \
  metadata/phase30_cairngorms_v2_height_result_freeze.json; do
  [[ -e "${required}" ]] || { echo "Missing frozen input: ${required}" >&2; exit 1; }
done
if [[ -f metadata/phase31_job_ids.txt ]]; then
  source metadata/phase31_job_ids.txt
  active=$(squeue -h -j "${train_job:-0},${aggregate_job:-0}" 2>/dev/null | wc -l)
  if [[ "${active}" -gt 0 ]]; then
    echo "Phase 31 already has ${active} active job records"
    cat metadata/phase31_job_ids.txt
    exit 0
  fi
  if [[ -f metadata/phase31_cairngorms_v2_height_strict_result_freeze.json ]]; then
    echo "Phase 31 is already complete"
    cat metadata/phase31_job_ids.txt
    exit 0
  fi
fi
train_job=$(sbatch --parsable scripts/jasmin/cairngorms_v2_height_strict_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/cairngorms_v2_height_strict_aggregate.sbatch)
printf 'train_job=%s\naggregate_job=%s\n' \
  "${train_job}" "${aggregate_job}" > metadata/phase31_job_ids.txt
chmod -R g+rwX configs docs scripts tests data metadata logs outputs
cat metadata/phase31_job_ids.txt
REMOTE_SCRIPT

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,references/tessera_v2,scripts/jasmin,logs,metadata,data/interim/phase27_cairngorms_v2,data/interim/phase27_cairngorms_v2_results,data/processed,outputs/tables,outputs/reports}"
rsync -a "${ROOT}/configs/tessera_v2.yaml" "${HOST}:${REMOTE}/configs/"
rsync -a "${ROOT}/docs/phase27_tessera_v2_design.md" "${HOST}:${REMOTE}/docs/"
rsync -a "${ROOT}/references/tessera_v2/fetch_v2_patch_aneesh_20260811.py" "${HOST}:${REMOTE}/references/tessera_v2/"
rsync -a \
  "${ROOT}/scripts/tessera_v2_common.py" \
  "${ROOT}/scripts/acquire_tessera_v2.py" \
  "${ROOT}/scripts/evaluate_tessera_v2_savelsbos_v2.py" \
  "${ROOT}/scripts/tessera_v2_cairngorms_v2_worker.py" \
  "${ROOT}/scripts/aggregate_tessera_v2_cairngorms_v2.py" \
  "${ROOT}/scripts/evaluate_ahn4_replication_ahn4.py" \
  "${ROOT}/scripts/cairngorms_components_components_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_height_adjusted.py" \
  "${HOST}:${REMOTE}/scripts/"
rsync -a "${ROOT}/scripts/jasmin/tessera_v2_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"

ssh "${HOST}" "ROOT='${REMOTE}' ENVIRONMENT='${ENVIRONMENT}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
required=(
  data/processed/phase26_ahn4_deciduous_cohort.parquet
  data/processed/phase26_ahn4_tessera.parquet
  data/processed/phase26_ahn4_conventional.parquet
  data/processed/phase25_cairngorms_corrected_lidar_targets.parquet
  data/processed/phase25_cairngorms_corrected_lidar_predictions.parquet
  data/interim/phase25_cairngorms_corrected_lidar/conventional.npy
  data/interim/phase25_cairngorms_corrected_lidar/row_id.npy
  data/interim/phase25_cairngorms_corrected_lidar/spatial_block.npy
  data/interim/phase26_ahn4_folds/fold_0.npz
  data/interim/phase26_ahn4_folds/fold_4.npz
)
for path in "${required[@]}"; do
  [[ -f "${path}" ]] || { echo "Missing frozen prerequisite: ${path}" >&2; exit 1; }
done
expected="f5d26913ff5f9c10dbba2ee131506997413b597149fac339c8337004604cb7a9"
observed=$(python - <<'PY'
import hashlib
from pathlib import Path
print(hashlib.sha256(Path("references/tessera_v2/fetch_v2_patch_aneesh_20260811.py").read_bytes()).hexdigest())
PY
)
[[ "${observed}" == "${expected}" ]] || { echo "Supplied v2 loader hash changed" >&2; exit 1; }
python -m py_compile \
  scripts/tessera_v2_common.py scripts/acquire_tessera_v2.py \
  scripts/evaluate_tessera_v2_savelsbos_v2.py \
  scripts/tessera_v2_cairngorms_v2_worker.py \
  scripts/aggregate_tessera_v2_cairngorms_v2.py
if [[ -f metadata/phase27_job_ids.txt ]]; then
  source metadata/phase27_job_ids.txt
  if squeue -h -j "${cairngorms_acquire_job:-0},${savelsbos_acquire_job:-0},${cairngorms_train_job:-0},${savelsbos_evaluate_job:-0},${cairngorms_aggregate_job:-0}" 2>/dev/null | grep -q .; then
    echo "Phase 27 is already active"
    cat metadata/phase27_job_ids.txt
    exit 0
  fi
fi
cairngorms_acquire_job=$(sbatch --parsable scripts/jasmin/tessera_v2_cairngorms_acquire.sbatch)
savelsbos_acquire_job=$(sbatch --parsable scripts/jasmin/tessera_v2_savelsbos_acquire.sbatch)
cairngorms_train_job=$(sbatch --parsable --dependency="afterok:${cairngorms_acquire_job}" scripts/jasmin/tessera_v2_cairngorms_train_array.sbatch)
savelsbos_evaluate_job=$(sbatch --parsable --dependency="afterok:${savelsbos_acquire_job}" scripts/jasmin/tessera_v2_savelsbos_evaluate.sbatch)
cairngorms_aggregate_job=$(sbatch --parsable --dependency="afterok:${cairngorms_train_job}" scripts/jasmin/tessera_v2_cairngorms_aggregate.sbatch)
cat > metadata/phase27_job_ids.txt <<EOF
cairngorms_acquire_job=${cairngorms_acquire_job}
savelsbos_acquire_job=${savelsbos_acquire_job}
cairngorms_train_job=${cairngorms_train_job}
savelsbos_evaluate_job=${savelsbos_evaluate_job}
cairngorms_aggregate_job=${cairngorms_aggregate_job}
EOF
chmod -R g+rwX configs docs references scripts logs metadata data/interim/phase27_cairngorms_v2 data/interim/phase27_cairngorms_v2_results outputs
cat metadata/phase27_job_ids.txt
REMOTE_SCRIPT

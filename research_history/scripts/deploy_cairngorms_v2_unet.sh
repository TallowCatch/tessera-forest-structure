#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/ameerfiras/ICCS/tessera-gedi-fhd"
HOST="jasmin-sci-vm02"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{configs,docs,scripts/jasmin,logs,metadata,data/interim/phase28_cairngorms_v2_dense/tiles,data/interim/phase28_cairngorms_v2_stream,data/interim/phase28_cairngorms_v2_unet,data/interim/phase28_cairngorms_v2_unet_results,data/processed,outputs/tables,outputs/figures,outputs/reports}"
scp "${ROOT}/configs/cairngorms_v2_unet.yaml" "${HOST}:${REMOTE}/configs/"
scp "${ROOT}/docs/phase28_cairngorms_v2_unet_design.md" "${HOST}:${REMOTE}/docs/"
scp \
  "${ROOT}/scripts/tessera_v2_common.py" \
  "${ROOT}/scripts/cairngorms_dense_10m_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_height_adjusted.py" \
  "${ROOT}/scripts/acquire_cairngorms_v2_unet_cairngorms_v2_dense.py" \
  "${ROOT}/scripts/prepare_cairngorms_v2_unet.py" \
  "${ROOT}/scripts/cairngorms_v2_unet_worker.py" \
  "${ROOT}/scripts/aggregate_cairngorms_v2_unet.py" \
  "${HOST}:${REMOTE}/scripts/"
scp "${ROOT}/scripts/jasmin/cairngorms_v2_unet_"*.sbatch "${HOST}:${REMOTE}/scripts/jasmin/"

ssh "${HOST}" "ROOT='${REMOTE}' ENVIRONMENT='${ENVIRONMENT}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
required=(
  references/tessera_v2/fetch_v2_patch_aneesh_20260811.py
  data/raw/scotland_lidar/LiDAR_metrics_2.tif
  data/processed/phase25_cairngorms_corrected_lidar_targets.parquet
  data/processed/phase27_cairngorms_v2_predictions.parquet
  data/interim/cairngorms_dense_10m/embedding_int8.npy
  data/interim/cairngorms_dense_10m/embedding_scale.npy
  data/interim/cairngorms_dense_10m/tessera_valid.npy
  data/interim/phase22_cairngorms_spatial_unet/row_id.npy
  data/interim/phase22_cairngorms_spatial_unet/row.npy
  data/interim/phase22_cairngorms_spatial_unet/column.npy
  data/interim/phase25_cairngorms_corrected_lidar/row_id.npy
  data/interim/phase25_cairngorms_corrected_lidar/spatial_block.npy
  data/interim/phase25_cairngorms_corrected_lidar/fold_0.npz
  data/interim/phase25_cairngorms_corrected_lidar/fold_4.npz
  data/interim/phase27_cairngorms_v2/patch_quantized.npy
  data/interim/phase27_cairngorms_v2/patch_scales.npy
  data/interim/phase27_cairngorms_v2/patch_valid.npy
  data/interim/phase27_cairngorms_v2/row_id.npy
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
  scripts/acquire_cairngorms_v2_unet_cairngorms_v2_dense.py \
  scripts/prepare_cairngorms_v2_unet.py \
  scripts/cairngorms_v2_unet_worker.py \
  scripts/aggregate_cairngorms_v2_unet.py
if [[ -f metadata/phase28_job_ids.txt ]]; then
  source metadata/phase28_job_ids.txt
  jobs="${acquire_job:-0},${prepare_job:-0},${train_job:-0},${aggregate_job:-0}"
  if squeue -h -j "${jobs}" 2>/dev/null | grep -q .; then
    echo "Phase 28 is already active"
    cat metadata/phase28_job_ids.txt
    exit 0
  fi
fi
acquire_job=$(sbatch --parsable scripts/jasmin/cairngorms_v2_unet_acquire.sbatch)
prepare_job=$(sbatch --parsable --dependency="afterok:${acquire_job}" scripts/jasmin/cairngorms_v2_unet_prepare.sbatch)
train_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" scripts/jasmin/cairngorms_v2_unet_train_array.sbatch)
aggregate_job=$(sbatch --parsable --dependency="afterok:${train_job}" scripts/jasmin/cairngorms_v2_unet_aggregate.sbatch)
cat > metadata/phase28_job_ids.txt <<EOF
acquire_job=${acquire_job}
prepare_job=${prepare_job}
train_job=${train_job}
aggregate_job=${aggregate_job}
EOF
chmod -R g+rwX configs docs scripts logs metadata data/interim/phase28_cairngorms_v2_dense data/interim/phase28_cairngorms_v2_stream data/interim/phase28_cairngorms_v2_unet data/interim/phase28_cairngorms_v2_unet_results outputs
cat metadata/phase28_job_ids.txt
REMOTE_SCRIPT

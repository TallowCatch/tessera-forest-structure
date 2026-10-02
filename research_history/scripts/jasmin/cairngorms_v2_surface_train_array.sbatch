#!/usr/bin/env bash
#SBATCH --job-name=cairn_p29_v2_surface
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-59%10
#SBATCH --output=logs/phase29_train_%A_%a.out
#SBATCH --error=logs/phase29_train_%A_%a.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
SEEDS=(20260804 20260805 20260806)
variant_index=$((SLURM_ARRAY_TASK_ID / 30))
within_variant=$((SLURM_ARRAY_TASK_ID % 30))
model_index=$((within_variant / 15))
remainder=$((within_variant % 15))
fold=$((remainder / 3))
seed_index=$((remainder % 3))

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"

if [[ "${variant_index}" -eq 0 ]]; then
  RAW_MODELS=(tessera_mlp_5x5 fused_mlp_5x5)
  python scripts/cairngorms_v2_surface_v2_surface_raw_worker.py \
    --model "${RAW_MODELS[$model_index]}" --fold "${fold}" --seed "${SEEDS[$seed_index]}"
else
  ADJUSTED_MODELS=(tessera fused)
  python scripts/cairngorms_v2_surface_v2_surface_adjusted_worker.py \
    --model "${ADJUSTED_MODELS[$model_index]}" --fold "${fold}" --seed "${SEEDS[$seed_index]}"
fi


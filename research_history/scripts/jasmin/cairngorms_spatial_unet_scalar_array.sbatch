#!/usr/bin/env bash
#SBATCH --job-name=cairn_p22_scalar
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-149%15
#SBATCH --output=logs/phase22_scalar_%A_%a.out
#SBATCH --error=logs/phase22_scalar_%A_%a.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
SCHEMES=(balanced regional)
MODELS=(conventional_mlp tessera_mlp_3x3 tessera_mlp_5x5 fused_mlp_5x5 tessera_cnn_5x5)
SEEDS=(20260804 20260805 20260806)

scheme_index=$((SLURM_ARRAY_TASK_ID / 75))
remainder=$((SLURM_ARRAY_TASK_ID % 75))
model_index=$((remainder / 15))
remainder=$((remainder % 15))
fold=$((remainder / 3))
seed_index=$((remainder % 3))

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/cairngorms_spatial_unet_spatial_unet_worker.py \
  --scheme "${SCHEMES[$scheme_index]}" \
  --model "${MODELS[$model_index]}" \
  --fold "${fold}" \
  --seed "${SEEDS[$seed_index]}"


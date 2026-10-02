#!/usr/bin/env bash
#SBATCH --job-name=cairn_p33_flip
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-29%10
#SBATCH --output=logs/phase33_train_%A_%a.out
#SBATCH --error=logs/phase33_train_%A_%a.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
SEEDS=(20260804 20260805 20260806)
variant_index=$((SLURM_ARRAY_TASK_ID / 15))
remainder=$((SLURM_ARRAY_TASK_ID % 15))
fold=$((remainder / 3))
seed_index=$((remainder % 3))
VARIANTS=(raw height_adjusted)

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"

python scripts/reflection_augmentation_flip_augmented_worker.py \
  --variant "${VARIANTS[$variant_index]}" \
  --fold "${fold}" \
  --seed "${SEEDS[$seed_index]}"


#!/usr/bin/env bash
#SBATCH --job-name=cairn_ordered
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-14%6

set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-ordered-patch"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
MODELS=(ordered_mlp ordered_skip_mlp residual_patch_cnn)
SEEDS=(20260731 20260801 20260802)

model_index=$((SLURM_ARRAY_TASK_ID / 5))
fold=$((SLURM_ARRAY_TASK_ID % 5))
model="${MODELS[$model_index]}"

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
for seed in "${SEEDS[@]}"; do
  python scripts/cairngorms_ordered_patch_worker.py \
    --model "${model}" \
    --fold "${fold}" \
    --seed "${seed}"
done

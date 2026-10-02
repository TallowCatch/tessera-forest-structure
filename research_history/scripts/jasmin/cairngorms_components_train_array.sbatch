#!/usr/bin/env bash
#SBATCH --job-name=cairn_p20_train
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-44%9

set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
MODELS=(conventional tessera fused)
SEEDS=(20260801 20260802 20260803)

model_index=$((SLURM_ARRAY_TASK_ID / 15))
remainder=$((SLURM_ARRAY_TASK_ID % 15))
fold=$((remainder / 3))
seed_index=$((remainder % 3))
model="${MODELS[$model_index]}"
seed="${SEEDS[$seed_index]}"

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python scripts/cairngorms_components_components_worker.py --model "${model}" --fold "${fold}" --seed "${seed}"

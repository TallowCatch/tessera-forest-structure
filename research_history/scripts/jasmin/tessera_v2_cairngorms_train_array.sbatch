#!/usr/bin/env bash
#SBATCH --job-name=cairn_p27_v2_train
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-29%10
#SBATCH --output=logs/phase27_cairngorms_train_%A_%a.out
#SBATCH --error=logs/phase27_cairngorms_train_%A_%a.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
MODELS=(tessera fused)
SEEDS=(20260804 20260805 20260806)
model_index=$((SLURM_ARRAY_TASK_ID / 15))
remainder=$((SLURM_ARRAY_TASK_ID % 15))
fold=$((remainder / 3))
seed_index=$((remainder % 3))
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/tessera_v2_cairngorms_v2_worker.py \
  --model "${MODELS[$model_index]}" --fold "${fold}" --seed "${SEEDS[$seed_index]}"

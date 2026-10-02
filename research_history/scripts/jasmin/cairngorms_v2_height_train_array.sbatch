#!/usr/bin/env bash
#SBATCH --job-name=cairn_p30_height
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=0-29%8
#SBATCH --output=logs/phase30_train_%A_%a.out
#SBATCH --error=logs/phase30_train_%A_%a.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
SEEDS=(20260804 20260805 20260806)
MODELS=(tessera_v2_mlp_5x5 tessera_v2_unet_audited)
model_index=$((SLURM_ARRAY_TASK_ID / 15))
within_model=$((SLURM_ARRAY_TASK_ID % 15))
fold=$((within_model / 3))
seed_index=$((within_model % 3))

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/cairngorms_v2_height_worker.py \
  --model "${MODELS[$model_index]}" \
  --fold "${fold}" \
  --seed "${SEEDS[$seed_index]}"

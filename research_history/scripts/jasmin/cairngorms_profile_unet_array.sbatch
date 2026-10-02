#!/usr/bin/env bash
#SBATCH --job-name=cairn_prof_unet
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-14%5

set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
SEEDS=(20260801 20260802 20260803)

fold=$((SLURM_ARRAY_TASK_ID / 3))
seed_index=$((SLURM_ARRAY_TASK_ID % 3))
seed="${SEEDS[$seed_index]}"

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python scripts/cairngorms_dense_profile_worker.py \
  --model compact_unet \
  --fold "${fold}" \
  --seed "${seed}"

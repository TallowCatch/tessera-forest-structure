#!/usr/bin/env bash
#SBATCH --job-name=cairn_p22_unet
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-29%5
#SBATCH --output=logs/phase22_unet_%A_%a.out
#SBATCH --error=logs/phase22_unet_%A_%a.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
SCHEMES=(balanced regional)
SEEDS=(20260804 20260805 20260806)

scheme_index=$((SLURM_ARRAY_TASK_ID / 15))
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
python scripts/cairngorms_spatial_unet_spatial_unet_worker.py \
  --scheme "${SCHEMES[$scheme_index]}" \
  --model tessera_unet \
  --fold "${fold}" \
  --seed "${SEEDS[$seed_index]}"


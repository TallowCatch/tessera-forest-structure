#!/usr/bin/env bash
#SBATCH --job-name=cairn_p21_prepare
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/phase21_prepare_%j.out
#SBATCH --error=logs/phase21_prepare_%j.err

set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
module load jaspy/3.12/v20250704
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python scripts/prepare_cairngorms_surface.py

#!/usr/bin/env bash
#SBATCH --job-name=cairn_dense_aggregate
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
module load jaspy/3.12/v20250704
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python scripts/aggregate_cairngorms_dense_10m.py

#!/usr/bin/env bash
#SBATCH --job-name=savel_p27_v2_evaluate
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/phase27_savelsbos_evaluate_%j.out
#SBATCH --error=logs/phase27_savelsbos_evaluate_%j.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/evaluate_tessera_v2_savelsbos_v2.py

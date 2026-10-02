#!/bin/bash
#SBATCH --account=forecol
#SBATCH --job-name=dutch_p37_eval
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=0-19%10
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --output=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase37_eval_%A_%a.out
#SBATCH --error=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase37_eval_%A_%a.err

set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd /gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python scripts/evaluate_dutch_expanded.py worker --index "$SLURM_ARRAY_TASK_ID"

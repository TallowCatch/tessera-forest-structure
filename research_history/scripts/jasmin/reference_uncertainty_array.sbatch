#!/bin/bash
#SBATCH --account=forecol
#SBATCH --job-name=cairn_p40_ref
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --array=0-19%10
#SBATCH --time=02:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --output=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase40_ref_%A_%a.out
#SBATCH --error=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase40_ref_%A_%a.err

set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd /gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m
export OMP_NUM_THREADS=1
python scripts/analyze_reference_uncertainty_cairngorms_reference_uncertainty.py worker --index "$SLURM_ARRAY_TASK_ID"

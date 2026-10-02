#!/bin/bash
#SBATCH --account=forecol
#SBATCH --job-name=cairn_p40_agg
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=01:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --output=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase40_aggregate_%j.out
#SBATCH --error=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase40_aggregate_%j.err

set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd /gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m
export MPLBACKEND=Agg
python scripts/analyze_reference_uncertainty_cairngorms_reference_uncertainty.py aggregate

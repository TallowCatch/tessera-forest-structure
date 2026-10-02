#!/bin/bash
#SBATCH --account=forecol
#SBATCH --job-name=dutch_p37_aggregate
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase37_aggregate_%j.out
#SBATCH --error=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase37_aggregate_%j.err

set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd /gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m
export MPLBACKEND=Agg
python scripts/evaluate_dutch_expanded.py aggregate

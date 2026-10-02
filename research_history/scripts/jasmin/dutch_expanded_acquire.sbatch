#!/bin/bash
#SBATCH --account=forecol
#SBATCH --job-name=dutch_p37_v1
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --output=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase37_acquire_%j.out
#SBATCH --error=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase37_acquire_%j.err

set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd /gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m
export DUTCH_MULTISITE_CONFIG=configs/dutch_expanded_transfer.yaml
export DUTCH_MULTISITE_CONFIG_KEY=phase37_dutch_expanded_transfer
export DUTCH_MULTISITE_PHASE=phase37
python scripts/acquire_dutch_multisite_dutch_v1.py

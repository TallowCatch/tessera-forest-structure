#!/bin/bash
#SBATCH --account=forecol
#SBATCH --job-name=dutch_p36_v1
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase36_acquire_%j.out
#SBATCH --error=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m/logs/phase36_acquire_%j.err

set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd /gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m
python scripts/acquire_dutch_multisite_dutch_v1.py

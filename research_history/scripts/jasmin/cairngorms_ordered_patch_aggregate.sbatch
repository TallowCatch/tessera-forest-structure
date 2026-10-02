#!/usr/bin/env bash
#SBATCH --job-name=cairn_aggregate
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-ordered-patch"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"

module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export PYTHONUNBUFFERED=1

cd "${PROJECT_ROOT}"
python scripts/aggregate_cairngorms_ordered_patch.py

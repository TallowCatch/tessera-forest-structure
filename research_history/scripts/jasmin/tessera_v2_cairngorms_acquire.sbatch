#!/usr/bin/env bash
#SBATCH --job-name=cairn_p27_v2_acquire
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/phase27_cairngorms_acquire_%j.out
#SBATCH --error=logs/phase27_cairngorms_acquire_%j.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/acquire_tessera_v2.py --site cairngorms

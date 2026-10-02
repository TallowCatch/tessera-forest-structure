#!/usr/bin/env bash
#SBATCH --job-name=cairn_p28_aggregate
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=logs/phase28_aggregate_%j.out
#SBATCH --error=logs/phase28_aggregate_%j.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/aggregate_cairngorms_v2_unet.py


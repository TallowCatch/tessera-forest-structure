#!/usr/bin/env bash
#SBATCH --job-name=ahn4_p26_tessera
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=high
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=logs/phase26_tessera_%j.out
#SBATCH --error=logs/phase26_tessera_%j.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ENVIRONMENT="/home/users/ameer096/venvs/tessera-cairngorms-lotus"
module load jaspy/3.12/v20250704
source "${ENVIRONMENT}/bin/activate"
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python scripts/acquire_ahn4_replication_ahn4_tessera.py

#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "mkdir -p '$remote_root/logs' '$remote_root/configs' '$remote_root/docs' '$remote_root/scripts/jasmin' '$remote_root/tests' '$remote_root/metadata'"
rsync -av "$local_root/configs/reference_uncertainty_cairngorms_reference_uncertainty.yaml" "$remote_host:$remote_root/configs/"
rsync -av "$local_root/docs/phase40_cairngorms_reference_uncertainty_design.md" "$remote_host:$remote_root/docs/"
rsync -av "$local_root/scripts/analyze_reference_uncertainty_cairngorms_reference_uncertainty.py" "$remote_host:$remote_root/scripts/"
rsync -av "$local_root/scripts/jasmin/reference_uncertainty_array.sbatch" "$local_root/scripts/jasmin/reference_uncertainty_aggregate.sbatch" "$remote_host:$remote_root/scripts/jasmin/"
rsync -av "$local_root/tests/test_reference_uncertainty.py" "$remote_host:$remote_root/tests/"

ssh "$remote_host" "
set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd '$remote_root'
if [ -f metadata/phase40_reference_uncertainty_result.json ]; then
  echo 'Phase 40 already has a result freeze; refusing to overwrite.' >&2
  exit 1
fi
rm -rf data/interim/phase40_reference_uncertainty_workers
python -m pytest -q tests/test_reference_uncertainty.py
python -m py_compile scripts/analyze_reference_uncertainty_cairngorms_reference_uncertainty.py
workers=\$(sbatch --parsable scripts/jasmin/reference_uncertainty_array.sbatch)
aggregate=\$(sbatch --parsable --dependency=afterok:\$workers scripts/jasmin/reference_uncertainty_aggregate.sbatch)
cat > metadata/phase40_job_chain.txt <<EOF
workers=\$workers
aggregate=\$aggregate
EOF
cat metadata/phase40_job_chain.txt
"

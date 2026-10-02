#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "mkdir -p '$remote_root/logs' '$remote_root/configs' '$remote_root/docs' '$remote_root/scripts/jasmin' '$remote_root/tests'"
rsync -av "$local_root/configs/dutch_rank_transfer.yaml" "$remote_host:$remote_root/configs/"
rsync -av "$local_root/docs/phase39_dutch_rank_transfer_design.md" "$remote_host:$remote_root/docs/"
rsync -av "$local_root/scripts/evaluate_dutch_reference_assisted.py" "$local_root/scripts/evaluate_dutch_rank_transfer.py" "$remote_host:$remote_root/scripts/"
rsync -av "$local_root/scripts/jasmin/dutch_rank_transfer_rank_transfer_array.sbatch" "$local_root/scripts/jasmin/dutch_rank_transfer_rank_transfer_aggregate.sbatch" "$remote_host:$remote_root/scripts/jasmin/"
rsync -av "$local_root/tests/test_dutch_rank_transfer.py" "$remote_host:$remote_root/tests/"

ssh "$remote_host" "
set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd '$remote_root'
if [ -f metadata/phase39_rank_transfer_result.json ]; then
  echo 'Phase 39 already has a result freeze; refusing to overwrite.' >&2
  exit 1
fi
python -m pytest -q tests/test_dutch_rank_transfer.py
python -m py_compile scripts/evaluate_dutch_rank_transfer.py
workers=\$(sbatch --parsable scripts/jasmin/dutch_rank_transfer_rank_transfer_array.sbatch)
aggregate=\$(sbatch --parsable --dependency=afterok:\$workers scripts/jasmin/dutch_rank_transfer_rank_transfer_aggregate.sbatch)
cat > metadata/phase39_job_chain.txt <<EOF
workers=\$workers
aggregate=\$aggregate
EOF
cat metadata/phase39_job_chain.txt
"

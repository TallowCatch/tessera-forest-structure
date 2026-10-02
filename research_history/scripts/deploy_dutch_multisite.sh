#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "mkdir -p '$remote_root/logs' '$remote_root/configs' '$remote_root/scripts/jasmin' '$remote_root/tests'"
rsync -av \
  "$local_root/configs/dutch_multisite_transfer.yaml" \
  "$remote_host:$remote_root/configs/"
rsync -av \
  "$local_root/scripts/prepare_dutch_multisite.py" \
  "$local_root/scripts/acquire_dutch_multisite_dutch_v1.py" \
  "$local_root/scripts/acquire_dutch_multisite_dutch_v2.py" \
  "$local_root/scripts/evaluate_dutch_multisite_dutch_transfer.py" \
  "$local_root/scripts/prepare_ahn4_replication_ahn4_cohort.py" \
  "$local_root/scripts/acquire_ahn4_replication_ahn4_tessera.py" \
  "$local_root/scripts/evaluate_ahn4_replication_ahn4.py" \
  "$local_root/scripts/tessera_v2_common.py" \
  "$remote_host:$remote_root/scripts/"
rsync -av \
  "$local_root/scripts/jasmin/dutch_multisite_prepare.sbatch" \
  "$local_root/scripts/jasmin/dutch_multisite_acquire.sbatch" \
  "$local_root/scripts/jasmin/dutch_multisite_evaluate.sbatch" \
  "$remote_host:$remote_root/scripts/jasmin/"
rsync -av \
  "$local_root/tests/test_dutch_multisite_transfer.py" \
  "$remote_host:$remote_root/tests/"

ssh "$remote_host" "
set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd '$remote_root'
python -m pytest -q tests/test_dutch_multisite_transfer.py
python -m py_compile scripts/prepare_dutch_multisite.py scripts/acquire_dutch_multisite_dutch_v1.py scripts/acquire_dutch_multisite_dutch_v2.py scripts/evaluate_dutch_multisite_dutch_transfer.py
prepare=\$(sbatch --parsable scripts/jasmin/dutch_multisite_prepare.sbatch)
acquire=\$(sbatch --parsable --dependency=afterok:\$prepare scripts/jasmin/dutch_multisite_acquire.sbatch)
evaluate=\$(sbatch --parsable --dependency=afterok:\$acquire scripts/jasmin/dutch_multisite_evaluate.sbatch)
cat > metadata/phase36_job_chain.txt <<EOF
prepare_job=\$prepare
acquire_job=\$acquire
evaluate_job=\$evaluate
EOF
cat metadata/phase36_job_chain.txt
"

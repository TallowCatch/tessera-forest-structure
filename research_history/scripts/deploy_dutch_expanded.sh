#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "mkdir -p '$remote_root/logs' '$remote_root/configs' '$remote_root/docs' '$remote_root/scripts/jasmin' '$remote_root/tests'"
rsync -av \
  "$local_root/configs/dutch_expanded_transfer.yaml" \
  "$remote_host:$remote_root/configs/"
rsync -av \
  "$local_root/docs/phase37_dutch_expanded_transfer_design.md" \
  "$remote_host:$remote_root/docs/"
rsync -av \
  "$local_root/scripts/prepare_dutch_expanded.py" \
  "$local_root/scripts/evaluate_dutch_expanded.py" \
  "$local_root/scripts/prepare_dutch_multisite.py" \
  "$local_root/scripts/acquire_dutch_multisite_dutch_v1.py" \
  "$local_root/scripts/evaluate_dutch_multisite_dutch_transfer.py" \
  "$local_root/scripts/prepare_ahn4_replication_ahn4_cohort.py" \
  "$local_root/scripts/acquire_ahn4_replication_ahn4_tessera.py" \
  "$local_root/scripts/evaluate_ahn4_replication_ahn4.py" \
  "$local_root/scripts/tessera_v2_common.py" \
  "$remote_host:$remote_root/scripts/"
rsync -av \
  "$local_root/scripts/jasmin/dutch_expanded_prepare.sbatch" \
  "$local_root/scripts/jasmin/dutch_expanded_acquire.sbatch" \
  "$local_root/scripts/jasmin/dutch_expanded_evaluate_array.sbatch" \
  "$local_root/scripts/jasmin/dutch_expanded_aggregate.sbatch" \
  "$remote_host:$remote_root/scripts/jasmin/"
rsync -av \
  "$local_root/tests/test_dutch_multisite_transfer.py" \
  "$local_root/tests/test_dutch_expanded_transfer.py" \
  "$remote_host:$remote_root/tests/"

ssh "$remote_host" "
set -euo pipefail
module load jaspy/3.12/v20250704
source /home/users/ameer096/venvs/tessera-cairngorms-lotus/bin/activate
cd '$remote_root'
if [ -f metadata/phase37_dutch_expanded_preparation.json ] || [ -f metadata/phase37_dutch_expanded_result.json ]; then
  echo 'Phase 37 already has frozen remote outputs; refusing to overwrite.' >&2
  exit 1
fi
python -m pytest -q tests/test_dutch_multisite_transfer.py tests/test_dutch_expanded_transfer.py
python -m py_compile scripts/prepare_dutch_expanded.py scripts/acquire_dutch_multisite_dutch_v1.py scripts/evaluate_dutch_expanded.py
prepare=\$(sbatch --parsable scripts/jasmin/dutch_expanded_prepare.sbatch)
acquire=\$(sbatch --parsable --dependency=afterok:\$prepare scripts/jasmin/dutch_expanded_acquire.sbatch)
evaluate=\$(sbatch --parsable --dependency=afterok:\$acquire scripts/jasmin/dutch_expanded_evaluate_array.sbatch)
aggregate=\$(sbatch --parsable --dependency=afterok:\$evaluate scripts/jasmin/dutch_expanded_aggregate.sbatch)
cat > metadata/phase37_job_chain.txt <<EOF
prepare_job=\$prepare
acquire_job=\$acquire
evaluate_job=\$evaluate
aggregate_job=\$aggregate
EOF
cat metadata/phase37_job_chain.txt
"

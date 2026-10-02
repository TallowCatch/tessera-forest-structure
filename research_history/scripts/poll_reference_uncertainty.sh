#!/bin/bash
set -euo pipefail

remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "
cd '$remote_root'
workers=\$(sed -n 's/^workers=//p' metadata/phase40_job_chain.txt 2>/dev/null)
aggregate=\$(sed -n 's/^aggregate=//p' metadata/phase40_job_chain.txt 2>/dev/null)
echo '=== job chain ==='
cat metadata/phase40_job_chain.txt 2>/dev/null || true
echo '=== LOTUS status ==='
if [ -n \"\$workers\" ]; then
  squeue -r -j \"\$workers,\$aggregate\" -o '%.18i %.25j %.2t %.10M %.10l %R' || true
fi
echo '=== worker outputs ==='
complete=\$(find data/interim/phase40_reference_uncertainty_workers -name 'worker_*.parquet' 2>/dev/null | wc -l)
echo \"\$complete/20 worker outputs complete\"
if [ -n \"\$workers\" ]; then
  tail -n 2 logs/phase40_ref_\${workers}_*.out 2>/dev/null | tail -n 20 || true
  grep -h -E 'Traceback|RuntimeError|ERROR|No space|failed' logs/phase40_ref_\${workers}_*.err 2>/dev/null | tail -n 12 || true
fi
echo '=== aggregate ==='
if [ -f metadata/phase40_reference_uncertainty_result.json ]; then
  python -c \"import json; x=json.load(open('metadata/phase40_reference_uncertainty_result.json')); print('complete: {} units, {} metrics, {} blocks per unit'.format(x['units'], x['metrics'], x['jackknife_blocks_per_unit']))\"
elif [ -n \"\$aggregate\" ]; then
  tail -n 15 logs/phase40_aggregate_\${aggregate}.out 2>/dev/null || echo 'not complete'
  tail -n 10 logs/phase40_aggregate_\${aggregate}.err 2>/dev/null || true
fi
"

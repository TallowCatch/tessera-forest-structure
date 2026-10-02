#!/bin/bash
set -euo pipefail

remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "
cd '$remote_root'
echo '=== job chain ==='
cat metadata/phase38_job_chain.txt 2>/dev/null || true
jobs=\$(awk -F= '{print \$2}' metadata/phase38_job_chain.txt 2>/dev/null | paste -sd, -)
workers=\$(awk -F= '\$1=="workers" {print \$2}' metadata/phase38_job_chain.txt 2>/dev/null)
aggregate=\$(awk -F= '\$1=="aggregate" {print \$2}' metadata/phase38_job_chain.txt 2>/dev/null)
echo '=== LOTUS status ==='
if [ -n \"\$jobs\" ]; then
  squeue -r -j \"\$workers,\$aggregate\" -o '%.18i %.25j %.2t %.10M %.10l %R' || true
fi
echo '=== worker outputs ==='
complete=\$(find data/interim/phase38_reference_assisted_workers -name 'worker_*.csv' 2>/dev/null | wc -l)
echo \"\$complete/20 worker outputs complete\"
if [ -n \"\$workers\" ]; then
  tail -n 2 logs/phase38_ref_\${workers}_*.out 2>/dev/null | tail -n 20 || true
  grep -h -E 'Traceback|RuntimeError|ERROR|No space|failed' logs/phase38_ref_\${workers}_*.err 2>/dev/null | tail -n 12 || true
fi
echo '=== aggregate ==='
if [ -f metadata/phase38_reference_assisted_result.json ]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase38_reference_assisted_result.json'))
print('complete:', x['sites'], 'sites;', x['metric_rows'], 'metric rows')
PY
else
  if [ -n \"\$aggregate\" ]; then
    tail -n 18 logs/phase38_aggregate_\${aggregate}.out 2>/dev/null || echo 'not complete'
    tail -n 12 logs/phase38_aggregate_\${aggregate}.err 2>/dev/null || true
  else
    echo 'not complete'
  fi
fi
"

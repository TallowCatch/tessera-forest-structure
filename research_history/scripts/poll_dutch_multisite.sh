#!/bin/bash
set -euo pipefail

remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "
cd '$remote_root'
echo '=== job chain ==='
cat metadata/phase36_job_chain.txt 2>/dev/null || true
jobs=\$(awk -F= '{print \$2}' metadata/phase36_job_chain.txt 2>/dev/null | paste -sd, -)
echo '=== LOTUS status ==='
if [ -n \"\$jobs\" ]; then
  squeue -j \"\$jobs\" -o '%.18i %.24j %.2t %.10M %.10l %R' || true
fi
echo '=== preparation ==='
if [ -f metadata/phase36_dutch_multisite_preparation.json ]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase36_dutch_multisite_preparation.json'))
print(f\"complete: {x['rows']:,} rows\")
for s in x['sites']:
    print(f\"{s['site_name']}: {s['retained_units']:,} units; years={s['tessera_year_counts']}\")
PY
else
  tail -n 12 logs/phase36_prepare_*.out 2>/dev/null || echo 'not started'
  tail -n 8 logs/phase36_prepare_*.err 2>/dev/null || true
fi
echo '=== public TESSERA v1.0 acquisition ==='
if [ -f metadata/phase36_dutch_multisite_v1_acquisition.json ]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase36_dutch_multisite_v1_acquisition.json'))
print(f\"complete: {x['valid_rows']:,}/{x['rows']:,} valid contexts\")
PY
else
  tail -n 15 logs/phase36_acquire_*.out 2>/dev/null || echo 'not started'
  tail -n 8 logs/phase36_acquire_*.err 2>/dev/null || true
fi
echo '=== evaluation ==='
if [ -f metadata/phase36_dutch_multisite_result.json ]; then
  cat outputs/reports/phase36_dutch_multisite_transfer.md
else
  tail -n 15 logs/phase36_evaluate_*.out 2>/dev/null || echo 'not started'
  tail -n 8 logs/phase36_evaluate_*.err 2>/dev/null || true
fi
"

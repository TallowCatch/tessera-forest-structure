#!/bin/bash
set -euo pipefail

remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "
cd '$remote_root'
echo '=== job chain ==='
cat metadata/phase37_job_chain.txt 2>/dev/null || true
jobs=\$(awk -F= '{print \$2}' metadata/phase37_job_chain.txt 2>/dev/null | paste -sd, -)
echo '=== LOTUS status ==='
if [ -n \"\$jobs\" ]; then
  squeue -j \"\$jobs\" -o '%.18i %.25j %.2t %.10M %.10l %R' || true
fi
echo '=== site screening and cohort ==='
if [ -f metadata/phase37_dutch_expanded_preparation.json ]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase37_dutch_expanded_preparation.json'))
print(f\"complete: {len(x['sites'])} sites, {x['rows']:,} rows\")
for s in x['sites']:
    print(f\"{s['forest_group'][:1].upper()}  {s['site_name']}: {s['retained_units']:,} units; {s['selection_reason']}\")
PY
else
  tail -n 20 logs/phase37_prepare_*.out 2>/dev/null || echo 'not started'
  tail -n 12 logs/phase37_prepare_*.err 2>/dev/null || true
fi
echo '=== public TESSERA v1.0 acquisition ==='
if [ -f metadata/phase37_dutch_expanded_v1_acquisition.json ]; then
  python - <<'PY'
import json
x=json.load(open('metadata/phase37_dutch_expanded_v1_acquisition.json'))
print(f\"complete: {x['valid_rows']:,}/{x['rows']:,} valid contexts\")
PY
else
  tail -n 18 logs/phase37_acquire_*.out 2>/dev/null || echo 'not started'
  grep -h -E 'Traceback|RuntimeError|ERROR|No space|failed' logs/phase37_acquire_*.err 2>/dev/null | tail -n 10 || true
fi
echo '=== evaluation workers ==='
expected=20
if [ -f outputs/tables/phase37_dutch_site_summary.csv ]; then
  expected=\$(awk 'END {print NR - 1}' outputs/tables/phase37_dutch_site_summary.csv)
fi
complete=\$(find data/interim/phase37_dutch_evaluation_workers -name 'worker_*.parquet' 2>/dev/null | wc -l)
echo \"\$complete/\$expected worker outputs complete\"
tail -n 2 logs/phase37_eval_*.out 2>/dev/null | tail -n 20 || true
grep -h -E 'Traceback|RuntimeError|ERROR|No space|failed' logs/phase37_eval_*.err 2>/dev/null | tail -n 10 || true
echo '=== final report ==='
if [ -f metadata/phase37_dutch_expanded_result.json ]; then
  cat outputs/reports/phase37_dutch_expanded_transfer.md
else
  tail -n 18 logs/phase37_aggregate_*.out 2>/dev/null || echo 'not complete'
  tail -n 10 logs/phase37_aggregate_*.err 2>/dev/null || true
fi
"

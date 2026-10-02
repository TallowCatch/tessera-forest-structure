#!/usr/bin/env bash
set -euo pipefail
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -u
cd "${ROOT}"
if [[ ! -f metadata/phase26_ahn4_download_job.txt ]]; then
  echo "AHN4 acquisition has not been submitted"; exit 0
fi
source metadata/phase26_ahn4_download_job.txt
echo "=== job ==="
squeue -j "${download_job}" -o '%.20i %.24j %.2t %.10M %.10l %R' 2>/dev/null || true
echo "=== files ==="
python - <<'PY'
from pathlib import Path
manifest = Path('metadata/phase26_ahn4_download_manifest.tsv').read_text().splitlines()[1:]
for row in manifest:
    filename, expected, _, _ = row.split('\t')
    path = Path('data/raw/ahn4') / filename
    size = path.stat().st_size if path.exists() else 0
    print(f'{filename}: {size:,}/{int(expected):,} bytes ({100 * size / int(expected):.1f}%)')
PY
echo "=== latest log ==="
tail -n 12 "logs/phase26_download_ahn4_${download_job}.out" 2>/dev/null || true
tail -n 12 "logs/phase26_download_ahn4_${download_job}.err" 2>/dev/null || true
if [[ -f metadata/phase26_ahn4_download_complete.txt ]]; then
  echo "complete: $(cat metadata/phase26_ahn4_download_complete.txt)"
else
  echo "not complete"
fi
REMOTE_SCRIPT

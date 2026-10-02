#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HOST="jasmin-sci-vm01"
REMOTE="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"

ssh "${HOST}" "umask 002; mkdir -p '${REMOTE}'/{data/raw/ahn4,metadata,logs,scripts/jasmin}"
rsync -a "${ROOT}/metadata/phase26_ahn4_download_manifest.tsv" "${HOST}:${REMOTE}/metadata/"
rsync -a "${ROOT}/scripts/jasmin/ahn4_replication_download_ahn4.sbatch" "${HOST}:${REMOTE}/scripts/jasmin/"

ssh "${HOST}" "ROOT='${REMOTE}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail
cd "${ROOT}"
if [[ -f metadata/phase26_ahn4_download_job.txt ]]; then
  source metadata/phase26_ahn4_download_job.txt
  if squeue -h -j "${download_job:-0}" 2>/dev/null | grep -q .; then
    echo "AHN4 download already active: ${download_job}"
    exit 0
  fi
fi
download_job=$(sbatch --parsable scripts/jasmin/ahn4_replication_download_ahn4.sbatch)
printf 'download_job=%s\n' "${download_job}" > metadata/phase26_ahn4_download_job.txt
chmod -R g+rwX data/raw/ahn4 metadata logs scripts/jasmin
cat metadata/phase26_ahn4_download_job.txt
REMOTE_SCRIPT

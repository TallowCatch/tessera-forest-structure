#!/usr/bin/env bash
#SBATCH --job-name=ahn4_metrics
#SBATCH --account=forecol
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/phase26_download_ahn4_%j.out
#SBATCH --error=logs/phase26_download_ahn4_%j.err

set -euo pipefail
ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
MANIFEST="${ROOT}/metadata/phase26_ahn4_download_manifest.tsv"
OUTPUT="${ROOT}/data/raw/ahn4"
mkdir -p "${OUTPUT}"
cd "${OUTPUT}"

tail -n +2 "${MANIFEST}" | while IFS=$'\t' read -r filename bytes checksum url; do
  if [[ -f "${filename}" ]] && [[ $(stat -c %s "${filename}") -eq ${bytes} ]] && \
      [[ $(md5sum "${filename}" | cut -d' ' -f1) == "${checksum}" ]]; then
    echo "verified ${filename}"; continue
  fi
  echo "downloading ${filename} (${bytes} bytes)"
  curl --location --fail --silent --show-error --retry 8 --retry-all-errors --continue-at - \
    --output "${filename}" "${url}"
  actual_bytes=$(stat -c %s "${filename}")
  actual_md5=$(md5sum "${filename}" | cut -d' ' -f1)
  [[ "${actual_bytes}" -eq "${bytes}" ]]
  [[ "${actual_md5}" == "${checksum}" ]]
  echo "verified ${filename}"
done

date -u +%FT%TZ > "${ROOT}/metadata/phase26_ahn4_download_complete.txt"

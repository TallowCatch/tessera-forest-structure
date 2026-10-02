#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

ssh "$remote_host" "test -f '$remote_root/metadata/phase37_dutch_expanded_result.json'" || {
  echo "Phase 37 evaluation is not complete yet." >&2
  exit 1
}
mkdir -p \
  "$local_root/outputs/figures/phase37_dutch_expanded" \
  "$local_root/outputs/tables" \
  "$local_root/outputs/reports" \
  "$local_root/metadata"
rsync -av \
  "$remote_host:$remote_root/outputs/figures/phase37_dutch_expanded/" \
  "$local_root/outputs/figures/phase37_dutch_expanded/"
rsync -av \
  "$remote_host:$remote_root/outputs/tables/phase37_dutch_"'*.csv' \
  "$local_root/outputs/tables/"
rsync -av \
  "$remote_host:$remote_root/outputs/reports/phase37_dutch_expanded_transfer.md" \
  "$local_root/outputs/reports/"
rsync -av \
  "$remote_host:$remote_root/metadata/phase37_dutch_expanded_"'*.json' \
  "$local_root/metadata/"
rsync -av \
  "$remote_host:$remote_root/metadata/phase37_dutch_expanded_v1_acquisition.json" \
  "$local_root/metadata/"
echo "Phase 37 outputs collected under $local_root/outputs."

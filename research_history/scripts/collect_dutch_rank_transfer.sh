#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

mkdir -p "$local_root/outputs/tables" "$local_root/outputs/reports" "$local_root/outputs/figures/phase39_rank_transfer" "$local_root/metadata"
rsync -av "$remote_host:$remote_root/outputs/tables/phase39_rank_transfer_fold_metrics.csv" "$local_root/outputs/tables/"
rsync -av "$remote_host:$remote_root/outputs/tables/phase39_rank_transfer_summary.csv" "$local_root/outputs/tables/"
rsync -av "$remote_host:$remote_root/outputs/tables/phase39_zero_shot_rank_summary.csv" "$local_root/outputs/tables/"
rsync -av "$remote_host:$remote_root/outputs/tables/phase39_rank_transfer_site_summary.csv" "$local_root/outputs/tables/"
rsync -av "$remote_host:$remote_root/outputs/reports/phase39_rank_transfer.md" "$local_root/outputs/reports/"
rsync -av "$remote_host:$remote_root/outputs/figures/phase39_rank_transfer/" "$local_root/outputs/figures/phase39_rank_transfer/"
rsync -av "$remote_host:$remote_root/metadata/phase39_rank_transfer_result.json" "$local_root/metadata/"

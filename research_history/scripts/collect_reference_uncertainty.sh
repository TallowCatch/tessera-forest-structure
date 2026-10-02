#!/bin/bash
set -euo pipefail

local_root=/Users/ameerfiras/ICCS/tessera-gedi-fhd
remote_host=jasmin-sci-vm01
remote_root=/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m

mkdir -p "$local_root/data/processed" "$local_root/outputs/tables" "$local_root/outputs/figures" "$local_root/outputs/reports" "$local_root/metadata"
rsync -av "$remote_host:$remote_root/data/processed/phase40_cairngorms_reference_uncertainty.parquet" "$local_root/data/processed/"
rsync -av "$remote_host:$remote_root/outputs/tables/phase40_reference_uncertainty_summary.csv" "$local_root/outputs/tables/"
rsync -av "$remote_host:$remote_root/outputs/tables/phase40_reference_uncertainty_fold_comparison.csv" "$local_root/outputs/tables/"
rsync -av "$remote_host:$remote_root/outputs/figures/phase40_reference_uncertainty.png" "$local_root/outputs/figures/"
rsync -av "$remote_host:$remote_root/outputs/reports/phase40_reference_uncertainty.md" "$local_root/outputs/reports/"
rsync -av "$remote_host:$remote_root/metadata/phase40_reference_uncertainty_result.json" "$local_root/metadata/"

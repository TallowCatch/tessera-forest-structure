#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-ordered-patch"
cd "${PROJECT_ROOT}"
mkdir -p logs data/interim/cairngorms_ordered_patch_results

array_output=$(sbatch \
  --output="${PROJECT_ROOT}/logs/ordered_%A_%a.out" \
  --error="${PROJECT_ROOT}/logs/ordered_%A_%a.err" \
  scripts/jasmin/cairngorms_ordered_patch_array.sbatch)
array_job=${array_output##* }

aggregate_output=$(sbatch \
  --dependency="afterok:${array_job}" \
  --output="${PROJECT_ROOT}/logs/aggregate_%j.out" \
  --error="${PROJECT_ROOT}/logs/aggregate_%j.err" \
  scripts/jasmin/cairngorms_ordered_patch_aggregate.sbatch)
aggregate_job=${aggregate_output##* }

printf 'array_job=%s\naggregate_job=%s\n' "${array_job}" "${aggregate_job}"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
cd "${PROJECT_ROOT}"
mkdir -p logs data/interim/cairngorms_dense_10m_results

prepare_output=$(sbatch \
  --output="${PROJECT_ROOT}/logs/prepare_%j.out" \
  --error="${PROJECT_ROOT}/logs/prepare_%j.err" \
  scripts/jasmin/cairngorms_dense_prepare.sbatch)
prepare_job=${prepare_output##* }

scalar_output=$(sbatch \
  --dependency="afterok:${prepare_job}" \
  --output="${PROJECT_ROOT}/logs/scalar_%A_%a.out" \
  --error="${PROJECT_ROOT}/logs/scalar_%A_%a.err" \
  scripts/jasmin/cairngorms_dense_scalar_array.sbatch)
scalar_job=${scalar_output##* }

unet_output=$(sbatch \
  --dependency="afterok:${prepare_job}" \
  --output="${PROJECT_ROOT}/logs/unet_%A_%a.out" \
  --error="${PROJECT_ROOT}/logs/unet_%A_%a.err" \
  scripts/jasmin/cairngorms_dense_unet_array.sbatch)
unet_job=${unet_output##* }

aggregate_output=$(sbatch \
  --dependency="afterok:${scalar_job}:${unet_job}" \
  --output="${PROJECT_ROOT}/logs/aggregate_%j.out" \
  --error="${PROJECT_ROOT}/logs/aggregate_%j.err" \
  scripts/jasmin/cairngorms_dense_aggregate.sbatch)
aggregate_job=${aggregate_output##* }

printf 'prepare_job=%s\nscalar_job=%s\nunet_job=%s\naggregate_job=%s\n' \
  "${prepare_job}" "${scalar_job}" "${unet_job}" "${aggregate_job}" \
  | tee metadata/cairngorms_dense_10m_job_ids.txt

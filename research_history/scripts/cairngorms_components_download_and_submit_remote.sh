#!/usr/bin/env bash
set -euo pipefail

ROOT="/gws/ssde/j25a/forecol/ameer096/tessera-gedi-fhd-dense-10m"
cd "${ROOT}"
mkdir -p data/raw/scotland_lidar/chm data/interim/phase20_cairngorms_component_results metadata logs

status_file="metadata/phase20_download_status.txt"
job_file="metadata/phase20_job_ids.txt"

download() {
  local name="$1"
  local url="$2"
  local path="$3"
  local expected="$4"
  local current=0
  if [[ -f "${path}" ]]; then
    current=$(stat -c %s "${path}")
    if [[ "${current}" == "${expected}" ]]; then
      printf '%s complete %s/%s\n' "${name}" "${current}" "${expected}" | tee -a "${status_file}"
      return
    fi
    mv "${path}" "${path}.part"
  fi
  if [[ -f "${path}.part" ]]; then
    current=$(stat -c %s "${path}.part")
  fi
  printf '%s downloading %s/%s\n' "${name}" "${current}" "${expected}" | tee -a "${status_file}"
  curl --location --fail --silent --show-error --retry 20 --retry-delay 20 \
    --continue-at - --user-agent 'Mozilla/5.0' --output "${path}.part" "${url}"
  current=$(stat -c %s "${path}.part")
  if [[ "${current}" != "${expected}" ]]; then
    printf '%s failed_size %s/%s\n' "${name}" "${current}" "${expected}" | tee -a "${status_file}"
    exit 1
  fi
  mv "${path}.part" "${path}"
  printf '%s complete %s/%s\n' "${name}" "${current}" "${expected}" | tee -a "${status_file}"
}

: > "${status_file}"
download \
  chm_1m \
  'https://2019.filemail.com/api/file/get?filekey=OvU1Vhhj6mbBvDAiYCcgVUk-EdefSkeQPXcuilZTbGZBa6W8rKgzoV2sJFWWxSUHuypex9XxxeoV2uyC-SzMpBXSIWI-icNXxyv27w' \
  'data/raw/scotland_lidar/chm/Cairngorms_standard_CHM_DSM_DTM_1m_res.tif' \
  6007617253
download \
  slope_mask \
  'https://2109.filemail.com/api/file/get?filekey=Yba2GU1uhzGRBwtqC95YgY8uqJ2GJlVruL_Cphj4H1B7eaP8173g9wbkFguklT3yZx0oCkZGnxTRhvCSQpbajjk' \
  'data/raw/scotland_lidar/chm/Cairngorms_slope_mask_45_1m.tif' \
  101932911
download \
  slope_of_slope_mask \
  'https://2109.filemail.com/api/file/get?filekey=E6WMNE85jrJ0sOGRBWt1P2xy1dffxqFC8DH23tkzyMAhuZCo3---PWVX0l394RuPshKru1-o2TRng_fy19hBA0YzFjvu82ChLeQ2Tw' \
  'data/raw/scotland_lidar/chm/Cairngorms_slope_of_slope_mask_84_5_1m.tif' \
  114554275

if [[ -f "${job_file}" ]]; then
  echo "Phase 20 jobs were already submitted:" | tee -a "${status_file}"
  cat "${job_file}" | tee -a "${status_file}"
  exit 0
fi

prepare_output=$(sbatch \
  --output="${ROOT}/logs/phase20_prepare_%j.out" \
  --error="${ROOT}/logs/phase20_prepare_%j.err" \
  scripts/jasmin/cairngorms_components_prepare.sbatch)
prepare_job=${prepare_output##* }

train_output=$(sbatch \
  --dependency="afterok:${prepare_job}" \
  --output="${ROOT}/logs/phase20_train_%A_%a.out" \
  --error="${ROOT}/logs/phase20_train_%A_%a.err" \
  scripts/jasmin/cairngorms_components_train_array.sbatch)
train_job=${train_output##* }

aggregate_output=$(sbatch \
  --dependency="afterok:${train_job}" \
  --output="${ROOT}/logs/phase20_aggregate_%j.out" \
  --error="${ROOT}/logs/phase20_aggregate_%j.err" \
  scripts/jasmin/cairngorms_components_aggregate.sbatch)
aggregate_job=${aggregate_output##* }

printf 'prepare_job=%s\ntrain_job=%s\naggregate_job=%s\n' \
  "${prepare_job}" "${train_job}" "${aggregate_job}" | tee "${job_file}"
printf 'submitted %s\n' "$(date -u +%FT%TZ)" | tee -a "${status_file}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/Users/ameerfiras/miniforge3/envs/tessera-crop-label-audit/bin/python}"
LOG_PATH="${ROOT}/outputs/reports/phase15_arran_background.log"
ERROR_LOG_PATH="${ROOT}/outputs/reports/phase15_arran_background.error.log"
PID_PATH="${ROOT}/outputs/reports/phase15_arran_background.pid"
LAUNCH_LABEL="uk.ac.cam.tessera-gedi-fhd.phase15-arran"

mkdir -p "${ROOT}/outputs/reports"

if launchctl print "gui/${UID}/${LAUNCH_LABEL}" >/dev/null 2>&1; then
  EXISTING_PID="$(
    launchctl print "gui/${UID}/${LAUNCH_LABEL}" |
      awk '/^[[:space:]]*pid = / {print $3; exit}'
  )"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf 'Phase 15 is already running as PID %s\n' "${EXISTING_PID}"
    exit 0
  fi
  launchctl bootout "gui/${UID}/${LAUNCH_LABEL}" >/dev/null 2>&1 || true
fi

: >"${LOG_PATH}"
: >"${ERROR_LOG_PATH}"
launchctl submit \
  -l "${LAUNCH_LABEL}" \
  -o "${LOG_PATH}" \
  -e "${ERROR_LOG_PATH}" \
  -- /usr/bin/env PYTHONUNBUFFERED=1 /usr/bin/nice -n 10 \
  "${PYTHON_BIN}" "${ROOT}/scripts/run_arran_lidar.py"

PID=""
for _ in {1..20}; do
  PID="$(
    launchctl print "gui/${UID}/${LAUNCH_LABEL}" 2>/dev/null |
      awk '/^[[:space:]]*pid = / {print $3; exit}'
  )"
  [[ -n "${PID}" ]] && break
  sleep 0.25
done
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
  printf 'Phase 15 did not remain active. See %s\n' "${ERROR_LOG_PATH}" >&2
  exit 1
fi
printf '%s\n' "${PID}" >"${PID_PATH}"
printf 'Started Phase 15 as PID %s\nLog: %s\nErrors: %s\n' \
  "${PID}" "${LOG_PATH}" "${ERROR_LOG_PATH}"

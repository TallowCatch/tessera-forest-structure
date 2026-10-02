#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-/Users/ameerfiras/miniforge3/envs/tessera-crop-label-audit/bin/python}"
LOG_PATH="${ROOT}/outputs/reports/phase14_scotland_background.log"
PID_PATH="${ROOT}/outputs/reports/phase14_scotland_background.pid"

mkdir -p "${ROOT}/outputs/reports"

if [[ -f "${PID_PATH}" ]]; then
  EXISTING_PID="$(cat "${PID_PATH}")"
  if kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf 'Phase 14 is already running as PID %s\n' "${EXISTING_PID}"
    exit 0
  fi
fi

nohup "${PYTHON_BIN}" "${ROOT}/scripts/run_scotland_lidar.py" \
  >"${LOG_PATH}" 2>&1 </dev/null &
PID="$!"
printf '%s\n' "${PID}" >"${PID_PATH}"
printf 'Started Phase 14 as PID %s\nLog: %s\n' "${PID}" "${LOG_PATH}"

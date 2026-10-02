#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG="${ROOT}/outputs/reports/cairngorms_dense_10m_deploy.log"
PID="${ROOT}/outputs/reports/cairngorms_dense_10m_deploy.pid"
mkdir -p "$(dirname "${LOG}")"

if [[ -f "${PID}" ]] && kill -0 "$(cat "${PID}")" 2>/dev/null; then
  echo "Deployment already running with PID $(cat "${PID}")"
  exit 0
fi

nohup bash "${ROOT}/scripts/deploy_cairngorms_dense_10m_to_jasmin.sh" \
  >"${LOG}" 2>&1 &
echo $! >"${PID}"
echo "Started JASMIN deployment with local PID $(cat "${PID}")"
echo "Log: ${LOG}"

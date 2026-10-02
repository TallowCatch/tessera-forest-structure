#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG="${ROOT}/outputs/reports/phase22_deploy.log"
PID="${ROOT}/outputs/reports/phase22_deploy.pid"
mkdir -p "$(dirname "${LOG}")"
if [[ -f "${PID}" ]] && kill -0 "$(cat "${PID}")" 2>/dev/null; then
  echo "Phase 22 deployment already runs as local PID $(cat "${PID}")"
  exit 0
fi
nohup bash "${ROOT}/scripts/deploy_cairngorms_spatial_unet_cairngorms_to_jasmin.sh" >"${LOG}" 2>&1 &
echo $! >"${PID}"
echo "Started Phase 22 deployment as local PID $(cat "${PID}")"
echo "Poll with: bash scripts/poll_cairngorms_spatial_unet_cairngorms_jasmin.sh"


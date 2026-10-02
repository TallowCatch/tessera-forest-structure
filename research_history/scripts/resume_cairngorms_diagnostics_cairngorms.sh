#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="${ROOT}/outputs/reports/phase19_cairngorms_background.pid"
STATE_PATH="${ROOT}/outputs/reports/phase19_cairngorms_pause_state.txt"

if [[ ! -f "${PID_PATH}" ]]; then
  printf 'No Phase 19 PID file exists.\n' >&2
  exit 1
fi
PID="$(tr -d '[:space:]' <"${PID_PATH}")"
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
  printf 'Phase 19 is no longer running. Use the background launcher to restart it.\n'
  exit 1
fi
PGID="$(ps -o pgid= -p "${PID}" | tr -d '[:space:]')"
if [[ -z "${PGID}" ]]; then
  printf 'Could not determine the Phase 19 process group.\n' >&2
  exit 1
fi
kill -CONT -- "-${PGID}"
printf 'resumed_utc=%s\npid=%s\nprocess_group=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${PID}" "${PGID}" \
  >"${STATE_PATH}"
printf 'Resumed Phase 19 process group %s.\n' "${PGID}"

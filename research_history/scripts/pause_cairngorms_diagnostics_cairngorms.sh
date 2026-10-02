#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_PATH="${ROOT}/outputs/reports/phase19_cairngorms_background.pid"
STATE_PATH="${ROOT}/outputs/reports/phase19_cairngorms_pause_state.txt"
SCHEDULE_LABEL="uk.ac.cam.tessera-gedi-fhd.phase19-pause"
DOMAIN="gui/$(id -u)"

if [[ ! -f "${PID_PATH}" ]]; then
  printf 'No Phase 19 PID file exists; the job has probably finished.\n'
elif ! PID="$(tr -d '[:space:]' <"${PID_PATH}")" ||
  [[ -z "${PID}" ]] ||
  ! kill -0 "${PID}" 2>/dev/null; then
  printf 'Phase 19 is no longer running.\n'
else
  PGID="$(ps -o pgid= -p "${PID}" | tr -d '[:space:]')"
  if [[ -z "${PGID}" ]]; then
    printf 'Could not determine the Phase 19 process group.\n' >&2
    exit 1
  fi
  kill -STOP -- "-${PGID}"
  printf 'paused_utc=%s\npid=%s\nprocess_group=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "${PID}" "${PGID}" \
    >"${STATE_PATH}"
  printf 'Paused Phase 19 process group %s.\n' "${PGID}"
fi

# The calendar agent is intentionally one-shot.
launchctl bootout "${DOMAIN}/${SCHEDULE_LABEL}" >/dev/null 2>&1 || true

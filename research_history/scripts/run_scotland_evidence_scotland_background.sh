#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="uk.ac.cam.tessera-gedi-fhd.phase16-scotland"
PLIST="${ROOT}/scripts/${LABEL}.plist"
PID_PATH="${ROOT}/outputs/reports/phase16_scotland_background.pid"
LOG_PATH="${ROOT}/outputs/reports/phase16_scotland_background.log"
ERROR_PATH="${ROOT}/outputs/reports/phase16_scotland_background.error.log"
DOMAIN="gui/${UID}"

mkdir -p "${ROOT}/outputs/reports"

if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
  EXISTING_PID="$(
    launchctl print "${DOMAIN}/${LABEL}" |
      awk '/^[[:space:]]*pid = / {print $3; exit}'
  )"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf 'Phase 16 is already running as PID %s\n' "${EXISTING_PID}"
    exit 0
  fi
  launchctl bootout "${DOMAIN}/${LABEL}" >/dev/null 2>&1 || true
fi

: >"${LOG_PATH}"
: >"${ERROR_PATH}"
launchctl bootstrap "${DOMAIN}" "${PLIST}"

PID=""
for _ in {1..40}; do
  CANDIDATE_PID="$(
    launchctl print "${DOMAIN}/${LABEL}" 2>/dev/null |
      awk '/^[[:space:]]*pid = / {print $3; exit}'
  )"
  if [[ -n "${CANDIDATE_PID}" ]] && kill -0 "${CANDIDATE_PID}" 2>/dev/null; then
    PID="${CANDIDATE_PID}"
    break
  fi
  sleep 0.25
done
if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
  printf 'Phase 16 did not remain active. See %s\n' "${ERROR_PATH}" >&2
  exit 1
fi
printf '%s\n' "${PID}" >"${PID_PATH}"
printf 'Started Phase 16 as PID %s\nLog: %s\nErrors: %s\n' \
  "${PID}" "${LOG_PATH}" "${ERROR_PATH}"

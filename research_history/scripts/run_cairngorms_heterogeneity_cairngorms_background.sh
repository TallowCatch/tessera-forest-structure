#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_PATH="${ROOT}/outputs/reports/phase17_cairngorms_background.log"
PID_PATH="${ROOT}/outputs/reports/phase17_cairngorms_background.pid"
PLIST_PATH="${ROOT}/scripts/uk.ac.cam.tessera-gedi-fhd.cairngorms_heterogeneity_cairngorms.plist"
LABEL="uk.ac.cam.tessera-gedi-fhd.phase17-cairngorms"
DOMAIN="gui/$(id -u)"

mkdir -p "${ROOT}/outputs/reports"

if launchctl print "${DOMAIN}/${LABEL}" >/dev/null 2>&1; then
  EXISTING_PID="$(
    launchctl print "${DOMAIN}/${LABEL}" |
      awk '/^[[:space:]]*pid = / {print $3; exit}'
  )"
  if [[ -n "${EXISTING_PID}" ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
    printf '%s\n' "${EXISTING_PID}" >"${PID_PATH}"
    printf 'Cairngorms heterogeneity study is already running as PID %s\n' \
      "${EXISTING_PID}"
    exit 0
  fi
  launchctl kickstart -k "${DOMAIN}/${LABEL}"
else
  launchctl bootstrap "${DOMAIN}" "${PLIST_PATH}"
fi

sleep 1
PID="$(
  launchctl print "${DOMAIN}/${LABEL}" |
    awk '/^[[:space:]]*pid = / {print $3; exit}'
)"
if [[ -z "${PID}" ]]; then
  printf 'LaunchAgent did not expose a running PID. Check %s\n' "${LOG_PATH}" >&2
  exit 1
fi
printf '%s\n' "${PID}" >"${PID_PATH}"

printf 'Started Cairngorms heterogeneity study as PID %s\n' "${PID}"
printf 'Log: %s\n' "${LOG_PATH}"

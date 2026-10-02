#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.pid"
LOG_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.log"
STATUS_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.exit_status"
RESULT="$ROOT/metadata/phase34_scotland_netherlands_pooled_result_freeze.json"
LABEL="org.iccs.tessera.phase34-pooled"

mkdir -p "$(dirname "$PID_FILE")"

if [[ -f "$RESULT" ]]; then
  echo "Phase 34 is already complete: $RESULT"
  exit 0
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Phase 34 is already running with PID $(cat "$PID_FILE")"
  exit 0
fi

cd "$ROOT"
rm -f "$STATUS_FILE"
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl remove "$LABEL"
fi
launchctl submit -l "$LABEL" -- /bin/bash \
  "$ROOT/scripts/run_pooled_transfer_scotland_netherlands_pooled_background.sh"
sleep 1
PID="$(cat "$PID_FILE")"
echo "Started Phase 34 through launchd with PID $PID"
echo "Log: $LOG_FILE"

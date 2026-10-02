#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/outputs/reports/phase35_scotland_netherlands_pooled_mlp.pid"
STATUS_FILE="$ROOT/outputs/reports/phase35_scotland_netherlands_pooled_mlp.exit_status"
RESULT="$ROOT/metadata/phase35_scotland_netherlands_pooled_mlp_result_freeze.json"
LABEL="org.iccs.tessera.phase35-pooled-mlp"

mkdir -p "$(dirname "$PID_FILE")"
if [[ -f "$RESULT" ]]; then
  echo "Phase 35 is already complete: $RESULT"
  exit 0
fi
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Phase 35 is already running with PID $(cat "$PID_FILE")"
  exit 0
fi

rm -f "$STATUS_FILE"
if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
  launchctl remove "$LABEL"
fi
launchctl submit -l "$LABEL" -- /bin/bash \
  "$ROOT/scripts/run_pooled_mlp_scotland_netherlands_pooled_mlp_background.sh"
sleep 1
echo "Started Phase 35 through launchd with PID $(cat "$PID_FILE")"
echo "Poll with: bash scripts/poll_pooled_mlp_scotland_netherlands_pooled_mlp.sh"


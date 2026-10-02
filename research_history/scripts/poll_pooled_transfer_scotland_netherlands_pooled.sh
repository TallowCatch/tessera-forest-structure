#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.pid"
LOG_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.log"
STATUS_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.exit_status"
RESULT="$ROOT/metadata/phase34_scotland_netherlands_pooled_result_freeze.json"

if [[ -f "$RESULT" ]]; then
  echo "complete"
elif [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  ps -p "$(cat "$PID_FILE")" -o pid,stat,etime,%cpu,%mem,command
elif [[ -f "$STATUS_FILE" ]] && [[ "$(cat "$STATUS_FILE")" != "0" ]]; then
  echo "failed with exit status $(cat "$STATUS_FILE")"
else
  echo "not running"
fi

echo "=== latest log ==="
if [[ -f "$LOG_FILE" ]]; then
  tail -n 30 "$LOG_FILE"
else
  echo "no log yet"
fi

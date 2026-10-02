#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/outputs/reports/phase35_scotland_netherlands_pooled_mlp.pid"
LOG_FILE="$ROOT/outputs/reports/phase35_scotland_netherlands_pooled_mlp.log"
STATUS_FILE="$ROOT/outputs/reports/phase35_scotland_netherlands_pooled_mlp.exit_status"
LABEL="org.iccs.tessera.phase35-pooled-mlp"

cd "$ROOT"
printf "%s\n" "$$" >"$PID_FILE"
: >"$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 \
  /Users/ameerfiras/miniforge3/bin/python -u \
  scripts/evaluate_pooled_mlp_scotland_netherlands_pooled_mlp.py
status=$?
printf "%s\n" "$status" >"$STATUS_FILE"
launchctl remove "$LABEL" >/dev/null 2>&1 || true
exit "$status"


#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.pid"
LOG_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.log"
STATUS_FILE="$ROOT/outputs/reports/phase34_scotland_netherlands_pooled.exit_status"

cd "$ROOT"
printf "%s\n" "$$" >"$PID_FILE"
: >"$LOG_FILE"
exec >>"$LOG_FILE" 2>&1

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /Users/ameerfiras/miniforge3/bin/python -u \
  scripts/evaluate_pooled_transfer_scotland_netherlands_pooled.py
status=$?
printf "%s\n" "$status" >"$STATUS_FILE"
exit "$status"

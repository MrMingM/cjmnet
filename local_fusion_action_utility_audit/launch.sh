#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
RUN="${RUN:-/data/cjm/datasets/logs/action_utility_audit_$(date +%Y%m%d_%H%M%S)}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
export RUN PY
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main

# Validate the resolved location before creating output files, also for resume.
"$PY" -c 'import sys; from local_fusion_action_utility_audit.common import validate_run; validate_run(sys.argv[1])' "$RUN"
if test -f "$RUN/pid" && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
  echo "RUN already has a live launcher PID" >&2
  exit 1
fi
export AUDIT_LAUNCH_REDIRECT=1
nohup sh local_fusion_action_utility_audit/run_all.sh "$@" >> "$RUN/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$PID" "$RUN/driver.log"

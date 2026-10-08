#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
: "${SOURCE_RUN:?Set SOURCE_RUN explicitly to the original action utility run}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/action_label_control_$(date +%Y%m%d_%H%M%S)}"
MODE="${MODE:-offline}"
export RUN SOURCE_RUN PY MODE
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
"$PY" -B -c 'import sys; from local_fusion_action_label_control.common import isolated_paths; isolated_paths(sys.argv[1], sys.argv[2], create=True)' "$SOURCE_RUN" "$RUN"
if test -f "$RUN/pid" && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
  echo "RUN already has a live launcher PID" >&2
  exit 1
fi
export LABEL_CONTROL_REDIRECT=1
nohup sh local_fusion_action_label_control/run_all.sh "$@" >> "$RUN/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\nMODE=%s\n' "$RUN" "$PID" "$RUN/driver.log" "$MODE"

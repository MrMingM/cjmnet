#!/bin/sh
set -eu
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/source_kill_stage2_precheck_$(date +%Y%m%d_%H%M%S)}"
export PY RUN
"$PY" -c 'import sys; from local_fusion_source_kill_stage2_precheck.common import output_path; output_path(sys.argv[1], server=True)' "$RUN"
if [ -f "$RUN/pid" ] && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
    printf '%s\n' 'RUN already has a live launcher PID' >&2
    exit 1
fi
nohup sh local_fusion_source_kill_stage2_precheck/run_precheck.sh >>"$RUN/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$PID" "$RUN/driver.log"

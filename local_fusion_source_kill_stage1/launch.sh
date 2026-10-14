#!/bin/sh
set -eu
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-2}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/source_kill_stage1_$(date +%Y%m%d_%H%M%S)}"
export PY RUN
"$PY" -m local_fusion_source_kill_stage1.common --run "$RUN" --validate-run
if [ -f "$RUN/pid" ] && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
    printf '%s\n' 'RUN already has a live driver PID' >&2
    exit 1
fi
nohup sh local_fusion_source_kill_stage1/run_stage1.sh >>"$RUN/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$PID" "$RUN/driver.log"

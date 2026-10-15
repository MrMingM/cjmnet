#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${PREVIOUS_RUN:?Set PREVIOUS_RUN to the failed v1 repair RUN}"
export PYTHONPATH="$ROOT:$(dirname -- "$ROOT")"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-3}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
RUN="${RUN:-/data/cjm/datasets/logs/source_kill_stage2_k2_repair_v2_$(date +%Y%m%d_%H%M%S)}"
"$PY" -c 'import sys; from local_fusion_source_kill_stage2.common import external_run,resource_check,settings; resource_check(settings()); external_run(sys.argv[1])' "$RUN"
nohup "$PY" -u -m local_fusion_source_kill_stage2_replay_fix_v2.driver --run "$RUN" --previous-run "$PREVIOUS_RUN" >>"$RUN/driver.log" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$PID" "$RUN/driver.log"

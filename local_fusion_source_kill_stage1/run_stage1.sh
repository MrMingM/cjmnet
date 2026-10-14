#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-2}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260927
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
RUN="${RUN:-/data/cjm/datasets/logs/source_kill_stage1_$(date +%Y%m%d_%H%M%S)}"
export PY RUN

if [ "${STAGE1_LOCK_HELD:-0}" != 1 ]; then
    exec "$PY" -u -m local_fusion_source_kill_stage1.common --run "$RUN" --lock-driver
fi

K0_PID=
K10_PID=
cleanup() {
    trap - EXIT HUP INT TERM
    if [ -n "$K0_PID" ]; then kill "$K0_PID" 2>/dev/null || :; fi
    if [ -n "$K10_PID" ]; then kill "$K10_PID" 2>/dev/null || :; fi
    if [ -n "$K0_PID" ]; then wait "$K0_PID" 2>/dev/null || :; fi
    if [ -n "$K10_PID" ]; then wait "$K10_PID" 2>/dev/null || :; fi
    "$PY" -m local_fusion_source_kill_stage1.common --run "$RUN" --state failed || :
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

"$PY" -u -m local_fusion_source_kill_stage1.common --run "$RUN" --preflight
"$PY" -u -m unittest local_fusion_source_kill_stage1.test_core

(
    export ROCR_VISIBLE_DEVICES=""
    export HIP_VISIBLE_DEVICES=""
    export CUDA_VISIBLE_DEVICES=""
    exec "$PY" -u -m local_fusion_source_kill_stage1.k0_group_margin --run "$RUN"
) >>"$RUN/k0.log" 2>&1 &
K0_PID=$!
printf '%s\n' "$K0_PID" >"$RUN/k0.pid"

"$PY" -u -m local_fusion_source_kill_stage1.k1_0_oracle_decomposition --run "$RUN" \
    >>"$RUN/k1_0.log" 2>&1 &
K10_PID=$!
printf '%s\n' "$K10_PID" >"$RUN/k1_0.pid"
printf 'K0_PID=%s K1_0_PID=%s\n' "$K0_PID" "$K10_PID"

STATUS=0
wait "$K0_PID" || STATUS=1
K0_PID=
if [ "$STATUS" -ne 0 ]; then
    kill "$K10_PID" 2>/dev/null || :
fi
wait "$K10_PID" || STATUS=1
K10_PID=
if [ "$STATUS" -ne 0 ]; then
    exit 1
fi

"$PY" -u -m local_fusion_source_kill_stage1.summarize_stage1 --run "$RUN"
trap - EXIT HUP INT TERM

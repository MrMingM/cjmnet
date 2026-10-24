#!/bin/sh
set -eu
: "${PROJECT_ROOT:=/home/cjm/OpenCOOD-main/cjmnet}"
: "${PYTHON:=/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${MODE:=all}" "${SEED:=20261019}" "${EPOCHS:=20}" "${AMP:=0}"
: "${BACKGROUND:=1}" "${RESUME:=0}"
: "${REUSE_DATA_FROM:=}"
: "${TRAIN_ROOT:=/data/cjm/datasets/v2xr-official-train-v2}"
: "${VALIDATE_ROOT:=/data/cjm/datasets/v2xr-official-validate-v1/validate}"
: "${PAIRED_VALIDATE_ROOT:=/data/cjm/datasets/v2xr-lidar-weather-validate-v1}"
case "$SEED" in ''|*[!0-9]*) echo 'SEED must be uint32' >&2; exit 2;; esac
[ "${#SEED}" -le 10 ] && [ "$SEED" -le 4294967295 ] || { echo 'SEED must be uint32' >&2; exit 2; }
SEED=$("$PYTHON" -c 'import sys; print(int(sys.argv[1]))' "$SEED")
case "$BACKGROUND:$RESUME" in 0:0|0:1|1:0|1:1) ;; *) echo 'BACKGROUND and RESUME must be 0 or 1' >&2; exit 2;; esac
if [ "$RESUME" = 1 ] || [ "$MODE" = report ]; then
    : "${OUTPUT_ROOT:?Recovery/report requires the explicit old OUTPUT_ROOT}"
else
    : "${OUTPUT_ROOT:=/data/cjm/datasets/logs/v2xr-raygen-e0e3-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
fi
case "$OUTPUT_ROOT" in /data/*) ;; *) echo 'OUTPUT_ROOT must be absolute on /data' >&2; exit 2;; esac
case "$MODE" in
    all|smoke) : "${GPU:?Set GPU to one physical HIP card}" ;;
    preflight|labels|report) ;;
    *) echo 'MODE must be all/preflight/labels/smoke/report' >&2; exit 2;;
esac
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
if [ -n "${GPU:-}" ]; then
    case "$GPU" in *[!0-9]*|'') echo 'GPU must be one physical integer card' >&2; exit 2;; esac
    ROCR_VISIBLE_DEVICES=$GPU
    export GPU ROCR_VISIBLE_DEVICES
else
    unset ROCR_VISIBLE_DEVICES
fi
PYTHONHASHSEED=$SEED
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/v2xr_raygen_e0e3}
export PROJECT_ROOT PYTHON MODE SEED EPOCHS AMP BACKGROUND RESUME OUTPUT_ROOT REUSE_DATA_FROM
export TRAIN_ROOT VALIDATE_ROOT PAIRED_VALIDATE_ROOT PYTHONHASHSEED TMPDIR
"$PYTHON" -B "$PROJECT_ROOT/v2xr_raygen_e0e3/paths.py" "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$TMPDIR"
cd "$PROJECT_ROOT"
sh -n "$PROJECT_ROOT/launch_v2xr_e0e3.sh"
sh -n "$PROJECT_ROOT/v2xr_raygen_e0e3/run_all.sh"
printf 'OUTPUT_ROOT=%s\nMODE=%s GPU=%s SEED=%s EPOCHS=%s\n' "$OUTPUT_ROOT" "$MODE" "${GPU:-CPU}" "$SEED" "$EPOCHS"
if [ "$BACKGROUND" = 1 ]; then
    nohup env BACKGROUND=0 sh "$PROJECT_ROOT/v2xr_raygen_e0e3/run_all.sh" >> "$OUTPUT_ROOT/launcher.log" 2>&1 < /dev/null &
    task_pid=$!
    printf '%s\n' "$task_pid" > "$OUTPUT_ROOT/launcher.pid.tmp"
    mv "$OUTPUT_ROOT/launcher.pid.tmp" "$OUTPUT_ROOT/launcher.pid"
    printf 'PID=%s\nLOG=%s/launcher.log\n' "$task_pid" "$OUTPUT_ROOT"
    exit 0
fi
exec sh "$PROJECT_ROOT/v2xr_raygen_e0e3/run_all.sh"

#!/bin/sh
set -eu
: "${PROJECT_ROOT:=/home/cjm/OpenCOOD-main/cjmnet}"
: "${PYTHON:=/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${MODE:=all}" "${BACKGROUND:=1}" "${RESUME:=0}"
: "${SEED:=20261017}" "${EPOCHS:=20}" "${MICROBATCH:=1}" "${AMP:=0}"
: "${TRAIN_ROOT:=/data/cjm/datasets/v2xr-official-train-v2}"
: "${VALIDATE_ROOT:=/data/cjm/datasets/v2xr-official-validate-v1/validate}"
: "${PAIRED_VALIDATE_ROOT:=/data/cjm/datasets/v2xr-lidar-weather-validate-v1}"
: "${GPU:?Set GPU to one idle physical ROCm card after checking rocm-smi}"
case "$MODE" in all|preflight|self-test|smoke|report) ;; *) echo 'Invalid MODE' >&2; exit 2;; esac
case "$GPU" in ''|*[!0-9]*) echo 'GPU must be one physical integer card' >&2; exit 2;; esac
case "$BACKGROUND:$RESUME" in 0:0|0:1|1:0|1:1) ;; *) echo 'BACKGROUND and RESUME require0/1' >&2; exit 2;; esac
if [ "$RESUME" = 1 ] || [ "$MODE" = report ]; then
    : "${OUTPUT_ROOT:?Specify the exact old output for recovery/report}"
else
    : "${OUTPUT_ROOT:=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
fi
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
ROCR_VISIBLE_DEVICES=$GPU
PYTHONHASHSEED=$SEED
TMPDIR=${TMPDIR:-/data/cjm/datasets/tmp/uecp_gspr_v2xr}
export PROJECT_ROOT PYTHON MODE BACKGROUND RESUME SEED EPOCHS MICROBATCH AMP
export OUTPUT_ROOT GPU ROCR_VISIBLE_DEVICES PYTHONHASHSEED TMPDIR
export TRAIN_ROOT VALIDATE_ROOT PAIRED_VALIDATE_ROOT
cd "$PROJECT_ROOT"
"$PYTHON" -B "$PROJECT_ROOT/uecp_gspr_v2xr/paths.py" "$OUTPUT_ROOT"
sh -n "$PROJECT_ROOT/uecp_gspr_v2xr/run_all.sh"
mkdir -p "$OUTPUT_ROOT" "$TMPDIR"
if [ "$BACKGROUND" = 1 ]; then
    nohup env BACKGROUND=0 sh "$PROJECT_ROOT/uecp_gspr_v2xr/run_all.sh" >> "$OUTPUT_ROOT/launcher.log" 2>&1 < /dev/null &
    pipeline_pid=$!
    printf '%s\n' "$pipeline_pid" > "$OUTPUT_ROOT/launcher.pid.tmp"
    mv "$OUTPUT_ROOT/launcher.pid.tmp" "$OUTPUT_ROOT/launcher.pid"
    printf 'OUTPUT_ROOT=%s\nPID=%s\nLOG=%s/launcher.log\n' "$OUTPUT_ROOT" "$pipeline_pid" "$OUTPUT_ROOT"
    exit 0
fi
exec "$PYTHON" -B "$PROJECT_ROOT/uecp_gspr_v2xr/pipeline.py" --output "$OUTPUT_ROOT"

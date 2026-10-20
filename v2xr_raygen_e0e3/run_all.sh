#!/bin/sh
set -eu
: "${PROJECT_ROOT:=/home/cjm/OpenCOOD-main/cjmnet}"
: "${PYTHON:=/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${OUTPUT_ROOT:?Use launch_v2xr_e0e3.sh or provide OUTPUT_ROOT}"
if [ -n "${REUSE_DATA_FROM:-}" ]; then
    : "${GPU:?Reuse requires the previously accepted physical GPU}"
    "$PYTHON" -B "$PROJECT_ROOT/v2xr_raygen_e0e3/successor.py" \
        --source "$REUSE_DATA_FROM" --output "$OUTPUT_ROOT" --gpu "$GPU"
    RESUME=1
    export RESUME
fi
exec "$PYTHON" -B "$PROJECT_ROOT/v2xr_raygen_e0e3/controller.py" --output "$OUTPUT_ROOT"

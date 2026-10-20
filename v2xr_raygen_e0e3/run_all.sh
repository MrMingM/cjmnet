#!/bin/sh
set -eu
: "${PROJECT_ROOT:=/home/cjm/OpenCOOD-main/cjmnet}"
: "${PYTHON:=/home/cjm/miniconda3/envs/opencood/bin/python}"
: "${OUTPUT_ROOT:?Use launch_v2xr_e0e3.sh or provide OUTPUT_ROOT}"
exec "$PYTHON" -B "$PROJECT_ROOT/v2xr_raygen_e0e3/controller.py" --output "$OUTPUT_ROOT"

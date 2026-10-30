#!/bin/sh
set -eu
MODE=${MODE:-preflight}
RESUME=${RESUME:-0}
GPU=${GPU:-5}
case "$MODE" in preflight|smoke|full) ;; *) exit 2 ;; esac
case "$RESUME" in 0|1) ;; *) exit 2 ;; esac
cd /home/cjm/OpenCOOD-main/cjmnet
export ROCR_VISIBLE_DEVICES="$GPU"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export TMPDIR=/data/cjm/datasets/tmp/scorw_training
mkdir -p "$TMPDIR"
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
if [ "$RESUME" = 1 ]; then
    exec "$PYTHON" -u -B -m scorw_training.run --mode "$MODE" --gpu "$GPU" --resume
fi
exec "$PYTHON" -u -B -m scorw_training.run --mode "$MODE" --gpu "$GPU"

#!/bin/sh
set -eu
cd "$(dirname "$0")"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES="${GPU:-5}" LR0_EMPTY_POLICY=zero PYTHONHASHSEED=20261017
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/data/cjm/datasets/tmp/v2xr_radar_diagnostics
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/v2xr-radar-diagnostics-v1
exec /home/cjm/miniconda3/envs/opencood/bin/python -uB diagnose.py


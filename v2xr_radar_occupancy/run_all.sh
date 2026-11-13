#!/bin/sh
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero
export TMPDIR=/data/cjm/datasets/tmp/v2xr_radar_occupancy
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
r=/data/cjm/datasets/logs/v2xr-radar-occupancy-v1
mkdir -p "$r" "$TMPDIR"
exec /home/cjm/miniconda3/envs/opencood/bin/python -uB launch.py


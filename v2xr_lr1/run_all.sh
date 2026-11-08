#!/bin/sh
set -eu
cd "$(dirname "$0")"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=1
export PYTHONHASHSEED=20261017
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TMPDIR=/data/cjm/datasets/tmp/v2xr_lr1
mkdir -p "$TMPDIR"
/home/cjm/miniconda3/envs/opencood/bin/python -B static_check.py
exec /home/cjm/miniconda3/envs/opencood/bin/python -uB run.py

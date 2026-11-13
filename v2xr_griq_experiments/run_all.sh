#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export GRIQ_OUT="${GRIQ_OUT:-/data/cjm/datasets/logs/v2xr-griq-experiments-v1}"
export GRIQ_GPU="${GRIQ_GPU:-1}"
export LR0_EMPTY_POLICY=zero PYTHONHASHSEED=20261017
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export TMPDIR=/data/cjm/datasets/tmp/v2xr-griq-experiments
mkdir -p "$GRIQ_OUT" "$TMPDIR"
exec "${GRIQ_PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}" -uB -m v2xr_griq_experiments.launch "$@"

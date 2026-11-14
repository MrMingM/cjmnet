#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export GRIQ2_OUT="${GRIQ2_OUT:-/data/cjm/datasets/logs/v2xr-griq-v2}"
export LR0_EMPTY_POLICY=zero PYTHONHASHSEED=20261017
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/data/cjm/datasets/tmp/v2xr-griq-v2
unset CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
mkdir -p "$GRIQ2_OUT" "$TMPDIR"
exec "${GRIQ2_PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}" -uB -m v2xr_griq_v2.launch "$@"

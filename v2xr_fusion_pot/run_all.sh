#!/bin/sh
set -eu
cd "$(dirname "$0")"
sh -n run_all.sh
export ROCR_VISIBLE_DEVICES=4
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
out=/data/cjm/datasets/logs/v2xr-fusion-pot-v1
TMPDIR="$out/tmp"
export TMPDIR
mkdir -p "$TMPDIR"
exec /home/cjm/miniconda3/envs/opencood/bin/python -uB run.py "$@"

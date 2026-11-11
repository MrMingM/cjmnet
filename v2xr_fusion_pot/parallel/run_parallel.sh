#!/bin/sh
set -eu
cd "$(dirname "$0")"
case "${1:-}" in
  P1|report) ROCR_VISIBLE_DEVICES=4 ;;
  P2) ROCR_VISIBLE_DEVICES=5 ;;
  P3) ROCR_VISIBLE_DEVICES=6 ;;
  *) echo 'Usage: sh run_parallel.sh P1|P2|P3|report' >&2; exit 2 ;;
esac
export ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TMPDIR=/data/cjm/datasets/logs/v2xr-fusion-pot-v1/tmp
export TMPDIR
exec /home/cjm/miniconda3/envs/opencood/bin/python -uB parallel_run.py "$1"

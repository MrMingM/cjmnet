#!/bin/sh
set -eu
cd "$(dirname "$0")"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20261017 LR0_EMPTY_POLICY=zero
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/data/cjm/datasets/tmp/v2xr_radar_evidence
mkdir -p "$TMPDIR" /data/cjm/datasets/logs/v2xr-radar-evidence-v1
exec /home/cjm/miniconda3/envs/opencood/bin/python -uB launch.py "$@"


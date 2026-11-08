#!/bin/sh
set -eu
cd "$(dirname "$0")"
mode=${1:-smoke}
if [ "$#" -gt 0 ]; then shift; fi
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
case "$mode" in
    smoke) ROCR_VISIBLE_DEVICES= ;;
    full) : "${ROCR_VISIBLE_DEVICES:?Set one idle physical HCU index}" ;;
    *) printf '%s\n' 'Usage: sh run_all.sh smoke|full [--seeds 20260906 ...]' >&2; exit 2 ;;
esac
export ROCR_VISIBLE_DEVICES PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}" -u -B experiment.py "$mode" "$@"

#!/bin/sh
set -eu
REPO=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONHASHSEED=20261013
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
: "${GPU:?Set GPU to an available physical HIP GPU index}"
export ROCR_VISIBLE_DEVICES="$GPU"
PY=${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}
exec "$PY" -u audit_reference_weather.py \
  --dataset-root "${DATASET_ROOT:-/data/cjm/datasets/opv2v-reference-weather-smoke-v1}"

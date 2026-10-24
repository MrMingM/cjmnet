#!/bin/sh
# First-frame diagnosis only; no training/repair/score-changing mode.
set -eu
HERE=$(CDPATH= cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=20261016
set -- "$PYTHON" -B -u "$HERE/nms_main.py" --run-root "${RUN_ROOT:-/data/cjm/datasets/logs/gspr-e01-v1}"
if [ "${SELF_TEST:-0}" = 1 ]; then set -- "$@" --self-test; fi
if [ "${TRACE_LAYERS:-0}" = 1 ]; then set -- "$@" --trace-layers; fi
if [ "${BN_COMPARE:-0}" = 1 ]; then set -- "$@" --compare-bn; fi
if [ -n "${OUTPUT_ROOT:-}" ]; then set -- "$@" --output-root "$OUTPUT_ROOT"; fi
if [ -n "${CHECKPOINT:-}" ]; then set -- "$@" --checkpoint "$CHECKPOINT"; fi
if [ -n "${GPU:-}" ]; then set -- "$@" --gpu "$GPU"; fi
exec "$@"

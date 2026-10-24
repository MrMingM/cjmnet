#!/bin/sh
set -eu
HERE=$(CDPATH= cd "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
export PYTHONHASHSEED=20261016
export PYTHONDONTWRITEBYTECODE=1
if [ "${SELF_TEST:-0}" = 1 ]; then
  exec "$PYTHON" -B "$HERE/tests_local.py"
fi
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
set -- "$PYTHON" -B -u "$HERE/controller.py" \
  --output-root "${OUTPUT_ROOT:-/data/cjm/datasets/logs/gspr-e01-formal-warmup-v1}" \
  --legacy-run-root "${LEGACY_RUN_ROOT:-/data/cjm/datasets/logs/gspr-e01-v1}" \
  --original-root "${ORIGINAL_ROOT:-/data/cjm/datasets/logs/v2xr-weather-l012-v1}" \
  --train-root "${TRAIN_ROOT:-/data/cjm/datasets/v2xr-official-train-v2}" \
  --gpu "${GPU:-2}" --resume "${RESUME:-0}"
if [ -n "${E0_EXISTING_ROOT:-}" ]; then set -- "$@" --e0-existing-root "$E0_EXISTING_ROOT"; fi
exec "$@"

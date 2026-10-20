#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$SCRIPT_DIR/../shared_weather_benchmark_p6/config/protocol.json}
RESUME=${RESUME:-1}
case "$RESUME" in
  0|1) ;;
  *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
exec "$PYTHON" -u -B "$SCRIPT_DIR/run.py" --config "$CONFIG" --resume "$RESUME"

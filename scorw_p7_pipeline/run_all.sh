#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
REPO=$(CDPATH= cd -P "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$REPO/scorw_p7/config/protocol.json}
RESUME=${RESUME:-1}
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
exec "$PYTHON" -u -B -m scorw_p7_pipeline.run --config "$CONFIG" --resume "$RESUME"

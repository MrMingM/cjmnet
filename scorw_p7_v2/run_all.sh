#!/bin/sh
set -eu
sh -n "$0"
SCRIPT_DIR=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
REPO=$(CDPATH= cd -P "$SCRIPT_DIR/.." && pwd)
PYTHON=${PYTHON:-/home/cjm/miniconda3/envs/opencood/bin/python}
CONFIG=${CONFIG:-$REPO/scorw_p7_v2/config/protocol.json}
RESUME=${RESUME:-1}
MODE=${MODE:-all}
case "$RESUME" in 0|1) ;; *) echo "RESUME must be 0 or 1" >&2; exit 2 ;; esac
case "$MODE" in all|checks|smoke|full) ;; *) echo "MODE must be all, checks, smoke or full" >&2; exit 2 ;; esac
export PYTHON CONFIG RESUME MODE
export PYTHONDONTWRITEBYTECODE=1
cd "$REPO"
if [ "$MODE" = all ]; then
    exec "$PYTHON" -u -B -m scorw_p7_v2.pipeline --config "$CONFIG" --resume "$RESUME"
fi
exec sh "$SCRIPT_DIR/run_stage.sh"

#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
export PYTHONPATH="$ROOT:$(dirname -- "$ROOT")"
RUN="${RUN:-/data/cjm/datasets/logs/source_kill_stage2_$(date +%Y%m%d_%H%M%S)}"
export RUN
COMPONENT="${COMPONENT:-both}"
case "$COMPONENT" in
    both) SCRIPT=local_fusion_source_kill_stage2/run_stage2.sh ;;
    k1) SCRIPT=local_fusion_source_kill_stage2/run_k1.sh ;;
    k2) SCRIPT=local_fusion_source_kill_stage2/run_k2.sh ;;
    *) echo 'COMPONENT must be both, k1 or k2' >&2; exit 1 ;;
esac
"$PY" -c 'import sys; from local_fusion_source_kill_stage2.common import external_run,resource_check,settings; resource_check(settings()); external_run(sys.argv[1])' "$RUN"
nohup sh "$SCRIPT" >>"$RUN/$COMPONENT-driver.log" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" >"$RUN/$COMPONENT.pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$PID" "$RUN/$COMPONENT-driver.log"

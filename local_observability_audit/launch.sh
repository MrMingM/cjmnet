#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
: "${GPU:?Specify idle physical GPU explicitly; inspect ps and rocm-smi first}"
RUN="${RUN:-/data/cjm/datasets/logs/observability_audit_$(date +%Y%m%d_%H%M%S)}"
export RUN GPU
case "$RUN" in /data/cjm/datasets/logs/observability_audit_*) ;; *) echo 'Invalid RUN' >&2; exit 1;; esac
if [ -e "$RUN" ]; then
    echo 'launch.sh requires a new RUN. Resume via nohup run_all.sh with the identical RUN/mode.' >&2
    exit 1
fi
mkdir "$RUN"
nohup sh local_observability_audit/run_all.sh "$@" >> "$RUN/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s/driver.log\n' "$RUN" "$PID" "$RUN"

#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
RUN="${RUN:-/data/cjm/datasets/logs/source_evidence_audit_$(date +%Y%m%d_%H%M%S)}"
BASE_RUN="${BASE_RUN:-/data/cjm/datasets/logs/action_utility_audit_20261007_210243}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
export RUN BASE_RUN PY
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
"$PY" -c 'import sys; from local_fusion_source_evidence_audit.common import external_run; external_run(sys.argv[1], True); external_run(sys.argv[2])' "$RUN" "$BASE_RUN"
if test -f "$RUN/pid" && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
  echo "RUN already has a live launcher PID" >&2
  exit 1
fi
nohup sh local_fusion_source_evidence_audit/run_all.sh "$@" >> "$RUN/driver.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN/pid"
printf 'RUN=%s\nPID=%s\nLOG=%s\n' "$RUN" "$PID" "$RUN/driver.log"

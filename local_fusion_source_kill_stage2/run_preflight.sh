#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
export PYTHONPATH="$ROOT:$(dirname -- "$ROOT")"
"$PY" -m local_fusion_source_kill_stage2.preflight --static
"$PY" -m unittest local_fusion_source_kill_stage2.test_core
"$PY" -m local_fusion_source_kill_stage2.smoke
"$PY" -m local_fusion_source_kill_stage2.preflight --run "${RUN:?set RUN}"

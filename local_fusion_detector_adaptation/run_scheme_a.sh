#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet

RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
OUT="${OUT:-$RUN/scheme_a_top256_$(date +%Y%m%d_%H%M%S)}"
PROBE_OUT="${PROBE_OUT:-$OUT/intervention_probe}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"

if [ -e "$OUT" ] || [ -e "$PROBE_OUT" ]; then
  printf 'Output already exists: %s or %s\n' "$OUT" "$PROBE_OUT" >&2
  exit 1
fi

export RUN OUT PY
export STAGE0_ONLY=0 AUDIT_POOL=top_256 CANDIDATE_FEATURES=1 ABLATION_ARM=F
export ABLATION_FRAMES="${ABLATION_FRAMES:-90}"

"$PY" -m unittest \
  local_fusion_detector_adaptation.test_candidate_audit \
  local_fusion_detector_adaptation.test_candidate_intervention_probe

bash local_fusion_detector_adaptation/run_candidate_audit.sh

set -- "$PY" -u -m local_fusion_detector_adaptation.candidate_intervention_probe \
  --input-root "$OUT" --output-dir "$PROBE_OUT" --arm F
if [ -n "${SCENE_MAP:-}" ]; then
  set -- "$@" --group-map "$SCENE_MAP"
fi
"$@"

printf 'Scheme A outputs: %s\n' "$PROBE_OUT"

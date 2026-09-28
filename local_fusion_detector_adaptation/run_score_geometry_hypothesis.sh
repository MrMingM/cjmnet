#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH="/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main${PYTHONPATH:+:$PYTHONPATH}"

RUN="${RUN:-/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227}"
PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
HYP_OUT="${HYP_OUT:-$RUN/top256_score_geometry_$(date +%Y%m%d_%H%M%S)}"
ARM="${ARM:-F}"

if [ "$ARM" != F ] && [ "$ARM" != 'F+D' ]; then
  printf 'ARM must be F or F+D\n' >&2
  exit 1
fi
if [ ! -x "$PY" ] || [ ! -f "$RUN/protocol.json" ]; then
  printf 'Missing Python or pilot protocol\n' >&2
  exit 1
fi
if [ "${CHECK_ONLY:-0}" = 1 ]; then
  "$PY" -c 'import local_fusion_detector_adaptation.candidate_discriminability; import local_fusion_detector_adaptation.candidate_intervention_probe; import local_fusion_detector_adaptation.candidate_hypothesis_replay'
  if [ -n "${EXISTING_AUDIT:-}" ]; then
    test -f "$EXISTING_AUDIT/candidate_audit.json"
  else
    RUN="$RUN" PY="$PY" CHECK_ONLY=1 STAGE0_ONLY=0 AUDIT_POOL=top_256 \
      sh local_fusion_detector_adaptation/run_candidate_audit.sh
  fi
  printf 'Preflight passed; no experiment was run.\n'
  exit 0
fi
if [ -e "$HYP_OUT" ]; then
  printf 'Output already exists: %s\n' "$HYP_OUT" >&2
  exit 1
fi
mkdir -p "$HYP_OUT"

if [ -n "${EXISTING_AUDIT:-}" ]; then
  DATA_ROOT="$EXISTING_AUDIT"
  if [ ! -f "$DATA_ROOT/candidate_audit.json" ]; then
    printf 'EXISTING_AUDIT has no candidate_audit.json: %s\n' "$DATA_ROOT" >&2
    exit 1
  fi
else
  DATA_ROOT="$HYP_OUT/extraction"
  FRAMES="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["validation_indices"]))' "$RUN/protocol.json")"
  printf 'Extracting %s frames per weather; arm=%s\n' "$FRAMES" "$ARM"
  RUN="$RUN" PY="$PY" OUT="$DATA_ROOT" STAGE0_ONLY=0 \
    AUDIT_POOL=top_256 ABLATION_FRAMES="$FRAMES" ABLATION_ARM="$ARM" \
    CANDIDATE_FEATURES=1 \
    sh local_fusion_detector_adaptation/run_candidate_audit.sh
fi

"$PY" -u -m local_fusion_detector_adaptation.candidate_hypothesis_replay \
  --input-root "$DATA_ROOT" --output-dir "$HYP_OUT/replay" --arm "$ARM"

"$PY" -u -m local_fusion_detector_adaptation.candidate_discriminability \
  --input-root "$DATA_ROOT" --output-dir "$HYP_OUT/passive"

"$PY" -u -m local_fusion_detector_adaptation.candidate_intervention_probe \
  --input-root "$DATA_ROOT" --output-dir "$HYP_OUT/intervention" --arm "$ARM"

printf 'SCORE-GEOMETRY HYPOTHESIS AUDIT COMPLETE: %s\n' "$HYP_OUT"

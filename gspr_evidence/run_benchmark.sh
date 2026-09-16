#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
PY=/home/cjm/miniconda3/envs/opencood/bin/python
RUN="${RUN:-/data/cjm/datasets/logs/gspr_evidence_seed20260913_20260913_094932}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
OUT="${RUN}_historical_test_$(date +%Y%m%d_%H%M%S)"
case "$OUT" in /data/cjm/datasets/logs/*) ;; *) exit 1 ;; esac
mkdir "$OUT"
exec > >(tee -a "$OUT/console.log") 2>&1
echo "Existing checkpoints: $RUN"
echo "Historical test results: $OUT"
"$PY" -m unittest gspr_evidence.test_benchmark -v
for variant in matching concat no_u; do
  for weather in clean fog rain snow; do
    "$PY" -m gspr_evidence.evaluate --evaluation-protocol benchmark \
      --config "$RUN/$variant/experiment.yaml" \
      --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
      --heads "$RUN/$variant/gain_best.pth" --weather "$weather" \
      --modes none a0b0 protocol learned full \
      --output-dir "$OUT/eval_${variant}_${weather}"
  done
done
"$PY" -m gspr_evidence.summarize --run "$OUT"
echo "Send back: $OUT/results.md"

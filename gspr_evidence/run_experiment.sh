#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260913
PY=/home/cjm/miniconda3/envs/opencood/bin/python
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
RUN="${RUN:-/data/cjm/datasets/logs/gspr_evidence_seed20260913_$(date +%Y%m%d_%H%M%S)}"
case "$RUN" in
  /data/cjm/datasets/logs/*) ;;
  *) echo 'RUN must be a new directory under /data/cjm/datasets/logs' >&2; exit 1 ;;
esac
if [[ -e "$RUN" ]]; then
  echo "New run required: $RUN already exists. See README for stage-specific resume." >&2
  exit 1
fi
mkdir -p "$RUN"
exec > >(tee -a "$RUN/console.log") 2>&1
printf 'RUN=%s\n' "$RUN"
common=(--frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT")
"$PY" -m unittest gspr_evidence.test_evidence gspr_communication.test_codec gspr_communication.test_tensors -v
"$PY" -m gspr_evidence.verify --config gspr_evidence/experiment.yaml "${common[@]}"
for split in train validation; do
  "$PY" -m gspr_evidence.prepare --config gspr_evidence/experiment.yaml "${common[@]}" \
    --split "$split" --output-dir "$RUN/cache_$split"
done
for variant in matching concat no_u; do
  "$PY" -m gspr_evidence.train --train-cache "$RUN/cache_train" --validation-cache "$RUN/cache_validation" \
    --variant "$variant" --output-dir "$RUN/$variant"
  for weather in clean fog rain snow; do
    modes=(--modes learned)
    if [[ "$variant" == matching ]]; then
      modes=(--modes none a0b0 protocol learned full)
    fi
    "$PY" -m gspr_evidence.evaluate --evaluation-protocol development --config "$RUN/$variant/experiment.yaml" "${common[@]}" \
      --heads "$RUN/$variant/gain_best.pth" --weather "$weather" "${modes[@]}" \
      --output-dir "$RUN/eval_${variant}_${weather}"
  done
done
"$PY" -m gspr_evidence.summarize --run "$RUN"
printf 'Send back: %s/results.md\n' "$RUN"

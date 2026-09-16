#!/usr/bin/env bash
set -euo pipefail
cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260916
PY=/home/cjm/miniconda3/envs/opencood/bin/python
FRONTEND_CONFIG="${FRONTEND_CONFIG:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth}"
OUT="${OUT:-/data/cjm/datasets/logs/ceif_min_$(date +%Y%m%d_%H%M%S)}"
extra=()
if [[ "${RESUME:-0}" == 1 ]]; then
  test -d "$OUT"; extra+=(--resume)
else
  if [[ -e "$OUT" ]]; then echo "Output already exists: $OUT" >&2; exit 1; fi
fi
echo "TRAIN_DIR=$OUT"
"$PY" -m unittest ceif_min.test_min ceif_min.test_training -v
"$PY" -m ceif_min.train --frontend-config "$FRONTEND_CONFIG" --frontend-checkpoint "$FRONTEND_CHECKPOINT" \
  --output-dir "$OUT" --epochs "${EPOCHS:-3}" --decoder-epochs "${DECODER_EPOCHS:-1}" \
  --queries "${QUERIES:-1024}" --smoke "${SMOKE:-0}" "${extra[@]}"
"$PY" -m ceif_min.evaluate --run "$OUT" --phase development \
  --output-dir "${OUT}_validation_$(date +%Y%m%d_%H%M%S)"
if [[ "${SMOKE:-0}" == 0 && "${TEST_AFTER_TRAIN:-1}" == 1 ]]; then
  "$PY" -m ceif_min.evaluate --run "$OUT" --phase benchmark \
    --output-dir "${OUT}_opv2vw_test_$(date +%Y%m%d_%H%M%S)"
else
  echo "Development complete. Standalone test: RUN=$OUT bash ceif_min/run_test.sh"
fi

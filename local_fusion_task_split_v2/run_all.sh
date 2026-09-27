#!/bin/sh
set -eu

cd /home/cjm/OpenCOOD-main/cjmnet
export PYTHONPATH=/home/cjm/OpenCOOD-main/cjmnet:/home/cjm/OpenCOOD-main
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export PYTHONHASHSEED=20260927

PY="${PY:-/home/cjm/miniconda3/envs/opencood/bin/python}"
FRONTEND_ROOT="${FRONTEND_ROOT:-/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155}"
FRONTEND_CONFIG="${FRONTEND_CONFIG:-$FRONTEND_ROOT/config.yaml}"
FRONTEND_CHECKPOINT="${FRONTEND_CHECKPOINT:-$FRONTEND_ROOT/net_best_validation.pth}"
V3_RUN="${V3_RUN:-/data/cjm/datasets/logs/local_fusion_v3_20260922_182832}"
B0_RUN="${B0_RUN:-/data/cjm/datasets/logs/task_split_pilot_20260927_120637}"
RUN="${RUN:-/data/cjm/datasets/logs/task_split_v2_$(date +%Y%m%d_%H%M%S)}"
export RUN

test -f "$FRONTEND_CONFIG"
test -f "$FRONTEND_CHECKPOINT"
test -f "$V3_RUN/residual/best.pth"
test -f "$B0_RUN/protocol.json"
test -f "$B0_RUN/decision_results.json"
test -f "$B0_RUN/Split.pth"
test -f local_fusion_task_split_v2/experiment.yaml

for root in   /data/scd/datasets/opv2v_official_data_dumping/test   /data/cjm/datasets/opv2v-w/fog/test   /data/cjm/datasets/opv2v-w/rain/test   /data/cjm/datasets/opv2v-w/snow/test
do
  test -d "$root"
done

echo "===== B1 stage 0: unit tests ====="
"$PY" -u -m local_fusion_task_split_v2.test_core

echo "===== B1 stage 1: train on OPV2V train (clean + online weather branches) ====="
echo "===== B1 stage 2: development validation on OPV2V validation ====="
"$PY" -u -m local_fusion_task_split_v2.pipeline   --config local_fusion_task_split_v2/experiment.yaml   --v3-config local_fusion_v3/experiment.yaml   --frontend-config "$FRONTEND_CONFIG"   --frontend-checkpoint "$FRONTEND_CHECKPOINT"   --v3-checkpoint "$V3_RUN/residual/best.pth"   --b0-run "$B0_RUN"   --run "$RUN"

EXPAND=$("$PY" -c 'import json,sys; print("1" if json.load(open(sys.argv[1], encoding="utf-8"))["decision"]["expand"] else "0")' "$RUN/decision_results.json")

if [ "$EXPAND" != "1" ]; then
  echo "===== B1 development gate failed ====="
  echo "Formal OPV2V/OPV2V-W test is intentionally skipped to protect the held-out test protocol."
  echo "Development result: $RUN/decision_results.json"
  exit 0
fi

echo "===== B1 stage 3: fixed-checkpoint formal benchmark ====="
echo "No retraining and no online weather augmentation are allowed in this stage."
"$PY" -u -m local_fusion_task_split_v2.benchmark   --config local_fusion_task_split_v2/experiment.yaml   --v3-config local_fusion_v3/experiment.yaml   --frontend-config "$FRONTEND_CONFIG"   --frontend-checkpoint "$FRONTEND_CHECKPOINT"   --v3-checkpoint "$V3_RUN/residual/best.pth"   --b0-run "$B0_RUN"   --run "$RUN"   --output "$RUN/benchmark"

echo "===== B1 full pipeline complete ====="
echo "Development: $RUN/decision_results.json"
echo "Formal benchmark: $RUN/benchmark/benchmark_summary.json"

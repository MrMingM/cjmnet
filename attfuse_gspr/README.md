# GSPR-AttFuse v1

This implementation inserts a Geometry-first Sensor-adaptive Pillar
Reliability Network between spconv voxel packing and PointPillar feature
encoding. It does not add temporal modeling or a cooperative reliability
branch.

## Why this boundary is used

OpenCOOD performs `SpVoxelPreprocessor` in CPU DataLoader workers. Running a
trainable denoiser before it would sever autograd. GSPR therefore consumes the
still-raw XYZI point slots in `voxel_features [M,T,4]` before PillarVFE creates
point embeddings or BEV features. No point is deleted and no coordinate is
changed, so `voxel_coords` and slot indices remain valid.

The `(9999,9999,9999)` sentinel approach is intentionally not used in the
training path. It can contaminate KNN/range/sensor statistics and turns soft
reliability into irreversible filtering. A thresholded export can be added as
an inference-only ablation after reliability calibration.

## Components

- Geometry branch: normalized xyz/range, elevation (ring proxy), pillar
  density, cluster offsets and pillar-center offsets.
- Pillar frequency branch: one-level low/high Haar-like decomposition on a
  coarse PointPillar-aligned XY grid. This replaces TripleMixer's three dense
  planes and its separate KDTree/projection pipeline.
- Radiometric branch: robust per-agent intensity normalization plus a
  sensor-statistics FiLM adapter. If a future loader supplies ring as feature
  5, set `ring_index: 4`; OPV2V XYZI uses elevation as the default proxy.
- Physics branch: range/intensity/density residual evidence.
- Fusion: non-negative two-class evidence produces point reliability and
  evidential uncertainty.
- Soft PointPillar: decorated point features are multiplied by reliability
  before PFN pooling. Point count and coordinates never change.

Outputs expose `point_reliability`, `point_uncertainty`, `pillar_reliability`
and `pillar_uncertainty` for future communication/fusion modules, but v1 does
not consume them after local PillarVFE.

## Server smoke test

Run from `/home/cjm/OpenCOOD-main/cjmnet`:

```bash
ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/cjm/OpenCOOD-main:/home/cjm/OpenCOOD-main/cjmnet \
/home/cjm/miniconda3/envs/opencood/bin/python verify_gspr.py
```

## Detection warm-start training

This command initializes every matching AttFuse layer from the selected clean
baseline checkpoint. New GSPR parameters start nearly transparent. The exact
checkpoint is loaded directly; it is not selected by directory scanning.

```bash
OPENCOOD_ROOT=/home/cjm/OpenCOOD-main
GSPR_DIR=/home/cjm/OpenCOOD-main/cjmnet
PYTHON=/home/cjm/miniconda3/envs/opencood/bin/python
BASELINE=/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth
RUN_DIR=/data/cjm/datasets/logs/gspr_attfuse_opv2v_$(date +%Y%m%d_%H%M%S)

mkdir -p "$RUN_DIR"
cd "$OPENCOOD_ROOT"

ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH="$OPENCOOD_ROOT:$GSPR_DIR" \
nohup "$PYTHON" "$GSPR_DIR/train_gspr_attfuse.py" \
  --config "$GSPR_DIR/gspr_attfuse_config.yaml" \
  --run-dir "$RUN_DIR" \
  --baseline-checkpoint "$BASELINE" \
  > "$RUN_DIR/train.log" 2>&1 &

echo $! > "$RUN_DIR/train.pid"
echo "RUN_DIR=$RUN_DIR"
echo "PID=$(cat "$RUN_DIR/train.pid")"
```

All run artifacts are forced under `/data/cjm/datasets`. The checked-in config
uses clean OPV2V train/validate. That is suitable for validating integration,
but it is not yet evidence that the reliability head learned weather noise.
Point-supervised training requires simulator labels to be carried through
voxelization as `point_reliability_target`; the model and loss already accept
that aligned tensor.

## Evaluation

OpenCOOD reads `config.yaml` from the model directory and loads the highest
`net_epoch*.pth`. For each frozen checkpoint, copy it and the config into a
separate evaluation directory, edit only `validate_dir`, then run:

```bash
ROCR_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/cjm/OpenCOOD-main:/home/cjm/OpenCOOD-main/cjmnet \
nohup /home/cjm/miniconda3/envs/opencood/bin/python \
  /home/cjm/OpenCOOD-main/cjmnet/inference_gspr_attfuse.py \
  --model_dir "$EVAL_DIR" \
  --fusion_method intermediate \
  > "$EVAL_DIR/inference.log" 2>&1 &
echo $! > "$EVAL_DIR/inference.pid"
```

Use clean test `/data/scd/datasets/opv2v_official_data_dumping/test` and the
three frozen OPV2V-W test paths under `/data/cjm/datasets/opv2v-w`. Never use
OPV2V-W AP to select a checkpoint.

## Reliability mechanism audit

After selecting a checkpoint on clean validation, audit the same frozen model
on all four test conditions. The audit reads only valid points retained in
voxel slots after range filtering and per-pillar truncation; it does not alter
the dataset or checkpoint.

```bash
EVAL_ROOT=/data/cjm/datasets/logs/gspr_attfuse_opv2v_20260904_154404/eval_best_cleanval_epoch11

cd /home/cjm/OpenCOOD-main
ROCR_VISIBLE_DEVICES=0 \
nohup bash /home/cjm/OpenCOOD-main/cjmnet/run_gspr_reliability_audit.sh \
  "$EVAL_ROOT" \
  > "$EVAL_ROOT/reliability_audit_all.log" 2>&1 &
echo $! > "$EVAL_ROOT/reliability_audit.pid"
```

The loop is intentionally sequential on one HCU. It reports reliability and
uncertainty distributions, threshold fractions, correlations with raw range,
intensity and pillar density, binned trends, fused evidence, and positive
branch-support diagnostics. Branch support is not an additive decomposition of
fused evidence because the trained v1 model applies softplus after summing the
branch logits and learned prior.

## External source-code decision

No additional paper repository is required for v1. TripleMixer supplies the
frequency-modeling and weather-simulation reference, while OpenCOOD supplies
the actual voxel/Pillar interface. Pulling LiSnowNet or 3D-OutDet into this
implementation now would add a second projection/preprocessing stack without
solving the OPV2V alignment problem. They remain useful later as independently
reproduced efficiency/denoising baselines, not as dependencies of GSPR.

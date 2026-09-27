# Local Fusion Task Split V2 (B1)

B1 starts from the completed B0 Split checkpoint and asks one narrow question:

> Is it useful to keep classification/regression source routing shared by
> default, and recover B0 task-specific routing only in local task-conflict
> regions?

## Frozen endpoints

B0 provides two trained source distributions:

- `w_cls_raw`: classification router weights;
- `w_reg_raw`: regression router weights.

Their common endpoint is:

```
w_common = 0.5 * (w_cls_raw + w_reg_raw)
```

B1 never retrains either B0 router.  A new gate `g in [0,1]` interpolates:

```
w_cls = w_common + g * (w_cls_raw - w_common)
w_reg = w_common + g * (w_reg_raw - w_common)
```

Thus `g=0` exactly reproduces B0 Split-Collapse and `g=1` exactly
reproduces B0 Split.

## Gate inputs

Only inference-visible signals are used, per B0 scale 0/1:

1. source-weight total variation:
   `0.5 * sum_source(abs(w_cls_raw-w_reg_raw))`;
2. whether classification/regression choose different top-weight sources;
3. frozen Collapse classification activity, resized to the scale.

No GT box, GT IoU, weather label, task-gap audit cohort, or hindsight outcome
enters the gate.

## Local versus Global

Both variants use the same two per-scale networks and exactly the same
parameter count:

```
3 channels -> 3x3 conv (8) -> SiLU -> 1x1 conv -> sigmoid
```

The final layer starts at weight 0 and bias -4, so initial
`g = sigmoid(-4) ~= 0.018`, close to Collapse.

- **Local-Gate** applies the network to the spatial signal maps.
- **Global-Gate** first averages the same three channels over the full map,
  broadcasts those pooled inputs back over the scale, and applies the identical
  network. Replicate padding keeps the result spatially constant while letting
  the full 3x3 kernel participate.

The comparison therefore isolates local conflict recognition from merely
learning a frame-level Split/Collapse interpolation.

## Training

Frozen:

- GSPR and per-agent encoding;
- both trained B0 Split routers;
- deblocks;
- `cls_head` and `reg_head`.

Trainable: only the B1 Gate.

Loss:

```
PointPillarLoss + 0.01 * mean(g)
```

The small gate penalty makes Collapse the default unless task loss supports
opening Split.

The exact B0 train and validation indices are reused for development. Training uses OPV2V train with paired clean and online-simulated-weather branches. Development validation uses OPV2V validation with clean plus online fog/rain/snow. If and only if the predeclared development gate passes, the same fixed last-epoch checkpoints are then evaluated once on OPV2V clean test and OPV2V-W fog/rain/snow test. OPV2V-W is never used for training, tuning, threshold selection, epoch selection, or architecture selection.

## Reported methods

- `B0-Collapse`
- `B0-Split`
- `Global-Gate`
- `Local-Gate`

B0 endpoints are reproduced from the frozen Split checkpoint; the pipeline
aborts if their AP differs from the saved B0 result by more than 1e-6.

Gate diagnostics include mean/P50/P75/P90 by scale, correlation with total
variation/activity, and mean gate for top-source-agree versus disagree cells.

## Feasibility gate

Primary comparison: `Local-Gate vs Global-Gate`.

Expansion requires all of:

- mean Fog/Rain/Snow AP70 gain >= 0.005;
- at least 2/3 weather AP70 gains positive;
- Local three-weather mean AP70 strictly above frozen B0 Split;
- Clean AP70 no more than 0.001 below Global;
- AP50 no more than 0.001 below Global in every condition;
- Local-vs-Global lost TP and new FP each <= 1% of Global TP count.

## Run

Default B0 checkpoint run:

`/data/cjm/datasets/logs/task_split_pilot_20260927_120637`

Run with:

```sh
sh local_fusion_task_split_v2/run_all.sh
```

Final result:

`$RUN/decision_results.json`


## End-to-end execution protocol

`run_all.sh` now executes the full scientific workflow in one command:

1. **Unit tests**
   - Verify B1 interpolation endpoints and Local/Global invariants.

2. **Training on OPV2V train**
   - Reuse the exact B0 training indices.
   - For each training frame, optimize the Gate on both:
     - the clean branch;
     - the online physics-weather branch already defined by the existing training config.
   - B0 routers, GSPR, encoders and detector heads remain frozen.
   - The saved B1 checkpoint is the fixed last epoch.

3. **Development validation on OPV2V validation**
   - Reuse the exact B0 validation indices.
   - Evaluate clean plus online fog/rain/snow.
   - Apply the predeclared B1 feasibility gate in `decision_results.json`.

4. **Formal held-out benchmark — only if development passes**
   - Clean: `/data/scd/datasets/opv2v_official_data_dumping/test`
   - Fog: `/data/cjm/datasets/opv2v-w/fog/test`
   - Rain: `/data/cjm/datasets/opv2v-w/rain/test`
   - Snow: `/data/cjm/datasets/opv2v-w/snow/test`
   - Use every frame in dataset order.
   - Disable online weather augmentation and data augmentation.
   - Always read `processed_lidar`; OPV2V-W files are already degraded.
   - Use the historical OpenCOOD non-global AP protocol.
   - No retraining or model selection is allowed after seeing test results.

If the development gate fails, `run_all.sh` intentionally stops before the held-out benchmark. This prevents OPV2V-W from becoming a second validation set.

Formal outputs, when reached:

```
$RUN/benchmark/benchmark_results.json
$RUN/benchmark/benchmark_summary.json
```

The formal benchmark reports the same four fixed methods:

- `B0-Collapse`
- `B0-Split`
- `Global-Gate`
- `Local-Gate`

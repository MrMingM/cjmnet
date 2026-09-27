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
  applies the identical network, and broadcasts one gate value over that
  scale/frame.

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

The exact B0 train and validation indices are reused.  This B1 pilot accesses
official validation with online simulated weather only and does not access
OPV2V-W.

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

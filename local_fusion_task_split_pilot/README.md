# Local Fusion Task-Split Pilot

This pilot tests one narrow hypothesis: classification and regression may
benefit from different CAV source mixtures after the frozen per-agent encoder.

## Arms

- **Shared**: two full v3 routers are trained, their source weights are averaged,
  and the same averaged weights are used by both classification and regression.
- **Split**: the same two routers are trained, but router A drives
  classification and router B drives regression.
- **Split-Collapse**: no retraining; the trained Split weights are averaged at
  inference to test whether keeping task-specific weights matters.
- **Split-Swap**: no retraining; the two trained task weights are exchanged.
  This is diagnostic only.

Both trainable arms have the same router parameter count. GSPR, per-agent
encoding, deblocks, cls_head and reg_head stay frozen. Both arms execute two
source-routing paths and two detector decoding paths. The only intended
functional difference is whether the two task weights are forced to be shared.

The exact train and validation frame indices are read from the completed
F-versus-F+D reference pilot. This experiment uses official validation with
online simulated weather only; it does not access OPV2V-W.

## Primary gate

Split is compared directly with Shared. Expansion requires:

- mean Fog/Rain/Snow AP70 gain >= 0.005;
- at least two weather conditions improve;
- Clean AP70 no more than 0.001 below Shared and v3_start;
- AP50 no more than 0.001 below Shared in every condition;
- lost original TP and new FP each <= 1% of Shared TP count.

Collapse and Swap are reported separately for mechanism interpretation and do
not change the predeclared primary feasibility gate.

## Run

Use `run_all.sh`. The default reference run is:

`/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227`

The final result is:

`$RUN/decision_results.json`

## B0 task-gap benefit audit

After a completed B0 run, `audit_task_gap.py` replays the exact saved validation
indices without retraining. It compares Split with Split-Collapse (primary) and
Shared (secondary) at the GT-target level.

Each GT target is classified as:

- `split_only`: detected by Split but not the reference;
- `reference_only`: detected by the reference but not Split;
- `both`;
- `neither`.

Within the exact oriented GT footprint and a 1.5x context footprint, the audit
records:

- local mean absolute classification/regression source-weight gap;
- source-weight total variation, `0.5 * sum(|w_cls-w_reg|)`;
- fraction of cells where classification and regression choose different
  top-weight sources.

The report includes category quantiles, pairwise AUROC diagnostics and
top-quartile enrichment. These are post-hoc GT-localized associations only:
they do not define a deployment threshold and do not prove causality.

Run:

`sh local_fusion_task_split_pilot/run_task_gap_audit.sh`

Default completed B0 run:

`/data/cjm/datasets/logs/task_split_pilot_20260927_120637`

Final report:

`$RUN/task_gap_audit/task_gap_audit.json`

## After the direction-B death test: action and region audits

The original GT-aware Oracle passed its preset ceiling gate. These two checks
ask how much of that ceiling remains when target regions come from predictions.
They do not train a detector or selector. Run them on the remote server.

`audit_oracle_actions.py` reads the original `protocol.json`,
`death_test_results.json`, and four weather `frames.jsonl` files. It counts
action changes, task-separated choices, and immediate focal recoveries. Same
and Task each make cumulative changes along separate paths, so these counts
are **path diagnostics**, not same-state causal gains or final TP identities.

`proposal_constrained_oracle.py` builds a Shared fused-score top256 pool from
geometry-valid decoded boxes before reading GT, then keeps the in-range boxes.
For each GT, the Oracle may use only its best-overlapping proposal box as its output-space ROI
(minimum BEV IoU 0.10), at the original 1.0x/1.5x expansions. GT still picks
the matching proposal and the best source action, so this remains a research
ceiling. In the Task path, Same and Task actions are evaluated from the **same
current state**, and the per-target record flags a genuine same-state Task win.
The separate full Same and Task paths supply frame-order AP30/AP50/AP70 and
final GT match identities. The final report compares the incremental Task
gain with the original GT-region Oracle; it does not claim deployable AP.

Default original Oracle run:

`/data/cjm/datasets/logs/task_source_oracle_v2_20260928_113748`

After syncing this code to `/home/cjm/OpenCOOD-main/cjmnet`, start the long
server run in the background:

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
OUT="/data/cjm/datasets/logs/proposal_task_oracle_$(date +%Y%m%d_%H%M%S)"
nohup env OUT="$OUT" sh local_fusion_task_split_pilot/run_proposal_constrained_oracle.sh > "$OUT.log" 2>&1 &
echo "$OUT"
```

Outputs: `$OUT.action_audit.json`,
`$OUT/protocol.json`, `$OUT/proposal_oracle_results.json`, and
`$OUT/{clean,fog,rain,snow}/frames.jsonl` plus condition summaries. The script
will refuse to overwrite an existing output directory or action audit.

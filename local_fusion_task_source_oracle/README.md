# Direction B Target-wise Task/Source Oracle Death Test

This is the final no-training viability test for direction B.

The test intentionally gives direction B GT and post-NMS hindsight. It asks a
stronger question than B0/B1:

> If an Oracle may choose the best collaborative source representation for
> every GT target, and classification/regression may choose independently,
> is there enough AP70 headroom to justify a publishable method?

## Baseline

The control is the completed trained B0 `Shared` arm. Its AP30/AP50/AP70 must
reproduce the saved B0 result within 1e-6 before Oracle results are accepted.

## Candidate source pool

The pool is computed once per frame from frozen tensors and contains:

- `KEEP_SHARED`;
- `single:i`: aligned source/CAV `i` alone through the frozen detector;
- `query:i`: attention fusion with source/CAV `i` as the query while all
  received source features remain available.

Including both single-source and query-conditioned collaborative candidates is
deliberately generous: the death test should not reject direction B merely
because one source parameterization is too weak.

## Oracle-Same

For every GT target, the Oracle may keep Shared or replace a local output ROI
with one candidate, but classification and regression must use the same
candidate.

## Oracle-Task

The Oracle sees the same candidate pool and ROI choices, but classification
and regression may independently choose different candidates. Therefore it
contains the Same-Source action family plus genuine task-separated actions.

ROI expansion is also Oracle-selected from `1.0x` and `1.5x` per target.

Target selection uses actual GT/post-NMS outcomes and starts from GTs missed by
Shared. KEEP_SHARED is always legal. The greedy objective first maximizes
matched GT count, then preserves already matched GTs, reduces new FP harm,
recovers the focal target, and improves focal score/IoU. This is a very strong
diagnostic, but it is still not a mathematical optimum over arbitrary
continuous source weights or every possible global action ordering.

## Pre-registered life/death bar

Direction B survives only if all are true on Fog/Rain/Snow development
validation:

1. Oracle-Task mean AP70 gain over Shared >= **+1.5 percentage points**;
2. at least **2/3** weather conditions gain >= **+1.0 point**;
3. Oracle-Task mean AP70 gain over Oracle-Same >= **+0.5 point**.

The third condition is essential. If Same-Source selection explains nearly all
of the Oracle gain, the useful direction is target-level source/agent
selection, not classification/localization task separation.

If any condition fails, the current direction B is marked for termination
under the user's publication bar. The thresholds must not be changed after
seeing results.

## Scope

The experiment reuses the exact B0 validation indices: clean plus online
fog/rain/snow. No OPV2V-W held-out test is accessed. No model is trained.

Run:

```sh
sh local_fusion_task_source_oracle/run_all.sh
```

Default B0:

`/data/cjm/datasets/logs/task_split_pilot_20260927_120637`

Final output:

`$RUN/death_test_results.json`

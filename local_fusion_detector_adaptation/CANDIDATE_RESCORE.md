# Frozen candidate rescore control

This is a read-only validation experiment on the completed F/F+D pilot. It
uses the same 90 saved frames per condition and the Stage-0 `top_256` pool.
There is no training, weather-specific calibration, GT in scoring, or
OPV2V-W test access. The source-only predictions use the frozen frontend
detector; they are evidence proxies, not causal source attribution.

For each fused candidate and source at the **same anchor**, compute that
source's standalone classification score and the planar BEV IoU between its
decoded box and the fused candidate box. The simple score is the maximum
across sources of `source score × box agreement`. Compare it with the fused
score, ego-only score, and maximum source score. Run these four scores on both
the original `score > 0.2` geometry-valid pool and the fixed top-256 pool.
All use the original rotated NMS threshold and final range check. No new
score threshold is fitted or applied. Original scorepass output must match
the repository postprocessor box for box, and original AP must match the
saved pilot before a report is accepted.

Primary results use the repository's original frame-order AP protocol.
Globally sorted AP is reported separately as a diagnostic. A candidate-level
AUC improvement from the earlier audit does not imply an AP improvement;
this experiment tests that gap. The source-agreement formula is a simple
control, not a publishable novelty claim.

The continuation screen is fixed before running: for the F arm on top-256,
source-agreement must beat the better of original and top-256 max-source AP70
by at least 0.5 percentage points on average across Fog/Rain/Snow, be positive
in at least two of those weathers, lose at most 0.1 point of Clean AP70, and
lose at most 0.1 point of AP50 in every condition relative to original F.
Passing only warrants another investigation; failing stops this simple
control from becoming a main-method candidate.

The launcher defaults to the exact completed pilot directory recorded in
`AI_CONTEXT.md` section 25. Check inputs without starting the long job:

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_rescore.sh
```

Run the experiment in the background with a unique output directory:

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227
OUT="$RUN/candidate_rescore_$(date +%Y%m%d_%H%M%S)"
nohup env RUN="$RUN" OUT="$OUT" \
  sh local_fusion_detector_adaptation/run_candidate_rescore.sh \
  > "$OUT.log" 2>&1 < /dev/null &
echo "PID=$! OUT=$OUT LOG=$OUT.log"
```

The launcher first runs `test_candidate_rescore` in the server environment.
The final report is `candidate_rescore.json` under `OUT`. Each weather also
has a `summary.json` and per-frame output counts. If scorepass reproduction
fails, the program stops instead of reporting a potentially misleading AP.

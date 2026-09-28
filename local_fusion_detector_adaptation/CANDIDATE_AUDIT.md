# Candidate pool and source audit

This is a read-only development diagnostic on the completed F/F+D pilot's
saved validation indices. It never trains, edits checkpoints, or evaluates
OPV2V-W test. Run the pool sweep first. Then choose a pool **before** reading
the candidate-level GT labels and run the source audit on that fixed pool.

The sweep compares the original score > 0.2 pool with score > 0.1, > 0.05,
top 128/256/512, score > 0.05 plus top 256, and all geometry-valid decoded
boxes. Top-K is applied after geometry filtering. Counts and IoU >= 0.7 GT
coverage are reported both before and after range filtering; these are
pre-NMS candidate statistics, not AP or deployable quality estimates. The
original-pool AP is checked against the saved pilot in every weather.
The fixed Stage 0 screen requires p90 <= 256 candidates per frame and at
least 90% of the all-geometry IoU70 GT coverage after range filtering in
every weather for both F and F+D. A passing pool is a feasible pool for the
next diagnostic, not evidence that a quality model will succeed.

The second run saves one JSON row per candidate in `candidate_rows.jsonl`.
`source_scores_same_anchor` and `source_box_iou_same_anchor` are from frozen
single-source frontend heads. They are **proxies**, not the classification
and localization contributions to the fused prediction. On evenly spaced
validation frames, `leave_one_source_out` gives the change from removing each
peer and rerunning the learned fusion and detector. Removal renormalizes the
remaining sources, so these conditional effects cannot be summed into a
source attribution. `gt_quality_change` and every `max_gt_iou` field use GT
only after prediction, for diagnosis. The F/F+D fusion-by-detector 2x2 swap
reports AP to separate fusion changes from downstream detector changes.

`quality_band=good` means maximum BEV GT IoU >= 0.7; `bad` means < 0.5.
`high_score_bad` additionally requires fused score >= 0.5. `final_duplicate_fp`
marks a high-quality candidate that survives to the final predictions but
gets no unique GT assignment. A high-quality pre-NMS duplicate that is later
suppressed is still labeled `good`; it is not misreported as a bad box.
`within_range` is reported separately so candidates outside the benchmark
range do not silently mix with usable candidates in the grouped summary.

On the server, start the Stage 0 job in the background. The launcher defaults
to the completed pilot recorded in `AI_CONTEXT.md` section 25:
`/data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227`.
Before starting the long job, the updated launcher can check every required
input without running inference:

```sh
CHECK_ONLY=1 sh local_fusion_detector_adaptation/run_candidate_audit.sh
```

It prints the missing path and exits if an input is absent.

```sh
nohup sh local_fusion_detector_adaptation/run_candidate_audit.sh \
  > /data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/candidate_stage0.log \
  2>&1 < /dev/null &
```

After inspecting `candidate_stage0/candidate_audit.json`, select a fixed pool
name from `candidate_pool_specs` other than the diagnostic `all_geometry`
reference. For example:

```sh
nohup env STAGE0_ONLY=0 AUDIT_POOL=top_256 \
  sh local_fusion_detector_adaptation/run_candidate_audit.sh \
  > /data/cjm/datasets/logs/fusion_detector_adaptation_20260927_103227/candidate_source_top_256.log \
  2>&1 < /dev/null &
```

Outputs include a summary per weather, per-arm pool counts and coverage,
AP for all 2x2 swaps in the second run, grouped candidate counts, and the
individual JSONL rows. The saved validation set is exploratory; a positive
proxy association alone does not establish a new method or generalization.

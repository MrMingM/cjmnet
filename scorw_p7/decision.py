"""Conservative preregistered project triage, not a detector result or proof of no effect."""
from scorw_p7.signal_metrics import METRICS


def decide(analysis, scenes, selection, config, smoke=False):
    d = config["decision"]
    val = analysis["partitions"]["independent_validation"]
    chosen = [s for s in scenes if s["partition"] == "independent_validation"]
    base = dict(detector_started=False,physical_realism_proven=False,thresholds=d,
        scope="fixed pool, sampled Clean-return-conditioned attenuation proxy; no dataset novelty or detection-success claim",
        next_action="archive this version; no automatic expansion, tuning, detector execution or new seeds")
    if smoke:
        return dict(base,decision="INCONCLUSIVE",reason="SMOKE is an identity/runtime check only")
    if selection["validation_status"] == "INCONCLUSIVE_VALIDATION_UNAVAILABLE":
        return dict(base,decision="INCONCLUSIVE_VALIDATION_UNAVAILABLE",reason="Fewer than8 unused non-future scenes; development retained, validation not executed",
                    project_recommendation="do not expand this version into a full weather dataset")
    quality = len(chosen) == d["required_validation_scenes"]
    checks = {}
    for s in chosen:
        c = s["counts"]
        q = (c["valid_target_instances"] >= d["minimum_valid_targets_per_scene"]
            and c["valid_unique_objects"] >= d["minimum_unique_objects_per_scene"]
            and c["valid_target_cav_pairs"] >= d["minimum_valid_pairs_per_scene"]
            and c["invalid_pair_fraction"] <= d["maximum_invalid_pair_fraction"]
            and (c["maximum_object_pair_share"] or 0) <= d["maximum_target_pair_share"]
            and s["balance"]["status"] == "passed" and s["negative_control"]["status"] == "passed")
        checks[s["scene"]] = q
        quality = quality and q
    margins = dict(delta_correlation=d["correlation_margin"],delta_C=d["C_mean_margin"],
                   delta_C_std=d["C_std_margin"],delta_C_positive_probability=d["C_positive_probability_margin"])

    def supported(name):
        item = val["metrics"][name]
        m,ci = item["mean"],item["ci95"]
        if m is None or ci is None or abs(m) < margins[name] or ci[0]*ci[1] <= 0:
            return False
        if (item["direction_consistency_fraction"] or 0) < d["minimum_direction_fraction"]:
            return False
        if any(v is None or v*m <= 0 or abs(v) < margins[name] for v in item["leave_one_scene"].values()):
            return False
        field = item["leave_one_field"]
        if (not field["available"] or (field["direction_fraction"] or 0) < d["minimum_leave_field_direction_fraction"]
                or field["maximum_aggregate_relative_change"] is None
                or field["maximum_aggregate_relative_change"] > d["maximum_leave_field_relative_change"]):
            return False
        stable = [v["same_direction_fraction"] for v in field["per_scene"].values()]
        if sum(v is not None and v >= d["minimum_leave_field_direction_fraction"] for v in stable)/len(stable) < d["minimum_direction_fraction"]:
            return False
        for scene in chosen:
            objects = scene["per_object_metrics"][name]
            effects = [abs(v) for v in objects.values() if v is not None]
            if not effects or sum(effects) == 0 or max(effects)/sum(effects) > d["maximum_effect_target_share"]:
                return False
            nominal = scene["metrics"][name]
            if nominal and any(v is None or v*nominal <= 0 for v in scene["leave_one_object"][name].values()):
                return False
        return True

    effects = {name:supported(name) for name in METRICS}
    # A nonsignificant interval is never evidence that an effect is negligible.
    small = {}
    for name in METRICS:
        item = val["metrics"][name]
        ci,field = item["ci95"],item["leave_one_field"]
        small[name] = bool(ci and max(map(abs,ci)) < margins[name] and field["available"]
            and field["maximum_aggregate_abs_change"] is not None
            and field["maximum_aggregate_abs_change"] < margins[name]/2)
    decision = "INCONCLUSIVE"
    if quality and effects["delta_correlation"] and any(effects[m] for m in METRICS if m != "delta_correlation"):
        decision = "CONTINUE_TO_DETECTOR"
    elif quality and all(small.values()):
        decision = "STOP_OR_REDESIGN"
    return dict(base,decision=decision,quality_sufficient=bool(quality),per_scene_quality=checks,
        supported_effects=effects,intervals_within_preregistered_small_effect_bounds=small,
        balance_scope="exact per-CAV per-fixed-ray/target distribution over all20, not per-repetition global equality",
        project_recommendation="only a future fixed-weight small AttFuse diagnostic is eligible; obtain user instruction before execution"
            if decision == "CONTINUE_TO_DETECTOR" else "stop expansion of this SCOR-W version into a full weather dataset; retain evidence without seeking a favorable seed/scene/parameter")

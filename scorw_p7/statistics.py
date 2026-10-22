"""Equal-scene exploratory inference. Weather recombinations are never scene replicates."""
import math
import random

import numpy as np

from scorw_p7.signal_metrics import METRICS, mean


def interval(values, config):
    valid = {s:float(v) for s,v in values.items() if v is not None}
    result = dict(mean=mean(valid.values()),ci95=None,total_scenes=len(values),valid_scenes=len(valid),
                  missing_scenes=sorted(set(values)-set(valid)),unit="road scene; all targets/frames/pool fields nested",
                  per_scene=values,bootstrap_replicates=0)
    if len(valid) >= config["minimum_scenes"]:
        rng = random.Random(config["seed"])
        v = list(valid.values())
        draws = [mean(rng.choices(v,k=len(v))) for _ in range(config["replicates"])]
        result.update(ci95=np.quantile(draws,[.025,.975]).tolist(),bootstrap_replicates=config["replicates"])
    direction = 1 if (result["mean"] or 0) > 0 else -1 if (result["mean"] or 0) < 0 else 0
    result.update(between_scene_std=float(np.std(list(valid.values()))) if valid else None,
        direction_consistency_fraction=mean(float(v*direction>0) for v in valid.values()) if direction else None,
        leave_one_scene={s:mean(v for other,v in valid.items() if other != s) for s in valid},
        interpretation="exploratory percentile scene bootstrap conditional on this frozen pool/sample; no detector or population guarantee")
    return result


def analyze(scenes, config):
    ids = [s["scene"] for s in scenes]
    if len(set(ids)) != len(ids) or any(s["pool_size"] != 20 for s in scenes):
        raise ValueError("Independent-scene inventory duplicated or incomplete20-field cycle")
    partitions = {}
    for partition in ("generator_development","independent_validation"):
        chosen = [s for s in scenes if s["partition"] == partition]
        metrics = {}
        for metric in METRICS:
            values = {s["scene"]:s["metrics"][metric] for s in chosen}
            item = interval(values,config["bootstrap"])
            variants, scene_stability = [], {}
            for scene in chosen:
                nominal = scene["metrics"][metric]
                auxiliary = [v["metrics"][metric] for v in scene["leave_one_field"]["results"]]
                available = [v for v in auxiliary if v is not None]
                scene_stability[scene["scene"]] = dict(count=len(available),
                    same_direction_fraction=mean(float(v*nominal>0) for v in available) if nominal else None,
                    maximum_abs_change=max((abs(v-nominal) for v in available),default=None) if nominal is not None else None)
                for ix,value in enumerate(auxiliary):
                    mixed = dict(values,**{scene["scene"]:value})
                    variants.append(dict(scene=scene["scene"],omitted_field=ix,mean=mean(mixed.values())))
            nominal = item["mean"]
            differences = [abs(v["mean"]-nominal) for v in variants if v["mean"] is not None] if nominal is not None else []
            item["leave_one_field"] = dict(variants=variants,per_scene=scene_stability,
                available=len(variants) == 20*len(chosen) and bool(chosen),
                maximum_aggregate_abs_change=max(differences,default=None),
                maximum_aggregate_relative_change=max(differences)/abs(nominal) if differences and nominal else None,
                direction_fraction=mean(float(v["mean"]*nominal>0) for v in variants if v["mean"] is not None) if nominal else None,
                scope="one scene's pool reduced to19 at a time, all other scenes unchanged; auxiliary sensitivity, not extra samples or a CI")
            metrics[metric] = item
        partitions[partition] = dict(independent_scene_count=len(chosen),scene_ids=[s["scene"] for s in chosen],metrics=metrics,
            effective_counts={s["scene"]:s["counts"] for s in chosen},
            weather_recombinations_per_scene=20,frames_per_scene=2,
            scope="development only" if partition == "generator_development" else "SCOR-W P5/P6 unused non-future scenes")
    return dict(partitions=partitions,independent_unit="scene",all_twenty_fields_retained_in_primary=True,
                no_matched_repetition_filter=True,frame_or_target_independence_assumed=False)

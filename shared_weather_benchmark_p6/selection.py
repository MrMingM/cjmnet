"""Metadata-only preregistration. No weather or detector result enters selection."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from shared_weather_benchmark_p6.core import identity, seed_for, digest


def repeat_seeds(config):
    return {str(i):seed_for(config["weather_root_seed"],"scorw-p6-repeat-v1",i)
            for i in config["repeat_indices"]}


def choose_scenes(old,config):
    partitions = old["all_scene_partition"]
    if set(partitions) != {"generator_development","physics_validation","future_model_development"}:
        raise ValueError("Missing complete legacy partition reservation")
    all_ids = [s for values in partitions.values() for s in values]
    if len(set(all_ids)) != len(all_ids) or set(all_ids) != set(old["profiles"]):
        raise ValueError("Legacy partition membership/profile inventory mismatch")
    used = set(config["historical_scenes"]) | {f["scene"] for f in old["frames"]}
    reserved = set(partitions["future_model_development"]) | set(old["selected"]["future_model_development"])
    selected,eligible = {},{}
    for partition,count in config["scene_counts"].items():
        names = [s for s in partitions[partition] if s not in used | reserved]
        buckets = defaultdict(list)
        for s in names:
            p = old["profiles"][s]
            buckets[(min(p["cav_count"],4),p["initial_xy_span_m"] >= 30)].append(s)
        for key in buckets:
            buckets[key].sort(key=lambda s:(seed_for(config["selection_seed"],"p6-scene-order",partition,s),s))
        order = [buckets[k][depth] for depth in range(max(map(len,buckets.values()),default=0))
                 for k in sorted(buckets) if depth < len(buckets[k])]
        eligible[partition] = order
        if len(order) < count:
            raise ValueError(f"Insufficient unused non-future scenes for {partition}: {len(order)} < {count}; reservation cannot be relaxed")
        selected[partition] = order[:count]
    return dict(selected=selected,eligible_order=eligible,excluded_historical=sorted(used),
        excluded_future=sorted(reserved),legacy_partitions_preserved=True,
        rule="within legacy dev/physics partitions; exclude all future partition and historical scenes; strata(CAV capped4,span>=30m), seeded order, interleave")


def contiguous_block(stamps,seed,count):
    if len(stamps) < count or len(set(stamps)) != len(stamps):
        raise ValueError("Too few/duplicate timestamps; no replacement scene allowed")
    numeric = list(map(int,stamps))
    differences = [b-a for a,b in zip(numeric,numeric[1:])]
    if numeric != sorted(numeric) or not differences or min(differences) <= 0 or len(set(differences)) != 1:
        raise ValueError("Mocked timestamp sequence is not continuous")
    start = seed % (len(stamps)-count+1)
    return stamps[start:start+count]


def build_freeze(old,config,inventory,old_selection_sha256):
    """Pure selection function also used by synthetic tests."""
    plan = choose_scenes(old,config)
    frames = []
    for partition,names in plan["selected"].items():
        for scene in names:
            item = inventory[scene]
            cavs = item["cavs"]
            if len(cavs) < 2 or len(set(cavs)) != len(cavs):
                raise ValueError("Cooperative selection requires distinct CAVs")
            stamps = contiguous_block(item["timestamps"],seed_for(config["selection_seed"],"p6-frame-block",scene),config["frames_per_scene"])
            for stamp in stamps:
                frames.append(dict(scene=scene,timestamp=stamp,cavs=cavs,ego=cavs[0],partition=partition))
    return dict(schema=1,**plan,frames=frames,source_inventory=inventory,
        legacy_selection_sha256=old_selection_sha256,selection_seed=config["selection_seed"],
        weather_seeds=repeat_seeds(config),repeat_indices=config["repeat_indices"],
        config_sha256=identity(config),detection_or_weather_results_used=False,
        time="same fixed two mocked timestamps across all seeds; physical emission times unavailable",
        scope="new relative to SCOR-W historical use; prior unrelated detector training is not erased")


def server_freeze(old,config,old_selection_sha256):
    from shared_weather_benchmark.dataset_io import approved_path,cav_order,timestamps
    root = approved_path(config["train_root"])
    plan = choose_scenes(old,config)
    inventory = {}
    for names in plan["selected"].values():
        for scene in names:
            folder = approved_path(root/scene,root)
            if folder.parent != root or folder.name != scene:
                raise ValueError("Scene link identity mismatch")
            cavs = cav_order(folder,root)
            stamps = timestamps(folder/cavs[0],root)
            for cav in cavs:
                if timestamps(folder/cav,root) != stamps:
                    raise ValueError("Cross-CAV timestamp inventory differs")
            inventory[scene] = dict(cavs=cavs,timestamps=stamps,timestamp_inventory_sha256=identity(stamps))
    return build_freeze(old,config,inventory,old_selection_sha256)


def scope_selection(freeze,config,smoke):
    selected = {p:names[:config["smoke_scenes_per_partition"]] if smoke else names
                for p,names in freeze["selected"].items()}
    scope = {s for names in selected.values() for s in names}
    return dict(schema=1,frames=[f for f in freeze["frames"] if f["scene"] in scope],
        selected=selected,smoke=smoke,formal_plan_identity=identity(freeze),
        detection_ap_used=False,full_plan=frozen_selection_identity(freeze),
        note="SMOKE is a preregistered subset; full run cannot reselect scenes after inspecting SMOKE")


def frozen_selection_identity(freeze):
    return identity(dict(selected=freeze["selected"],frames=freeze["frames"],weather_seeds=freeze["weather_seeds"],
                         excluded_historical=freeze["excluded_historical"],excluded_future=freeze["excluded_future"]))

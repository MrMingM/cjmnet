"""History-aware metadata-only selection, with an explicit validation-unavailable state."""
from collections import defaultdict

from scorw_p7.core import identity, seed_for, digest


def choose(history, config):
    old, p6 = history["old"], history["p6"]
    partitions = old["all_scene_partition"]
    names = [s for values in partitions.values() for s in values]
    if (set(partitions) != {"generator_development", "physics_validation", "future_model_development"}
            or len(names) != len(set(names)) or set(names) != set(old["profiles"])):
        raise ValueError("Incomplete or overlapping historical scene partitions")
    reserved = set(partitions["future_model_development"]) | set(old["selected"]["future_model_development"])
    used = {f["scene"] for f in old["frames"]} | {f["scene"] for f in p6["frames"]}
    p6_selected = p6["selected"]
    dev = sorted(p6_selected["generator_development"])
    if (len(dev) != 4 or len(set(dev)) != 4 or set(dev) & reserved or not set(dev) <= set(names)
            or set(p6_selected["physics_validation"]) & set(dev)
            or len(p6_selected["physics_validation"]) != 8
            or set(p6["excluded_future"]) != reserved):
        raise ValueError("P6 development/reserved scene identities inconsistent")
    eligible = set(names) - used - reserved
    buckets = defaultdict(list)
    for s in eligible:
        p = old["profiles"][s]
        buckets[(min(p["cav_count"],4), p["initial_xy_span_m"] >= 30)].append(s)
    for key in buckets:
        buckets[key].sort(key=lambda s: (seed_for(config["selection_seed"], "p7-scene", s), s))
    order = [buckets[k][i] for i in range(max(map(len,buckets.values()), default=0))
             for k in sorted(buckets) if i < len(buckets[k])]
    available = len(order) >= 8
    selected = dict(generator_development=dev, independent_validation=order[:8] if available else [])
    return dict(selected=selected, eligible_order=order, excluded_historical=sorted(used), excluded_future=sorted(reserved),
                validation_status="available_8_new_scenes" if available else "INCONCLUSIVE_VALIDATION_UNAVAILABLE",
                requested_validation_scenes=8, available_validation_candidates=len(order),
                rule="metadata-only CAV-count/layout strata; seeded hash order; no weather/AP result selection",
                novelty="unused in SCOR-W P5/P6 analysis; historical detector training exposure is not erased")


def continuous(stamps, count, seed):
    numeric = [int(s) for s in stamps]
    differences = [b-a for a,b in zip(numeric,numeric[1:])]
    if (len(stamps) < count or len(set(numeric)) != len(stamps) or numeric != sorted(numeric)
            or not differences or min(differences) <= 0 or len(set(differences)) != 1):
        raise ValueError("Invalid mocked timestamp continuity; no replacement scene/frame")
    start = seed % (len(stamps)-count+1)
    return stamps[start:start+count]


def server_selection(history, config):
    from shared_weather_benchmark.dataset_io import approved_path, cav_order, timestamps, read_metadata
    from scorw_p7.core import server_train
    root = server_train(config["train_root"])
    plan = choose(history, config)
    if {p.name for p in root.iterdir() if p.is_dir()} != set(history["old"]["profiles"]):
        raise ValueError("Original scene inventory changed since historical partitions")
    frames, inventory, inputs = [], {}, []
    for partition, scenes in plan["selected"].items():
        for scene in scenes:
            folder = approved_path(root/scene, root)
            if folder.parent != root or folder.name != scene:
                raise ValueError("Scene alias changes identity")
            cavs = cav_order(folder, root)
            if len(cavs) > 20 or len(cavs) < 2:
                raise ValueError("Cooperative pool requires 2..20 distinct CAVs")
            stamps = timestamps(folder/cavs[0], root)
            historical_stamps = history["old"].get("profile_timestamp_inventory_sha256",{}).get(scene)
            if historical_stamps is not None and identity(stamps) != historical_stamps:
                raise ValueError("Original timestamp inventory changed since historical selection")
            for cav in cavs:
                if timestamps(folder/cav, root) != stamps:
                    raise ValueError("Timestamp identity differs across CAVs")
            chosen = continuous(stamps, 2, seed_for(config["selection_seed"], "p7-frames", scene))
            if partition == "generator_development":
                old_frames = [f for f in history["p6"]["frames"] if f["scene"] == scene]
                chosen = sorted(f["timestamp"] for f in old_frames)
                if len(chosen) != 2 or any(f["cavs"] != cavs or f["ego"] != cavs[0] for f in old_frames):
                    raise ValueError("P6 development CAV/frame identities changed")
                if chosen not in [stamps[i:i+2] for i in range(len(stamps)-1)]:
                    raise ValueError("Historical development frames are no longer consecutive")
            # Initial layout is the only pose information allowed in selection.
            poses = []
            for cav in cavs:
                path = approved_path(folder/cav/(stamps[0]+".yaml"), root)
                sha = digest(path)
                poses.append(read_metadata(path, root)["lidar_pose"])
                if digest(path) != sha:
                    raise ValueError("Selection metadata changed during read")
                inputs.append(dict(path=str(path), sha256=sha))
            import math
            span = max(math.dist(a[:2], b[:2]) for a in poses for b in poses)
            profile = history["old"]["profiles"][scene]
            if len(cavs) != profile["cav_count"] or abs(span-profile["initial_xy_span_m"]) > 1e-9:
                raise ValueError("Historical selection layout changed")
            inventory[scene] = dict(cavs=cavs, ego=cavs[0], timestamp_count=len(stamps),
                                    timestamps_sha256=identity(stamps), selected=chosen)
            frames.extend(dict(scene=scene, timestamp=s, cavs=cavs, ego=cavs[0], partition=partition) for s in chosen)
    return dict(plan, frames=frames, inventories=inventory, metadata_inputs=inputs,
                history_identity=history["identity"], weather_computed=False, frames_per_scene=2)

"""Keep scene/ego identities; qualify adjacent frames using Clean poses only."""
from copy import deepcopy
import math

from scorw_p7.core import digest, identity


def frame_connectivity(scene, stamp, cavs, ego, read_pose, radius):
    """Use exactly the frozen audit's inclusive XY Euclidean distance rule."""
    import numpy as np
    if (len(cavs) != len(set(cavs)) or ego not in cavs or not 2 <= len(cavs) <= 20
            or radius != 70.0):
        raise ValueError("Invalid CAV identity or changed communication radius")
    origins = {}
    for cav in cavs:
        pose = read_pose(scene, stamp, cav)
        if len(pose) != 6 or not all(math.isfinite(float(v)) for v in pose):
            raise ValueError(f"Invalid lidar_pose: {scene}/{stamp}/{cav}")
        origins[cav] = np.asarray(pose[:3], dtype=float)
    distances = {c: float(np.linalg.norm((origins[c] - origins[ego])[:2])) for c in cavs}
    active = [c for c in cavs if distances[c] <= radius]
    return dict(scene=scene, timestamp=stamp, ego=ego, cavs=list(cavs),
                xy_distance_from_ego_m=distances, active_cavs=active,
                active_cav_count=len(active), admissible=len(active) >= 2)


def qualify_selection(plan, config, read_timestamps, read_pose):
    """Pure metadata planner; no PCD, GT evidence, weather or model results."""
    result = deepcopy(plan)
    radius = config["frame_selection"]["communication_range_m"]
    if radius != 70.0 or config["frames_per_scene"] != 2:
        raise ValueError("Frame repair must preserve two frames and the frozen 70m rule")
    original_selected = deepcopy(plan["selected"])
    reports, frames, failed_dev, unavailable_val = {}, [], [], []
    for partition, scenes in original_selected.items():
        for scene in scenes:
            info = result["inventories"][scene]
            cavs, ego = info["cavs"], info["ego"]
            stamps = list(read_timestamps(scene))
            if len(stamps) != info["timestamp_count"] or identity(stamps) != info["timestamps_sha256"]:
                raise ValueError("Timestamp inventory changed during preflight: " + scene)
            original = list(info["selected"])
            old_frames = [f for f in plan["frames"] if f["scene"] == scene]
            if (len(old_frames) != 2 or [f["timestamp"] for f in old_frames] != original
                    or any(f["cavs"] != cavs or f["ego"] != ego or f["partition"] != partition for f in old_frames)):
                raise ValueError("Original frame/CAV/ego identities inconsistent: " + scene)
            starts = [i for i in range(len(stamps)-1) if stamps[i:i+2] == original]
            if len(starts) != 1:
                raise ValueError("Original two-frame block is not adjacent: " + scene)
            start = starts[0]
            order = [start] if partition == "generator_development" else list(range(start, len(stamps)-1)) + list(range(start))
            cache, checked, chosen = {}, [], []
            for i in order:
                block = stamps[i:i+2]
                for stamp in block:
                    if stamp not in cache:
                        cache[stamp] = frame_connectivity(scene, stamp, cavs, ego, read_pose, radius)
                passed = all(cache[stamp]["admissible"] for stamp in block)
                checked.append(dict(start_index=i, timestamps=block, admissible=passed))
                if passed:
                    chosen = block
                    break
            reports[scene] = dict(partition=partition, ego=ego, cavs=list(cavs), original_frames=original,
                selected_frames=chosen, changed_from_v1=bool(chosen and chosen != original),
                tested_blocks=checked, frame_checks=[cache[s] for s in stamps if s in cache],
                status="qualified" if chosen else "failed_frozen_development_frames" if partition == "generator_development" else "no_admissible_consecutive_pair",
                timestamps_inspected=len(cache), timestamp_inventory_sha256=identity(stamps))
            info.update(original_v1_selected=original, selected=chosen)
            if not chosen:
                (failed_dev if partition == "generator_development" else unavailable_val).append(scene)
            else:
                frames.extend(dict(scene=scene, timestamp=s, cavs=list(cavs), ego=ego, partition=partition) for s in chosen)
    # Never replace one of the eight chosen scenes or report a reduced cohort as eight.
    if unavailable_val:
        result["selected"]["independent_validation"] = []
        frames = [f for f in frames if f["partition"] != "independent_validation"]
        result["validation_status"] = "INCONCLUSIVE_VALIDATION_UNAVAILABLE"
        result["validation_unavailable_reason"] = (
            "The originally selected eight validation scenes cannot all supply two adjacent frames "
            "with at least two CAVs within the unchanged 70m XY radius. No admissible block in: "
            + ", ".join(unavailable_val) + ". All eight validation scenes are left unexecuted; "
            "no replacement scenes, future scenes, weather outcomes or detector results were used.")
    elif result["validation_status"] == "INCONCLUSIVE_VALIDATION_UNAVAILABLE":
        result["validation_unavailable_reason"] = (
            "Fewer than eight historically unused, non-future train scenes were available under "
            "the frozen P7 scene selection; development retained, validation unexecuted.")
    result.update(frames=frames, original_v1_scene_selection=original_selected,
        connectivity_preflight=dict(status="failed" if failed_dev else "passed",
            rule=config["frame_selection"], scene_reports=reports,
            failed_development_scenes=failed_dev, unavailable_validation_scenes=unavailable_val,
            attempted_validation_scenes=original_selected["independent_validation"],
            all_validation_unexecuted=not bool(result["selected"]["independent_validation"]),
            scene_replacement=False, weather_computed=False, pcd_or_gt_evidence_used=False))
    return result


def require_connectivity(plan):
    """A bad frozen development frame remains a hard failure with full identities."""
    report = plan["connectivity_preflight"]
    if report["status"] != "passed":
        bad = [f for s in report["failed_development_scenes"]
               for f in report["scene_reports"][s]["frame_checks"] if not f["admissible"]]
        raise ValueError("Frozen development frames fail the unchanged 70m communication gate; "
                         "see P7_SCENE_SELECTION.json. Failed frame identities/distances: " + str(bad))


def verify_metadata_inputs(plan):
    for row in plan["metadata_inputs"]:
        if digest(row["path"]) != row["sha256"]:
            raise ValueError("Consulted selection metadata changed: " + row["path"])


def verify_audited_connectivity(plan, audited_frames, expected_frames):
    expected = [(f["scene"], f["timestamp"]) for f in expected_frames]
    actual = [(f["scene"], f["timestamp"]) for f in audited_frames]
    if len(expected) != len(set(expected)) or sorted(expected) != sorted(actual):
        raise ValueError("Preflight/frozen coordinate audit frame inventory mismatch")
    for frame in audited_frames:
        report = plan["connectivity_preflight"]["scene_reports"][frame["scene"]]
        matches = [f for f in report["frame_checks"] if f["timestamp"] == frame["timestamp"]]
        if (len(matches) != 1 or frame["timestamp"] not in report["selected_frames"]
                or frame["ego"] != report["ego"] or frame["cavs"] != report["cavs"]
                or frame["partition"] != report["partition"]
                or frame["active_cavs"] != matches[0]["active_cavs"]):
            raise ValueError("Preflight/frozen coordinate audit CAV identity mismatch: "
                             + frame["scene"] + "/" + frame["timestamp"])


def server_selection(history, config):
    from scorw_p7.select_scenes import server_selection as original_selection
    from scorw_p7.core import server_train
    from shared_weather_benchmark.dataset_io import approved_path, timestamps, read_metadata
    plan = original_selection(history, config)
    root = server_train(config["train_root"])
    inputs = {row["path"]: row["sha256"] for row in plan["metadata_inputs"]}
    pose_cache = {}

    def stamps(scene):
        return timestamps(root/scene/plan["inventories"][scene]["cavs"][0], root)

    def pose(scene, stamp, cav):
        key = (scene, stamp, cav)
        if key not in pose_cache:
            path = approved_path(root/scene/cav/(stamp+".yaml"), root)
            sha = digest(path)
            data = read_metadata(path, root)["lidar_pose"]
            if digest(path) != sha or (str(path) in inputs and inputs[str(path)] != sha):
                raise ValueError("Selection metadata changed during pose-only preflight: " + str(path))
            inputs[str(path)] = sha
            pose_cache[key] = data
        return pose_cache[key]

    result = qualify_selection(plan, config, stamps, pose)
    # Rejected-frame metadata also determines the selection and is part of its identity.
    for path, sha in inputs.items():
        if digest(path) != sha:
            raise ValueError("Consulted selection metadata changed before freeze: " + path)
    result["metadata_inputs"] = [dict(path=p, sha256=inputs[p]) for p in sorted(inputs)]
    return result

"""Linux-only OPV2V inputs. No model, Torch, dataset augmentation, or test IO."""
from __future__ import annotations

import os
from pathlib import Path
import sys

from shared_weather_benchmark.provenance import REPO, digest, identity, seed_for
from shared_weather_benchmark.geometry import pose_matrix


def server_only():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("P3/P4 real OPV2V IO must run on the Linux server")


def approved_path(path, root=None):
    p = Path(path).expanduser().resolve(strict=True)
    if any(part.lower() in {"test", "validate", "validation"} or "opv2v-w" in part.lower()
           or "v2xr" in part.lower() or "v2x-r" in part.lower() for part in p.parts):
        raise ValueError("Only original OPV2V train is allowed: " + str(p))
    if root is not None and p != root and root not in p.parents:
        raise ValueError("Nested link escapes original train: " + str(p))
    return p


def validate_roots(config):
    server_only()
    source = approved_path(config["train_root"])
    if source.name != "train":
        raise ValueError("train_root must resolve to original train directory")
    output = Path(config["output_root"]).expanduser().resolve()
    data = Path("/data/cjm/datasets").resolve(strict=True)
    if data not in output.parents or output == source or source in output.parents or output in source.parents:
        raise ValueError("Output must be a separate directory under /data/cjm/datasets")
    forbidden = ("opv2v-w", "opv2v-reference-weather", "opv2v-physics", "v2xr", "v2x-r")
    if (REPO == output or REPO in output.parents or output.name in {"train","test","validate"}
            or any(any(x in part.lower() for x in forbidden) for part in output.parts)):
        raise ValueError("Refusing frozen/unsafe output location")
    return source, output


def read_metadata(path, root):
    import yaml
    p = approved_path(path,root)
    # Original OPV2V includes Python/NumPy YAML tags. Only trusted original train.
    # Do not invoke yaml_parser/eval hooks from model configuration loaders.
    with p.open(encoding="utf-8") as stream:
        value = yaml.load(stream,Loader=yaml.Loader)
    if not isinstance(value,dict) or "yaml_parser" in value:
        raise ValueError("Unexpected frame YAML schema")
    pose_matrix(value["lidar_pose"])
    if not isinstance(value.get("vehicles"),dict):
        raise ValueError("Missing vehicles/object identities")
    return value


def cav_order(scene, root):
    paths = [approved_path(p,root) for p in scene.iterdir() if p.is_dir()]
    # Check names on original entries too; aliases cannot impersonate another CAV.
    cavs = sorted(p.name for p in scene.iterdir() if p.is_dir())
    if not cavs or any(not c.lstrip("-").isdigit() for c in cavs) or len(paths) != len(set(paths)):
        raise ValueError("Invalid/aliased CAV identities")
    if any(p.parent != scene.resolve(strict=True) or p.name != name for p,name in zip(
            [approved_path(scene/c,root) for c in cavs],cavs)):
        raise ValueError("CAV directory link changes scene/vehicle identity")
    return cavs[1:]+cavs[:1] if int(cavs[0]) < 0 else cavs


def timestamps(folder, root):
    stamps = sorted(p.stem for p in folder.glob("*.yaml") if "additional" not in p.name)
    if not stamps or any(not s.isdigit() for s in stamps) or len(set(map(int,stamps))) != len(stamps):
        raise ValueError("Invalid/duplicate mocked timestamps")
    if list(map(int,stamps)) != sorted(map(int,stamps)):
        raise ValueError("Timestamp lexical/numeric order mismatch")
    for stamp in stamps:
        approved_path(folder/(stamp+".yaml"),root)
    return stamps


def select_scenes(config, smoke=False):
    """Freeze split by scene, stratified by CAV count / initial maximum XY span.

    Only first-frame poses used for layout. No points, GT outcomes or model AP.
    Future-model scenes are reserved; their PCDs are never read in this prototype.
    """
    import math
    root = approved_path(config["train_root"])
    strata = {}
    profile = {}
    profile_inputs = []
    timestamp_inventories = {}
    for scene in sorted(root.iterdir()):
        if not scene.is_dir():
            continue
        resolved = approved_path(scene,root)
        if resolved.parent != root or resolved.name != scene.name:
            raise ValueError("Scene alias changes source identity")
        cavs = cav_order(scene,root)
        scene_stamps = timestamps(scene/cavs[0],root)
        timestamp_inventories[scene.name] = identity(scene_stamps)
        stamp = scene_stamps[0]
        poses = []
        for cav in cavs:
            yaml = approved_path(scene/cav/(stamp+".yaml"),root)
            before = digest(yaml)
            poses.append(read_metadata(yaml,root)["lidar_pose"])
            if digest(yaml) != before:
                raise ValueError("Scene-profile input changed while reading")
            profile_inputs.append(dict(scene=scene.name,cav=cav,timestamp=stamp,
                                       yaml=str(yaml),sha256=before,role="selection_layout_only"))
        span = max(math.dist(a[:2],b[:2]) for a in poses for b in poses)
        profile[scene.name] = dict(cav_count=len(cavs), initial_xy_span_m=span)
        key = (min(len(cavs),4),span >= 30)
        strata.setdefault(key,[]).append(scene.name)
    assigned = {name: [] for name in config["partitions"]}
    cursor = 0
    for key in sorted(strata):
        for name in sorted(strata[key],key=lambda s: seed_for(config["seed"],"scene-split",s)):
            assigned[config["partitions"][cursor % 3]].append(name)
            cursor += 1
    n = config["scenes_per_partition"]
    selected = {}
    for partition, names in assigned.items():
        buckets = {}
        for name in names:
            p = profile[name]
            key = (min(p["cav_count"],4),p["initial_xy_span_m"] >= 30)
            buckets.setdefault(key,[]).append(name)
        interleaved = []
        for depth in range(max((len(v) for v in buckets.values()),default=0)):
            for key in sorted(buckets):
                if depth < len(buckets[key]):
                    interleaved.append(buckets[key][depth])
        selected[partition] = interleaved[:n]
    if any(len(v) != n for v in selected.values()):
        raise ValueError("Too few scenes for the three disjoint partitions")
    frames = []
    continuity = {}
    for partition in config["partitions"][:2]:
        for scene in selected[partition]:
            cavs = cav_order(root/scene,root)
            stamps = timestamps(root/scene/cavs[0],root)
            for cav in cavs:
                if timestamps(root/scene/cav,root) != stamps:
                    raise ValueError("CAV timestamp identity mismatch: " + scene + "/" + cav)
            differences = [int(b)-int(a) for a,b in zip(stamps,stamps[1:])]
            if not differences or min(differences) <= 0 or len(set(differences)) != 1:
                raise ValueError("Non-contiguous mocked timestamp sequence: " + scene)
            continuity[scene] = dict(count=len(stamps), index_step=differences[0],
                                      seconds_per_step=None, physical_scan_timing="unavailable")
            count = config["smoke_frames_per_scene"] if smoke else config["frames_per_scene"]
            if len(stamps) < count:
                raise ValueError("Scene too short; selection rule cannot silently change")
            start = seed_for(config["seed"],"frame-block",scene) % (len(stamps)-count+1)
            for stamp in stamps[start:start+count]:
                frames.append(dict(scene=scene,timestamp=stamp,cavs=cavs,ego=cavs[0],partition=partition))
    return dict(schema=1,rule="strata(count capped4, initial span>=30m); SHA order; round-robin split; interleave strata and first N each; seeded contiguous block",
                selected=selected, all_scene_partition=assigned, profiles=profile,
                selection_input_hashes=profile_inputs,profile_timestamp_inventory_sha256=timestamp_inventories,
                continuity=continuity, frames=frames, smoke=smoke, detection_ap_used=False)


def source_paths(root, frame, cav):
    prefix = root/frame["scene"]/cav/frame["timestamp"]
    pcd,yaml = approved_path(prefix.with_suffix(".pcd"),root),approved_path(prefix.with_suffix(".yaml"),root)
    if any(p.parent != prefix.parent.resolve(strict=True) or p.stem != frame["timestamp"] for p in (pcd,yaml)):
        raise ValueError("Input link changes frame/CAV identity")
    return pcd,yaml


def read_points(path):
    import numpy as np
    from opencood.utils.pcd_utils import pcd_to_np
    fields = None
    declared = None
    with Path(path).open("rb") as stream:
        for _ in range(100):
            line = stream.readline().decode("ascii").strip()
            if line.startswith("FIELDS "):
                fields = line.split()[1:]
            if line.startswith("POINTS "):
                declared = int(line.split()[1])
            if line.startswith("DATA "):
                break
        else:
            raise ValueError("Missing PCD DATA header")
    if fields != ["x","y","z","rgb"]:
        raise ValueError("Unsupported intensity encoding; expected x y z rgb")
    points = pcd_to_np(str(path)) if declared else np.empty((0,4),dtype=np.float32)
    if declared is None or len(points) != declared or points.shape != (declared,4) or not np.isfinite(points).all():
        raise ValueError("PCD point count/schema/finite mismatch")
    if len(points) and ((points[:,3] < 0).any() or (points[:,3] > 1).any()):
        raise ValueError("Intensity outside normalized RGB range")
    return points


def ensure_link(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_symlink() and target.resolve(strict=True) == source.resolve(strict=True):
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError("Existing unrelated metadata: " + str(target))
    target.symlink_to(source.resolve(strict=True))

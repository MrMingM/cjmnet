"""Clean-only stratified ray sampling. No weather or predictions enter sampling."""
from collections import defaultdict
import math

import numpy as np

from shared_weather_benchmark.geometry import object_matrix, inverse
from shared_weather_benchmark.audit import world_xyz
from scorw_p7.core import identity, seed_for, digest, atomic_json, save_arrays, transaction


def associations(world, raw, objects):
    own = ((raw[:,0] >= -1.95) & (raw[:,0] <= 2.95) & (raw[:,1] >= -1.1) & (raw[:,1] <= 1.1))
    masks = {}
    for oid, obj in sorted(objects.items()):
        m = np.asarray(inverse(object_matrix(obj)), dtype=float)
        local = world @ m[:3,:3].T + m[:3,3]
        masks[oid] = np.flatnonzero((np.abs(local) <= np.asarray(obj["extent"])+1e-7).all(axis=1) & ~own)
    return masks


def sample_cloud(raw, world, row, frame, masks, common_counts, config):
    """Count-weighted ratio on a fixed stratified sample; not full-cloud truth."""
    budget = config["sampling"]
    edges = np.asarray(budget["distance_edges_m"], dtype=float)
    ranges = np.linalg.norm(raw[:,:3].astype(float), axis=1)
    bins = np.searchsorted(edges, ranges, side="right") - 1
    groups = []
    for oid, indices in sorted(masks.items()):
        if len(indices) and ((bins[indices] < 0).any() or (bins[indices] >= len(edges)-1).any()):
            raise ValueError("Ray falls outside preregistered range bins")
        for b in sorted(set(bins[indices].tolist())):
            for positive in (False, True):
                ix = indices[(bins[indices] == b) & ((raw[indices,3] > 0) == positive)]
                if not len(ix):
                    continue
                order = sorted(ix.tolist(), key=lambda i: (seed_for(config["sampling_seed"], "p7-ray",
                    row["scene"], row["timestamp"], row["cav"], row["pcd_sha256"], i), i))
                groups.append(dict(oid=oid, bin=int(b), positive=positive, indices=order, take=1,
                                   priority=0 if common_counts.get(oid,0) >= 2 else 1))
    groups.sort(key=lambda g: (g["priority"], g["oid"], g["bin"], g["positive"]))
    maximum = budget["max_memberships_per_cloud"]
    if len(groups) > maximum:
        raise ValueError(f"Ray budget cannot cover every Clean stratum: {len(groups)} > {maximum}; no target dropped")
    used = len(groups)
    while used < maximum:
        progressed = False
        for g in groups:
            if used >= maximum:
                break
            if g["take"] < min(budget["max_per_stratum"], len(g["indices"])):
                g["take"] += 1
                used += 1
                progressed = True
        if not progressed:
            break
    chosen = sorted({i for g in groups for i in g["indices"][:g["take"]]})
    slot = {point: i for i, point in enumerate(chosen)}
    rows = []
    for oid, indices in sorted(masks.items()):
        parts = [g for g in groups if g["oid"] == oid]
        sampled = [i for g in parts for i in g["indices"][:g["take"]]]
        weights = [len(g["indices"])/g["take"] for g in parts for _ in range(g["take"])]
        original_intensity = raw[indices,3].astype(float)
        sampled_intensity = raw[sampled,3].astype(float)
        denom = math.fsum(float(a)*float(b) for a,b in zip(sampled_intensity,weights))
        if original_intensity.sum() > 0 and denom <= 0:
            raise ValueError("Positive Clean signal lost by sampling; positive-intensity strata must be represented")
        obj = frame["objects"][oid]
        vector = np.asarray(row["pose"][:3]) - np.asarray(object_matrix(obj))[:3,3]
        rows.append(dict(scene=row["scene"], timestamp=row["timestamp"], object_id=oid, cav=row["cav"],
            ego=frame["ego"], is_ego=row["cav"] == frame["ego"], active=row["cav"] in frame["active_cavs"],
            partition=frame["partition"], clean_gt_sha256=frame["clean_gt_sha256"],
            clean_points=len(indices), sampled_points=len(sampled), sample_fraction=len(sampled)/len(indices) if len(indices) else None,
            clean_signal=float(original_intensity.sum()), sampled_clean_signal=float(sampled_intensity.sum()),
            fixed_weighted_clean_denominator=denom, ray_slots=[slot[i] for i in sampled], weights=weights,
            missing_reason="no_clean_points" if not len(indices) else "zero_clean_signal" if denom == 0 else None,
            distance_m=float(np.linalg.norm(vector)),
            view_angle_rad=(math.atan2(vector[1],vector[0])-math.radians(obj["angle"][1])+math.pi)%(2*math.pi)-math.pi))
    audit = dict(scene=row["scene"], timestamp=row["timestamp"], cav=row["cav"],
        full_cloud_points=len(raw), associated_memberships=sum(len(v) for v in masks.values()),
        full_unique_associated_points=len(set(i for v in masks.values() for i in v.tolist())),
        sampled_memberships=used, unique_sampled_rays=len(chosen),
        membership_fraction=used/sum(len(v) for v in masks.values()) if groups else None,
        strata=[dict(object_id=g["oid"], distance_bin=g["bin"], positive_clean_intensity=g["positive"],
                     complete=len(g["indices"]), sampled=g["take"]) for g in groups],
        budget=budget, overlap="box associations may overlap; selected original ray is integrated once per field",
        estimator=budget["estimator"], sample_selected_before_weather=True)
    indices = np.asarray(chosen, dtype=np.int64)
    arrays = dict(world_endpoints=world[indices], original_xyzi=raw[indices], original_index=indices,
                  origin=np.asarray(row["pose"][:3], dtype=np.float64),
                  point_id=np.asarray([identity([row["scene"],row["timestamp"],row["cav"],row["pcd_sha256"],i]) for i in chosen], dtype="U64"))
    audit["rays_identity"] = identity(dict(source=row, indices=chosen, gt=frame["clean_gt_sha256"],
        memberships=[dict(object_id=r["object_id"], slots=r["ray_slots"], weights=r["weights"], denominator=r["fixed_weighted_clean_denominator"]) for r in rows]))
    return arrays, rows, audit


def extract_scene(scene, coordinate_root, output, config, contract):
    from shared_weather_benchmark.dataset_io import read_points, source_paths
    from scorw_p7.core import read_json, server_train
    root = server_train(config["train_root"])
    inputs = read_json(coordinate_root/"INPUT_MANIFEST.json")
    frames = [f for f in read_json(coordinate_root/"CLEAN_GT.json") if f["scene"] == scene]
    index = {(r["timestamp"],r["cav"]):r for r in inputs if r["scene"] == scene}
    clouds = []
    for frame in frames:
        raw, xyz, masks = {}, {}, {}
        for cav in frame["cavs"]:
            row = index[(frame["timestamp"],cav)]
            pcd,yaml = source_paths(root,frame,cav)
            if str(pcd) != row["pcd"] or str(yaml) != row["yaml"]:
                raise ValueError("Clean source link changed scene/CAV/frame identity")
            if digest(row["pcd"]) != row["pcd_sha256"] or digest(row["yaml"]) != row["yaml_sha256"]:
                raise ValueError("Original Clean changed after coordinate audit")
            raw[cav] = read_points(row["pcd"])
            xyz[cav] = world_xyz(raw[cav], row["pose"])
            masks[cav] = associations(xyz[cav],raw[cav],frame["objects"])
            if digest(row["pcd"]) != row["pcd_sha256"]:
                raise ValueError("Original Clean changed while reading")
        common = {oid:sum(raw[c][masks[c][oid],3].astype(float).sum() > 0 for c in frame["active_cavs"])
                  for oid in frame["objects"]}
        for cav in sorted(frame["cavs"]):
            row = index[(frame["timestamp"],cav)]
            target = output/frame["timestamp"]/cav
            def build(code):
                arrays, rows, audit = sample_cloud(raw[cav],xyz[cav],row,frame,masks[cav],common,config)
                save_arrays(target/"rays.npz", **arrays)
                atomic_json(target/"targets.json",rows)
                atomic_json(target/"SAMPLING.json",audit)
            transaction(target, dict(parent=contract,source=row,gt=frame["clean_gt_sha256"],sampling=config["sampling"],
                                     seed=config["sampling_seed"]), build)
            clouds.append(target)
    return clouds

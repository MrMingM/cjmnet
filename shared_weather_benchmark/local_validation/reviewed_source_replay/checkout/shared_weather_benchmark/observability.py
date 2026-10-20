"""Fixed clean-GT point associations and continuous multi-CAV evidence proxies."""
from __future__ import annotations

import math
from collections import Counter

from shared_weather_benchmark.audit import world_xyz
from shared_weather_benchmark.dataset_io import read_points
from shared_weather_benchmark.generator import sidecar_validate, validate_record
from shared_weather_benchmark.geometry import object_matrix, inverse
from shared_weather_benchmark.provenance import read_json, atomic_json, atomic_text, identity, check_hashes, digest


def classify(rows, min_returns=3, good_ratio=.5, disparity=.4):
    """Diagnostic labels, with an explicit residual group to avoid forced labels."""
    counts = [r["weather_hard_count"] for r in rows]
    if max(counts,default=0) < min_returns:
        return "A_all_CAV_below_count_threshold"
    ego = next(r for r in rows if r["is_ego"])
    peers = [r for r in rows if not r["is_ego"]]
    if ego["weather_hard_count"] < min_returns and any(r["weather_hard_count"] >= min_returns for r in peers):
        return "B_ego_below_neighbor_has_evidence"
    ratios = [r["effective_intensity_ratio"] for r in rows if r["effective_intensity_ratio"] is not None]
    if all(n >= min_returns for n in counts) and len(ratios) >= 2 and max(ratios)-min(ratios) >= disparity:
        return "C_all_have_returns_quality_disparity"
    if (all(n >= min_returns for n in counts) and len(ratios) == len(rows)
            and min(ratios) >= good_ratio):
        return "D_all_good_under_diagnostic_thresholds"
    return "residual_mixed_or_weak"


def contains_numpy(world, obj):
    import numpy as np
    m = np.asarray(inverse(object_matrix(obj)))
    local = world@m[:3,:3].T+m[:3,3]
    return (np.abs(local) <= np.asarray(obj["extent"])+1e-7).all(axis=1)


def inspect_cloud(output,record,row):
    import numpy as np
    with np.load(output/record["sidecar"],allow_pickle=False) as payload:
        data = {k:payload[k] for k in payload.files}
    raw, weather = data["original_xyzi"],data["weather_xyzi"]
    emitted, retained, origin, tau = (data[k] for k in ("emitted","retained","origin","integral"))
    sidecar_validate(raw,weather,emitted,retained,origin,tau)
    if (not np.array_equal(data["original_index"],np.arange(len(raw)))
            or data["frame_identity"].tolist() != [row["scene"],row["timestamp"],row["cav"],row["pcd_sha256"]]
            or not np.array_equal(raw,read_points(row["pcd"]))):
        raise ValueError("Point/source identity mismatch in sidecar")
    expected = np.full(len(raw),-1,dtype=np.int64)
    expected[emitted] = np.arange(int(emitted.sum()))
    if not np.array_equal(expected,data["output_index"]):
        raise ValueError("Output index not a one-to-one ordered mapping")
    written = read_points(output/record["pcd"])
    if not np.array_equal(written,data["serialized_xyzi"][emitted]):
        raise ValueError("PCD/sidecar serialization mismatch")
    if (origin[emitted] == 2).any():
        raise ValueError("Unknown origins prohibit a true-return comparison in v1")
    return data


def audit_observability(config,output,contract):
    import numpy as np
    manifest = read_json(output/"DATA_MANIFEST.json")
    if manifest["contract"] != contract:
        raise ValueError("Dataset contract mismatch")
    check_hashes(output,manifest["record_hashes"])
    if manifest["clean_gt_sha256"] != digest(output/"CLEAN_GT.json") or manifest["input_manifest_sha256"] != digest(output/"INPUT_MANIFEST.json"):
        raise ValueError("Frozen clean GT changed")
    rows = read_json(output/"INPUT_MANIFEST.json")
    inputs = {(r["scene"],r["timestamp"],r["cav"]):r for r in rows}
    gt_frames = read_json(output/"CLEAN_GT.json")
    gt_index = {(f["scene"],f["timestamp"]):f for f in gt_frames}
    observations = []
    for name in manifest["records"]:
        record = read_json(output/name)
        key = (record["scene"],record["timestamp"],record["cav"])
        row = inputs[key]
        validate_record(output,record,row,contract)
        data = inspect_cloud(output,record,row)
        frame = gt_index[key[:2]]
        raw, weather, tau = data["original_xyzi"],data["weather_xyzi"],data["integral"]
        xyz = world_xyz(raw,row["pose"])
        weather_xyz = world_xyz(weather,row["pose"])
        # Same fixed own-body rectangle as the inspected detector loader.
        own = ((raw[:,0] >= -1.95) & (raw[:,0] <= 2.95) & (raw[:,1] >= -1.1) & (raw[:,1] <= 1.1))
        for oid,obj in frame["objects"].items():
            mask = contains_numpy(xyz,obj) & ~own
            hard = mask & data["retained"]
            simulated = data["emitted"] & (data["origin"] == 1)
            n = int(mask.sum())
            clean_energy = float(raw[mask,3].astype(float).sum())
            energy = float(weather[hard,3].astype(float).sum())
            center = np.asarray(object_matrix(obj))[:3,3]
            vector = np.asarray(row["pose"][:3])-center
            angle = (math.atan2(vector[1],vector[0])-math.radians(obj["angle"][1])+math.pi) % (2*math.pi)-math.pi
            value = dict(mode=record["mode"],scene=row["scene"],timestamp=row["timestamp"],
                object_id=oid,cav=row["cav"],partition=row["partition"],
                is_ego=row["cav"] == frame["ego"],active=row["cav"] in frame["active_cavs"],
                clean_gt_sha256=frame["clean_gt_sha256"],clean_hard_count=n,
                weather_hard_count=int(hard.sum()),
                simulated_returns_in_box=int((contains_numpy(weather_xyz,obj) & simulated).sum()),
                simulated_from_clean_object_rays=int((mask & simulated).sum()),
                distance_m=float(np.linalg.norm(vector)),view_angle_rad=angle,
                retention_ratio=float(hard.sum()/n) if n else None,
                effective_intensity_ratio=energy/clean_energy if clean_energy else None,
                mean_transmittance=float(np.exp(-2*tau[mask]).mean()) if n else None,
                mean_integral=float(tau[mask].mean()) if n else None,
                integral_p90=float(np.quantile(tau[mask],.9)) if n else None,
                intensity_threshold_counts={str(threshold):int((hard & (weather[:,3] >= threshold)).sum())
                    for threshold in config["observability"]["intensity_threshold_sensitivity"]},
                serialized_threshold_counts={str(threshold):int((hard & (data["serialized_xyzi"][:,3] >= threshold)).sum())
                    for threshold in config["observability"]["intensity_threshold_sensitivity"]},
                association="clear return in fixed clean box; object-surface truth unavailable")
            observations.append(value)
    # All four conditions must have the exact same GT/CAV denominator.
    denominator = {}
    for mode in config["protocols"]:
        denominator[mode] = sorted((r["scene"],r["timestamp"],r["object_id"],r["cav"],r["clean_hard_count"],r["clean_gt_sha256"])
                                   for r in observations if r["mode"] == mode)
    if any(v != denominator["E0"] for v in denominator.values()):
        raise ValueError("Protocol GT/point/CAV denominator mismatch")
    groups = {}
    for row in observations:
        if row["active"]:
            groups.setdefault((row["mode"],row["scene"],row["timestamp"],row["object_id"]),[]).append(row)
    targets = []
    settings = config["observability"]
    for key, values in sorted(groups.items()):
        ego = next(r for r in values if r["is_ego"])
        peers = [r for r in values if not r["is_ego"]]
        available = [r["effective_intensity_ratio"] for r in peers if r["effective_intensity_ratio"] is not None]
        gap = max(available)-ego["effective_intensity_ratio"] if available and ego["effective_intensity_ratio"] is not None else None
        labels = {}
        for n in settings["sensitivity_min_returns"]:
            for q in settings["sensitivity_good_ratio"]:
                labels[f"count={n},ratio={q}"] = classify(values,n,q,settings["disparity"])
        main_label = classify(values,settings["min_returns"],settings["good_ratio"],settings["disparity"])
        targets.append(dict(mode=key[0],scene=key[1],timestamp=key[2],object_id=key[3],
            partition=ego["partition"],category=main_label,sensitivity=labels,
            all_zero_hard_returns=all(r["weather_hard_count"] == 0 for r in values),
            ego_zero_neighbor_positive=ego["weather_hard_count"] == 0 and any(r["weather_hard_count"] > 0 for r in peers),
            all_zero_clean_returns=all(r["clean_hard_count"] == 0 for r in values),
            ego_best_neighbor_intensity_gap=gap,ego_effective_ratio=ego["effective_intensity_ratio"],
            best_neighbor_effective_ratio=max(available) if available else None,
            per_cav_retention={r["cav"]:r["retention_ratio"] for r in values}))
    atomic_text(output/"OBSERVABILITY_ROWS.jsonl","".join(__import__("json").dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in observations))
    atomic_json(output/"OBSERVABILITY_TARGETS.json",targets)
    by_scene = {}
    for scene in sorted({r["scene"] for r in targets}):
        by_scene[scene] = {}
        for mode in config["protocols"]:
            subset = [r for r in targets if r["scene"] == scene and r["mode"] == mode]
            gaps = [r["ego_best_neighbor_intensity_gap"] for r in subset if r["ego_best_neighbor_intensity_gap"] is not None]
            by_scene[scene][mode] = dict(target_instances=len(subset),
                categories=dict(Counter(r["category"] for r in subset)),
                all_zero_hard_fraction=sum(r["all_zero_hard_returns"] for r in subset)/len(subset),
                ego_zero_neighbor_positive_fraction=sum(r["ego_zero_neighbor_positive"] for r in subset)/len(subset),
                all_zero_clean_fraction=sum(r["all_zero_clean_returns"] for r in subset)/len(subset),
                mean_ego_best_neighbor_intensity_gap=float(np.mean(gaps)) if gaps else None,
                median_ego_best_neighbor_intensity_gap=float(np.median(gaps)) if gaps else None)
    summary = dict(status="passed",contract=contract,rows=len(observations),target_instances=len(targets),
        by_scene=by_scene,
        gt_cav_denominator_sha256=identity(denominator["E0"]),thresholds=settings,
        label_semantics="A/B below count threshold; literal zero flags separate; C/D intensity proxy; residual retained",
        point_semantics="known clear-return origin, GT box membership proxy; no per-point semantic truth",
        detectability_claim=False,primary_energy="pre-RGB-quantization",serial_quantization_reported=True)
    atomic_json(output/"OBSERVABILITY_SUMMARY.json",summary)
    return summary

"""Signal proxies on fixed Clean-associated points, independent of predictions."""
from __future__ import annotations

from collections import Counter, defaultdict
import itertools
import math

from shared_weather_benchmark.audit import world_xyz
from shared_weather_benchmark.geometry import object_matrix
from shared_weather_benchmark.observability import contains_numpy
from shared_weather_benchmark_p6.core import (
    mean, stats, identity, bin_name, write_jsonl, atomic_json, atomic_text)


def signal(clean, weather, quantized, retained, tau, thresholds):
    import numpy as np
    clean, weather, quantized = map(np.asarray, (clean, weather, quantized))
    retained, tau = np.asarray(retained), np.asarray(tau)
    if any(a.ndim != 1 or len(a) != len(clean) for a in (clean, weather, quantized, retained, tau)):
        raise ValueError("Unaligned signal arrays")
    if retained.dtype != np.dtype(bool):
        raise ValueError("Retained mask must be boolean, not point indices")
    if any(not np.isfinite(a).all() for a in (clean, weather, quantized, tau)):
        raise ValueError("Nonfinite signal")
    if any((a < 0).any() for a in (clean, weather, quantized, tau)):
        raise ValueError("Negative signal/optical depth")
    n = len(clean)
    c = float(clean.astype(float).sum())
    w = float(weather[retained].astype(float).sum())
    q = float(quantized[retained].astype(float).sum())
    threshold = {}
    for t in thresholds:
        cc, ww, qq = clean >= t, retained & (weather >= t), retained & (quantized >= t)
        threshold[str(t)] = dict(clean=int(cc.sum()), pre=int(ww.sum()), post=int(qq.sum()),
            clean_already_below=int((~cc).sum()), newly_below_pre=int((cc & ~ww).sum()),
            newly_below_post=int((cc & ~qq).sum()),
            pre_post_disagree=int((ww != qq).sum()))
    return dict(clean_signal=c, weather_signal=w, serialized_signal=q,
        R=w/c if c > 0 else None, R_post=q/c if c > 0 else None,
        missing_reason="no_clean_points" if not n else "zero_clean_signal" if c == 0 else None,
        clean_points=n, retained_points=int(retained.sum()),
        raw_retention=float(retained.sum()/n) if n else None,
        T=mean(np.exp(-2*tau).tolist()) if n else None,
        intensity_weighted_T=float((clean.astype(float)*np.exp(-2*tau)).sum()/c) if c > 0 else None,
        threshold=threshold, pre_zero_points=int((retained & (weather == 0)).sum()),
        post_zero_points=int((retained & (quantized == 0)).sum()),
        quantization_only_zero_points=int((retained & (weather > 0) & (quantized == 0)).sum()))


def cloud_rows(data, row, frame, mode, repeat, thresholds):
    """Every frozen object x CAV appears, including missing/zero-signal cases."""
    import numpy as np
    raw, weather = data["original_xyzi"], data["weather_xyzi"]
    xyz = world_xyz(raw, row["pose"])
    weather_xyz = world_xyz(weather, row["pose"])
    own = ((raw[:,0] >= -1.95) & (raw[:,0] <= 2.95) & (raw[:,1] >= -1.1) & (raw[:,1] <= 1.1))
    result = []
    for oid, obj in frame["objects"].items():
        mask = contains_numpy(xyz, obj) & ~own
        hard = mask & data["retained"]
        simulated = data["emitted"] & (data["origin"] == 1)
        tau = data["integral"][mask]
        values = signal(raw[mask,3], weather[mask,3], data["serialized_xyzi"][mask,3],
                        data["retained"][mask], tau, thresholds)
        center = np.asarray(object_matrix(obj))[:3,3]
        vector = np.asarray(row["pose"][:3])-center
        angle = (math.atan2(vector[1],vector[0])-math.radians(obj["angle"][1])+math.pi) % (2*math.pi)-math.pi
        n = int(mask.sum())
        # The original names remain for exact legacy formula rechecks. New
        # explicit energies/missing reasons and post-encoding values are added.
        item = dict(mode=mode, scene=row["scene"], timestamp=row["timestamp"], object_id=oid,
            cav=row["cav"], partition=row["partition"], is_ego=row["cav"] == frame["ego"],
            active=row["cav"] in frame["active_cavs"], clean_gt_sha256=frame["clean_gt_sha256"],
            clean_hard_count=n, weather_hard_count=int(hard.sum()),
            simulated_returns_in_box=int((contains_numpy(weather_xyz,obj) & simulated).sum()),
            simulated_from_clean_object_rays=int((mask & simulated).sum()),
            distance_m=float(np.linalg.norm(vector)), view_angle_rad=angle,
            retention_ratio=values["raw_retention"], effective_intensity_ratio=values["R"],
            mean_transmittance=float(np.exp(-2*tau).mean()) if n else None,
            mean_integral=float(tau.mean()) if n else None,
            integral_p90=float(np.quantile(tau,.9)) if n else None,
            intensity_threshold_counts={str(t):int((hard & (weather[:,3] >= t)).sum()) for t in thresholds},
            serialized_threshold_counts={str(t):int((hard & (data["serialized_xyzi"][:,3] >= t)).sum()) for t in thresholds},
            association="clear return in fixed clean box; object-surface truth unavailable")
        result.append(dict(item, repeat_index=repeat, **values))
    return result


def row_key(row, mode=True):
    return tuple(row[k] for k in (("mode",) if mode else ()) + ("scene","timestamp","object_id","cav"))


def check_denominators(rows, modes):
    expected = None
    for mode in modes:
        subset = [r for r in rows if r["mode"] == mode]
        keys = [row_key(r, False) for r in subset]
        if not subset or len(set(keys)) != len(keys):
            raise ValueError("Missing/duplicate mode target-CAV rows: " + mode)
        values = sorted((row_key(r, False),r["clean_gt_sha256"],r["clean_hard_count"],r["clean_signal"],r["active"])
                        for r in subset)
        if expected is None:
            expected = values
        elif values != expected:
            raise ValueError("Protocol GT/CAV/Clean denominator mismatch")
    return identity(expected)


def target_signal(rows):
    ego = [r for r in rows if r["is_ego"]]
    if len(ego) != 1:
        raise ValueError("Exactly one ego required in target group")
    ego = ego[0]
    peers = sorted((r for r in rows if not r["is_ego"]),key=lambda r:r["cav"])
    valid = [r for r in peers if r["R"] is not None]
    best = max(valid,key=lambda r:r["R"]) if valid else None
    best_post = max((r["R_post"] for r in valid),default=None)
    gap = best["R"]-ego["R"] if best is not None and ego["R"] is not None else None
    clean_gap = max((r["clean_signal"] for r in peers),default=None)
    weather_gap = max((r["weather_signal"] for r in peers),default=None)
    if peers:
        clean_gap -= ego["clean_signal"]
        weather_gap -= ego["weather_signal"]
    return dict(scene=ego["scene"],timestamp=ego["timestamp"],object_id=ego["object_id"],
        mode=ego["mode"],partition=ego["partition"],repeat_index=ego["repeat_index"],
        ego_cav=ego["cav"],active_cav_count=len(rows),neighbor_count=len(peers),valid_neighbor_count=len(valid),
        valid_neighbor_ids=[r["cav"] for r in valid],best_neighbor_cav=best["cav"] if best else None,
        C=gap,C_post=best_post-ego["R_post"] if best_post is not None and ego["R_post"] is not None else None,
        C_missing_reason="ego_signal_missing" if ego["R"] is None else "no_valid_neighbor" if not valid else None,
        ego_R=ego["R"],ego_distance_m=ego["distance_m"],
        ego_clean_points=ego["clean_hard_count"],ego_clean_signal=ego["clean_signal"],
        ego_weather_signal=ego["weather_signal"],
        best_neighbor_R=best["R"] if best else None,
        best_neighbor_distance_m=best["distance_m"] if best else None,
        best_neighbor_clean_signal=best["clean_signal"] if best else None,
        best_neighbor_weather_signal=best["weather_signal"] if best else None,
        absolute_clean_neighbor_ego_gap=clean_gap,absolute_weather_neighbor_ego_gap=weather_gap,
        weather_added_absolute_gap=weather_gap-clean_gap if peers else None,
        clean_ratio_advantage=0.0 if gap is not None else None,
        clean_zero_ego_neighbor_positive=ego["clean_hard_count"] == 0 and any(r["clean_hard_count"] > 0 for r in peers),
        raw_zero_ego_neighbor_positive=ego["weather_hard_count"] == 0 and any(r["weather_hard_count"] > 0 for r in peers))


def targets_for(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["active"]:
            groups[(row["mode"],row["scene"],row["timestamp"],row["object_id"])].append(row)
    return [target_signal(values) for _,values in sorted(groups.items())]


def pearson(xs, ys):
    pairs = [(x,y) for x,y in zip(xs,ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return dict(paired=len(pairs),value=None,reason="fewer_than_three")
    x,y = zip(*pairs)
    xx,yy = mean(x),mean(y)
    a,b = sum((z-xx)**2 for z in x),sum((z-yy)**2 for z in y)
    if a*b <= 1e-20:
        return dict(paired=len(pairs),value=None,reason="constant_or_near_constant")
    return dict(paired=len(pairs),value=sum((u-xx)*(v-yy) for u,v in pairs)/math.sqrt(a*b),reason=None)


def pair_analysis(rows, config):
    groups = defaultdict(list)
    for r in rows:
        if r["active"]:
            groups[(r["scene"],r["mode"],r["timestamp"],r["object_id"])].append(r)
    bins = defaultdict(list)
    for (scene,mode,stamp,oid),values in sorted(groups.items()):
        for a,b in itertools.combinations(sorted(values,key=lambda r:r["cav"]),2):
            delta = b["distance_m"]-a["distance_m"]
            item = dict(a=a["cav"],b=b["cav"],R_a=a["R"],R_b=b["R"],T_a=a["T"],T_b=b["T"],
                tau_difference=b["mean_integral"]-a["mean_integral"] if a["mean_integral"] is not None and b["mean_integral"] is not None else None,
                distance_difference_m=delta,
                ego_neighbor_distance_delta_m=(b["distance_m"]-a["distance_m"] if a["is_ego"] else
                                               a["distance_m"]-b["distance_m"] if b["is_ego"] else None))
            strata = config["strata"]
            labels = ["all", "cav_pair="+a["cav"]+"/"+b["cav"],
                "target_distance="+bin_name(max(a["distance_m"],b["distance_m"]),strata["distance_m"]),
                "distance_difference="+bin_name(abs(delta),strata["distance_difference_m"]),
                "clean_points="+bin_name(min(a["clean_hard_count"],b["clean_hard_count"]),strata["clean_points"]),
                "clean_signal="+bin_name(min(a["clean_signal"],b["clean_signal"]),strata["clean_signal"])]
            if a["is_ego"] or b["is_ego"]:
                labels.append("ego_neighbor")
            for label in labels:
                bins[(scene,mode,label)].append(item)
    result = {}
    for (scene,mode,label),values in sorted(bins.items()):
        result.setdefault(scene,{}).setdefault(mode,{})[label] = dict(total_pairs=len(values),
            R_correlation=pearson([r["R_a"] for r in values],[r["R_b"] for r in values]),
            T_correlation=pearson([r["T_a"] for r in values],[r["T_b"] for r in values]),
            tau_difference=stats(r["tau_difference"] for r in values),
            distance_difference_m=stats(r["distance_difference_m"] for r in values),
            ego_neighbor_distance_delta_m=stats(r["ego_neighbor_distance_delta_m"] for r in values))
    return result


def summarize(rows, config, modes):
    denominator = check_denominators(rows,modes)
    targets = targets_for(rows)
    by_scene = {}
    threshold, quantization = {}, {}
    for scene in sorted({r["scene"] for r in rows}):
        by_scene[scene] = {}
        threshold[scene],quantization[scene] = {},{}
        for mode in modes:
            rr = [r for r in rows if r["scene"] == scene and r["mode"] == mode and r["active"]]
            tt = [t for t in targets if t["scene"] == scene and t["mode"] == mode]
            if not rr or not tt:
                raise ValueError("Empty active target/CAV denominator: "+scene+"/"+mode)
            strata = defaultdict(list)
            for t in tt:
                for label in ("valid_neighbors="+str(t["valid_neighbor_count"]),
                              "active_cavs="+str(t["active_cav_count"]),
                              "ego_distance="+bin_name(t["ego_distance_m"],config["strata"]["distance_m"]),
                              "ego_clean_points="+bin_name(t["ego_clean_points"],config["strata"]["clean_points"]),
                              "ego_clean_signal="+bin_name(t["ego_clean_signal"],config["strata"]["clean_signal"])):
                    strata[label].append(t["C"])
            cstats = stats(t["C"] for t in tt)
            cstats["positive_fraction_valid"] = mean([float(t["C"] > 0) for t in tt if t["C"] is not None])
            by_scene[scene][mode] = dict(partition=rr[0]["partition"],rows=len(rr),targets=len(tt),
                R=stats(r["R"] for r in rr),R_post=stats(r["R_post"] for r in rr),C=cstats,
                C_post=stats(t["C_post"] for t in tt),C_by_stratum={k:stats(v) for k,v in sorted(strata.items())},
                missing_R=dict(Counter(r["missing_reason"] or "valid" for r in rr)),
                missing_C=dict(Counter(t["C_missing_reason"] or "valid" for t in tt)),
                valid_neighbor_counts=dict(Counter(str(t["valid_neighbor_count"]) for t in tt)),
                clean_signal=stats(r["clean_signal"] for r in rr),
                weather_signal=stats(r["weather_signal"] for r in rr),
                raw_retention=stats(r["raw_retention"] for r in rr),
                intensity_weighted_T=(sum(r["clean_signal"]*r["intensity_weighted_T"] for r in rr if r["intensity_weighted_T"] is not None)
                    /sum(r["clean_signal"] for r in rr)) if sum(r["clean_signal"] for r in rr) > 0 else None,
                absolute_clean_gap=stats(t["absolute_clean_neighbor_ego_gap"] for t in tt),
                absolute_weather_gap=stats(t["absolute_weather_neighbor_ego_gap"] for t in tt),
                weather_added_absolute_gap=stats(t["weather_added_absolute_gap"] for t in tt),
                clean_ratio_advantage=stats(t["clean_ratio_advantage"] for t in tt),
                per_cav={c:dict(R=stats(r["R"] for r in rr if r["cav"] == c),
                    clean_signal=stats(r["clean_signal"] for r in rr if r["cav"] == c),
                    rows=sum(r["cav"] == c for r in rr)) for c in sorted({r["cav"] for r in rr})})
            threshold[scene][mode] = {str(t): {k:sum(r["threshold"][str(t)][k] for r in rr)
                for k in rr[0]["threshold"][str(t)]} for t in config["thresholds"]}
            quantization[scene][mode] = dict(rows=len(rr),
                R_post_minus_pre=stats(r["R_post"]-r["R"] if r["R"] is not None else None for r in rr),
                pre_zero_points=sum(r["pre_zero_points"] for r in rr),
                post_zero_points=sum(r["post_zero_points"] for r in rr),
                quantization_only_zero_points=sum(r["quantization_only_zero_points"] for r in rr),
                note="box associations can overlap; counts are target-associated, not unique cloud points")
    return dict(status="complete_signal_proxy_analysis",denominator_sha256=denominator,
        by_scene=by_scene,pair_analysis=pair_analysis(rows,config),targets=targets,
        threshold=threshold,quantization=quantization,detectability_claim=False,
        primary="pre-encoding continuous intensity on fixed Clean-associated original returns")


def write_analysis(output, rows, config, modes):
    result = summarize(rows,config,modes)
    write_jsonl(output/"P6B_TARGET_SIGNAL_ROWS.jsonl",rows)
    write_jsonl(output/"P6B_TARGET_COMPLEMENTARITY.jsonl",result["targets"])
    atomic_json(output/"P6B_SCENE_SUMMARY.json",{k:v for k,v in result.items() if k not in {"targets","threshold","quantization"}})
    atomic_json(output/"P6B_QUANTIZATION_AUDIT.json",result["quantization"])
    atomic_json(output/"P6B_THRESHOLD_SENSITIVITY.json",dict(status="uncalibrated_approximation",by_scene=result["threshold"]))
    atomic_text(output/"P6B_RESULTS.md","# P6-B signal observability proxy\n\n仅测量信号代理，不是detector recoverability。\n\n"+
        "R=天气后保留原始关联点强度和/Clean强度和；C=最佳有效邻车R−ego R。零分母与无有效邻车保留为null并计数。\n\n"+
        "主结果见P6B_SCENE_SUMMARY.json；阈值未标定，全部阈值同时报告。量化前后分开；点数A/B主要继承Clean。\n")
    return result

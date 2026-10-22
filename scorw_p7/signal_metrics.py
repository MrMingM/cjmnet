"""Within-target weather correlation and fixed-domain cooperative signal proxies."""
from collections import Counter, defaultdict
from itertools import combinations
import math

import numpy as np

from scorw_p7.balanced_assignment import assignment
from scorw_p7.core import close
from scorw_p7.integrate_pool import strict_balance

METRICS = ("delta_correlation", "delta_C", "delta_C_std", "delta_C_positive_probability")


def mean(values):
    values = [float(v) for v in values if v is not None]
    return math.fsum(values)/len(values) if values else None


def distribution(values):
    if values is None:
        return dict(mean=None, variance=None, std=None, positive_probability=None, quantiles=None)
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError("Invalid fixed-pool distribution")
    return dict(mean=float(a.mean()),variance=float(a.var()),std=float(a.std()),
                positive_probability=float((a > 0).mean()),
                quantiles={str(q):float(np.quantile(a,q)) for q in (0,.05,.25,.5,.75,.95,1)})


def centered_pair(x, y, floor):
    if x is None or y is None:
        return dict(correlation=None,covariance=None,reason="missing_clean_signal",repetitions=0)
    x, y = np.asarray(x,dtype=float), np.asarray(y,dtype=float)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pair weather-domain mismatch")
    a,b = x-math.fsum(x.tolist())/len(x),y-math.fsum(y.tolist())/len(y)
    vx,vy,cov = (math.fsum(v.tolist())/len(x) for v in (a*a,b*b,a*b))
    reason = "zero_or_near_zero_weather_variance" if min(vx,vy) <= floor else None
    return dict(correlation=max(-1.,min(1.,cov/math.sqrt(vx*vy))) if reason is None else None,
                covariance=cov,reason=reason,repetitions=len(x),variance_a=vx,variance_b=vy,
                centering="within this fixed target/timestamp/CAV over the finite weather pool")


def subtract(a, b):
    return a-b if a is not None and b is not None else None


def object_average(items, key):
    by_object = defaultdict(list)
    for r in items:
        by_object[r["object_id"]].append(r[key])
    return {oid:mean(values) for oid,values in sorted(by_object.items())}


def calculate(rows, plan, config, field_indices=None, details=True):
    """No weather-conditioned target filtering. Missingness is fixed by Clean."""
    if not rows:
        raise ValueError("No frozen target/CAV rows")
    size = plan["size"]
    subset = list(range(20)) if field_indices is None else list(field_indices)
    if len(subset) != size or len(set(subset)) != size:
        raise ValueError("Pool/subset identities do not align")
    keys = [(r["scene"],r["timestamp"],r["object_id"],r["cav"]) for r in rows]
    if len(set(keys)) != len(rows) or len({r["scene"] for r in rows}) != 1:
        raise ValueError("Duplicate rows or cross-scene identity contamination")
    groups = defaultdict(list)
    missing = Counter()
    marginal_errors = []
    for r in rows:
        if r["cav"] not in plan["cavs"] or r["ego"] != plan["ego"]:
            raise ValueError("Frozen CAV identity changed")
        if (r["R"] is None) != (r["fixed_weighted_clean_denominator"] <= 0):
            raise ValueError("Signal validity differs from its frozen Clean denominator")
        missing[r["missing_reason"] or "valid"] += 1
        for name in ("R","tau","T"):
            values = r[name]
            if values is not None:
                a = np.asarray(values,dtype=float)
                if a.shape != (20,) or not np.isfinite(a).all():
                    raise ValueError("Missing/nonfinite weather field in a fixed target/CAV row")
                marginal_errors.append(strict_balance(a[subset,None],plan,config["tolerance"]))
        if r["active"]:
            groups[(r["timestamp"],r["object_id"])].append(r)
    pairs, targets = [], []
    negative_errors, single_errors = [], []
    for (stamp,oid), values in sorted(groups.items()):
        cavs = sorted(r["cav"] for r in values)
        if plan["ego"] not in cavs or len(cavs) != len(set(cavs)):
            raise ValueError("Target group missing ego or has duplicate CAV")
        by_cav = {r["cav"]:r for r in values}
        if len({r["clean_gt_sha256"] for r in values}) != 1:
            raise ValueError("Target Clean GT identities differ")
        signals, depths = {}, {}
        for mode,schedules in plan["schedules"].items():
            signals[mode],depths[mode] = {},{}
            for c in cavs:
                ix = np.asarray(subset)[schedules[c]]
                signals[mode][c] = np.asarray(by_cav[c]["R"])[ix] if by_cav[c]["R"] is not None else None
                depths[mode][c] = np.asarray(by_cav[c]["tau"])[ix] if by_cav[c]["tau"] is not None else None
        for a,b in combinations(cavs,2):
            measured = {m:centered_pair(s[a],s[b],config["tolerance"]["variance_floor"]) for m,s in signals.items()}
            if (measured["E2"]["correlation"] is None) != (measured["E3"]["correlation"] is None):
                raise ValueError("Balanced pair marginal variances have inconsistent validity")
            for metric in ("correlation","covariance"):
                x,y = measured["E2"][metric],measured["E2_shift"][metric]
                if (x is None) != (y is None):
                    raise ValueError("Negative-control pair validity changed")
                if x is not None:
                    close(x,y,config["tolerance"])
                    negative_errors.append(abs(x-y))
            pairs.append(dict(scene=rows[0]["scene"],timestamp=stamp,object_id=oid,a=a,b=b,
                E2=measured["E2"],E3=measured["E3"],
                delta_correlation=subtract(measured["E2"]["correlation"],measured["E3"]["correlation"]),
                delta_covariance=subtract(measured["E2"]["covariance"],measured["E3"]["covariance"])))
        valid_neighbors = sorted(c for c in cavs if c != plan["ego"] and by_cav[c]["R"] is not None)
        valid_cavs = sorted(c for c in cavs if by_cav[c]["R"] is not None)
        ego_valid = by_cav[plan["ego"]]["R"] is not None
        result = {}
        for mode,s in signals.items():
            ego = s[plan["ego"]]
            best = np.max(np.stack([s[c] for c in valid_neighbors]),axis=0) if valid_neighbors else None
            C = best-ego if ego_valid and best is not None else None
            stack = np.stack([s[c] for c in valid_cavs]) if valid_cavs else None
            depth = np.stack([depths[mode][c] for c in valid_cavs]) if valid_cavs else None
            result[mode] = dict(C=distribution(C),C_values=C.tolist() if C is not None else None,
                ego_R=ego.tolist() if ego is not None else None,best_neighbor_R=best.tolist() if best is not None else None,
                across_cav_range=distribution(stack.max(axis=0)-stack.min(axis=0)) if stack is not None else distribution(None),
                across_cav_std=distribution(stack.std(axis=0)) if stack is not None else distribution(None),
                optical_depth_range=distribution(depth.max(axis=0)-depth.min(axis=0)) if depth is not None else distribution(None),
                per_cav_R={c:s[c].tolist() if s[c] is not None else None for c in cavs})
        for c in valid_cavs:
            shifted = np.asarray(plan["schedules"]["E2_shift"][c])
            if not np.array_equal(signals["E2_shift"][c],signals["E2"][c][shifted]):
                raise ValueError("Shared-field reorder changed joint weather realizations")
        for metric in ("mean","variance","std","positive_probability"):
            x,y = result["E2"]["C"][metric],result["E2_shift"]["C"][metric]
            if x is not None:
                close(x,y,config["tolerance"])
                negative_errors.append(abs(x-y))
        if result["E2"]["C_values"] is not None and not np.array_equal(
                np.sort(result["E2"]["C_values"]),np.sort(result["E2_shift"]["C_values"])):
            raise ValueError("Shared reorder changed the complete C distribution")
        for name in ("across_cav_range","across_cav_std","optical_depth_range"):
            for metric in ("mean","variance","std","positive_probability"):
                x,y = result["E2"][name][metric],result["E2_shift"][name][metric]
                if x is not None:
                    close(x,y,config["tolerance"])
        delta = subtract(result["E2"]["C"]["mean"],result["E3"]["C"]["mean"])
        if len(valid_neighbors) == 1 and ego_valid:
            if abs(delta) > config["tolerance"]["single_neighbor_delta_C"]:
                raise ValueError("Single-neighbor full-cycle mean delta C must equal zero")
            single_errors.append(abs(delta))
        targets.append(dict(scene=rows[0]["scene"],timestamp=stamp,object_id=oid,
            valid_neighbor_count=len(valid_neighbors),valid_neighbor_ids=valid_neighbors,
            missing_C_reason="ego_clean_signal_missing" if not ego_valid else "no_valid_neighbor" if not valid_neighbors else None,
            clean_by_cav={c:{k:by_cav[c][k] for k in ("clean_points","sampled_points","clean_signal","fixed_weighted_clean_denominator","distance_m","view_angle_rad","sample_fraction")} for c in cavs},
            E2=result["E2"],E3=result["E3"],delta_C=delta,
            delta_C_std=subtract(result["E2"]["C"]["std"],result["E3"]["C"]["std"]),
            delta_C_positive_probability=subtract(result["E2"]["C"]["positive_probability"],result["E3"]["C"]["positive_probability"])))
    objects = {metric:object_average(pairs if metric == "delta_correlation" else targets,metric) for metric in METRICS}
    metrics = {metric:mean(v.values()) for metric,v in objects.items()}
    valid_pairs = [p for p in pairs if p["delta_correlation"] is not None]
    object_counts = Counter(p["object_id"] for p in valid_pairs)
    object_effects = [abs(x) for x in objects["delta_correlation"].values() if x is not None]
    effect_total = math.fsum(object_effects)
    counts = dict(target_instances=len(targets),unique_objects=len({r["object_id"] for r in targets}),
        valid_target_instances=sum(t["delta_C"] is not None for t in targets),
        valid_unique_objects=len({t["object_id"] for t in targets if t["delta_C"] is not None}),
        candidate_target_cav_pairs=len(pairs),valid_target_cav_pairs=len(valid_pairs),
        invalid_pair_fraction=1-len(valid_pairs)/len(pairs) if pairs else 1.,
        missing_R=dict(missing),invalid_pair_reasons=dict(Counter(p["E2"]["reason"] or "valid" for p in pairs)),
        missing_C=dict(Counter(t["missing_C_reason"] or "valid" for t in targets)),
        valid_neighbor_counts=dict(Counter(str(t["valid_neighbor_count"]) for t in targets)),
        maximum_object_pair_share=max(object_counts.values(),default=0)/len(valid_pairs) if valid_pairs else None,
        maximum_object_abs_correlation_effect_share=max(object_effects,default=0)/effect_total if effect_total else None)
    repeats = {}
    for mode in ("E2","E3"):
        repeats[mode] = {str(j):mean(t[mode]["C_values"][j] for t in targets if t[mode]["C_values"] is not None) for j in range(size)}
    subgroups = {}
    for label,predicate in (("one_neighbor",lambda n:n==1),("multiple_neighbors",lambda n:n>=2)):
        selected = [t for t in targets if predicate(t["valid_neighbor_count"])]
        subgroups[label] = dict(target_instances=len(selected),metrics={m:mean(object_average(selected,m).values()) for m in METRICS if m != "delta_correlation"},
            E2_C_mean=mean(t["E2"]["C"]["mean"] for t in selected),E3_C_mean=mean(t["E3"]["C"]["mean"] for t in selected),
            E2_C_positive_probability=mean(t["E2"]["C"]["positive_probability"] for t in selected),
            E3_C_positive_probability=mean(t["E3"]["C"]["positive_probability"] for t in selected),
            note="single-neighbor mean delta C is constrained to zero by balance; distributional changes remain meaningful")
    out = dict(scene=rows[0]["scene"],partition=rows[0]["partition"],pool_size=size,counts=counts,metrics=metrics,
        per_object_metrics=objects,subgroups=subgroups,within_pool_repeat_C_means=repeats,
        within_pool_C_mean_distribution={m:distribution([v for v in repeats[m].values() if v is not None]) if any(v is not None for v in repeats[m].values()) else distribution(None) for m in repeats},
        balance=dict(status="passed",fixed_target_CAV_marginals_exact=True,mean_max_error=max(marginal_errors,default=0.)),
        negative_control=dict(status="passed",maximum_error=max(negative_errors,default=0.)),
        single_neighbor_check=dict(status="passed",targets=len(single_errors),maximum_abs_delta_C=max(single_errors,default=0.)),
        weighting="per-object mean over its frames/pairs, then equal-object scene mean; road scene is the independent unit")
    if details:
        out.update(pairs=pairs,targets=targets)
    return out


def analyze_scene(rows, plan, config):
    result = calculate(rows,plan,config)
    leave_field = []
    if len(plan["cavs"]) <= 19:
        smaller = assignment(plan["cavs"],plan["ego"],19,config["negative_control_shift"])
        for removed in range(20):
            item = calculate(rows,smaller,config,[j for j in range(20) if j != removed],details=False)
            leave_field.append(dict(omitted_field=removed,metrics=item["metrics"]))
    result["leave_one_field"] = dict(status="available" if leave_field else "unavailable_20_CAVs_cannot_fit_19_fields",
        scope="auxiliary balanced 19-field sensitivity only; primary always retains all20; no new fields or outcome-based omission",
        coupling="reindex remaining fields with the frozen identity-rank offsets modulo19; this also changes E3 coupling",results=leave_field)
    result["leave_one_object"] = {m:{oid:mean(v for k,v in values.items() if k != oid)
                                     for oid in values} for m,values in result["per_object_metrics"].items()}
    return result

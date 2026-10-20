"""Scene-level physical diagnostics and paired E2/E3 comparisons, no AP."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
import random
import statistics

from shared_weather_benchmark.provenance import read_json, atomic_json, atomic_text, seed_for, identity
from shared_weather_benchmark.generator import make_field
from shared_weather_benchmark.ray_integrator import integrate, integrate_numpy


def mean(values):
    values = [float(x) for x in values if x is not None and math.isfinite(x)]
    return statistics.fmean(values) if values else None


def correlation(xs,ys):
    pairs = [(a,b) for a,b in zip(xs,ys) if a is not None and b is not None]
    if len(pairs) < 3:
        return None
    x,y = zip(*pairs)
    mx,my = mean(x),mean(y)
    xx,yy = sum((v-mx)**2 for v in x),sum((v-my)**2 for v in y)
    return sum((a-mx)*(b-my) for a,b in pairs)/math.sqrt(xx*yy) if xx*yy > 1e-20 else None


def paired_scene_interval(deltas,config):
    if len(deltas) < config["minimum_scenes_for_interval"]:
        return dict(mean=mean(deltas),ci95=None,status="insufficient_scenes_descriptive_only",scenes=len(deltas))
    rng = random.Random(config["seed"])
    samples = sorted(mean(rng.choices(deltas,k=len(deltas))) for _ in range(config["replicates"]))
    return dict(mean=mean(deltas),ci95=[samples[int(.025*len(samples))],samples[int(.975*len(samples))]],
                scenes=len(deltas),unit="paired scene resampling; continuous frames not independent")


def field_diagnostics(config,selection,domains):
    import numpy as np
    by_scene = {}
    tolerance = config["integration"]["sample_error_tolerance"]
    for scene in sorted(domains):
        frame = next(f for f in selection["frames"] if f["scene"] == scene)
        low, high = np.array(domains[scene][:3]),np.array(domains[scene][3:])
        rng = random.Random(seed_for(config["seed"],"fixed-field-probes",scene))
        probes = np.array([[rng.uniform(a,b) for a,b in zip(low,high)] for _ in range(256)])
        scene_data = {}
        shared_arrays = []
        for mode in ("E1","E2","E3"):
            values, covariances, errors, reverse_errors, hashes, probe_hashes = [], [], [], [], [], []
            seen = set()
            for cav in frame["cavs"]:
                field = make_field(config,scene,cav,mode,domains[scene])
                hashes.append(field.manifest()["sha256"])
                alpha = field.alpha_numpy(probes)
                probe_hashes.append(identity(alpha.tolist()))
                scalar = np.array([field.alpha(p.tolist()) for p in probes[:16]])
                if not np.allclose(alpha[:16],scalar,atol=1e-12,rtol=0):
                    raise ValueError("Scalar/batch field disagreement")
                if mode == "E2":
                    shared_arrays.append(alpha)
                if field.manifest()["sha256"] in seen:
                    continue
                seen.add(field.manifest()["sha256"])
                values.extend(alpha.tolist())
                # Measured correlation of the alpha field, not an asserted exact length.
                curve = []
                for lag in [0,5,10,25,50,100]:
                    starts = probes.copy()
                    starts[:,0] = low[0]+(probes[:,0]-low[0])/(high[0]-low[0])*max(high[0]-low[0]-lag,0)
                    if lag > high[0]-low[0]:
                        curve.append(dict(lag_m=lag,correlation=None)); continue
                    ends = starts.copy(); ends[:,0] += lag
                    curve.append(dict(lag_m=lag,correlation=correlation(field.alpha_numpy(starts),field.alpha_numpy(ends))))
                covariances.append(curve)
                # Numeric ray check on fixed world segments, independent of object/model.
                start = (low+high)/2
                end = probes[:8]
                coarse = integrate_numpy(field,start,end,config["integration"]["step_m"])
                fine = integrate_numpy(field,start,end,config["integration"]["step_m"]/2)
                error = float(np.max(np.abs(coarse-fine)))
                errors.append(error)
                rev = abs(integrate(field,start.tolist(),end[0].tolist(),.5)-integrate(field,end[0].tolist(),start.tolist(),.5))
                reverse_errors.append(rev)
                if error > tolerance or rev > 1e-9:
                    raise ValueError("Real-domain numerical path convergence/reversal gate failed")
            scene_data[mode] = dict(alpha_mean=mean(values),alpha_std=statistics.pstdev(values),
                alpha_quantiles=np.quantile(values,[0,.1,.5,.9,1]).tolist(),
                field_hashes=hashes,correlation_curves=covariances,
                fixed_probe_alpha_sha256=probe_hashes,
                nominal_latent_length_m=config["field"]["length_m"],
                max_step_halving_optical_depth_error=max(errors),max_reversal_error=max(reverse_errors))
        if any(not np.array_equal(shared_arrays[0],a) for a in shared_arrays[1:]):
            raise ValueError("Shared world position returned different alpha across CAVs")
        scene_data["shared_point_cross_CAV_max_error"] = 0.
        by_scene[scene] = scene_data
    return by_scene


def aggregate_clouds(records, bin_count):
    values = [r["statistics"] for r in records]
    total = sum(r["clean"] for r in values)
    hist = lambda key: [sum(r[key][i] for r in values) for i in range(len(values[0][key]))]
    bins = []
    for i in range(bin_count):
        clean = sum(r["bins"][i]["clean"] for r in values)
        hard = sum(r["bins"][i]["retained_hard"] for r in values)
        bins.append(dict(clean=clean,retained_hard=hard,retention=hard/clean if clean else None,
            mean_integral=sum(r["bins"][i]["integral_sum"] for r in values)/clean if clean else None,
            mean_transmittance=sum(r["bins"][i]["t_sum"] for r in values)/clean if clean else None))
    emitted = sum(r["emitted"] for r in values)
    return dict(clouds=len(records),clean_points=total,emitted=emitted,
        hard_retention=sum(r["retained_hard"] for r in values)/total,
        output_count_ratio=emitted/total,simulated_fraction=sum(r["simulated"] for r in values)/emitted if emitted else None,
        range_anomaly_fraction=sum(r["range_anomalies"] for r in values)/emitted if emitted else None,
        mean_integral=sum(r["mean_integral"]*r["clean"] for r in values)/total,
        mean_transmittance=sum(r["mean_transmittance"]*r["clean"] for r in values)/total,
        clean_intensity_histogram=hist("clean_intensity_histogram"),weather_intensity_histogram=hist("weather_intensity_histogram"),
        bins=bins)


def compare(config,output):
    manifest = read_json(output/"DATA_MANIFEST.json")
    records = [read_json(output/p) for p in manifest["records"]]
    targets = read_json(output/"OBSERVABILITY_TARGETS.json")
    observations = [json.loads(line) for line in (output/"OBSERVABILITY_ROWS.jsonl").read_text("utf-8").splitlines()]
    selection = read_json(output/"SELECTION.json")
    fields = field_diagnostics(config,selection,read_json(output/"FIELD_DOMAINS.json"))
    scenes = {}
    for scene in sorted(fields):
        modes = {}
        for mode in config["protocols"]:
            selected = [r for r in targets if r["scene"] == scene and r["mode"] == mode]
            clouds = [r for r in records if r["scene"] == scene and r["mode"] == mode]
            scope = [r for r in observations if r["scene"] == scene and r["mode"] == mode and r["active"]]
            pairs = defaultdict(list)
            for r in scope:
                pairs[r["cav"]].append(r)
            corr = {}
            cavs = sorted(pairs)
            for i,a in enumerate(cavs):
                aa = {(r["timestamp"],r["object_id"]):r for r in pairs[a]}
                for b in cavs[i+1:]:
                    bb = {(r["timestamp"],r["object_id"]):r for r in pairs[b]}
                    keys = sorted(set(aa)&set(bb))
                    corr[a+"/"+b] = dict(paired_objects=len(keys),
                        retention_correlation=correlation([aa[k]["retention_ratio"] for k in keys],[bb[k]["retention_ratio"] for k in keys]),
                        effective_intensity_correlation=correlation([aa[k]["effective_intensity_ratio"] for k in keys],[bb[k]["effective_intensity_ratio"] for k in keys]))
            sensitivity = {}
            for key in selected[0]["sensitivity"]:
                sensitivity[key] = dict(Counter(r["sensitivity"][key] for r in selected))
            modes[mode] = dict(physical=aggregate_clouds(clouds,len(config["distance_bins_m"])-1),
                targets=len(selected),categories=dict(Counter(r["category"] for r in selected)),
                sensitivity=sensitivity,
                all_zero_fraction=mean([r["all_zero_hard_returns"] for r in selected]),
                ego_zero_neighbor_positive_fraction=mean([r["ego_zero_neighbor_positive"] for r in selected]),
                all_zero_clean_fraction=mean([r["all_zero_clean_returns"] for r in selected]),
                mean_effective_ratio=mean([r["effective_intensity_ratio"] for r in scope]),
                mean_ego_best_neighbor_gap=mean([r["ego_best_neighbor_intensity_gap"] for r in selected]),
                cross_cav_correlations=corr)
        e2,e3 = modes["E2"]["physical"],modes["E3"]["physical"]
        alpha_delta = fields[scene]["E2"]["alpha_mean"]-fields[scene]["E3"]["alpha_mean"]
        retention_delta = e2["hard_retention"]-e3["hard_retention"]
        bin_delta = [(a["retention"]-b["retention"]) if a["retention"] is not None and b["retention"] is not None else None
                     for a,b in zip(e2["bins"],e3["bins"])]
        t_delta = e2["mean_transmittance"]-e3["mean_transmittance"]
        bin_t_delta = [(a["mean_transmittance"]-b["mean_transmittance"]) if a["mean_transmittance"] is not None and b["mean_transmittance"] is not None else None
                       for a,b in zip(e2["bins"],e3["bins"])]
        matched = (abs(alpha_delta) <= config["matching"]["alpha_mean_absolute_tolerance"]
                   and abs(retention_delta) <= config["matching"]["retention_absolute_tolerance"]
                   and all(abs(x) <= config["matching"]["distance_bin_absolute_tolerance"] for x in bin_delta if x is not None)
                   and abs(t_delta) <= config["matching"]["transmittance_absolute_tolerance"]
                   and all(abs(x) <= config["matching"]["distance_bin_transmittance_tolerance"] for x in bin_t_delta if x is not None))
        scenes[scene] = dict(partition=next(f["partition"] for f in selection["frames"] if f["scene"] == scene),
            protocols=modes,matching=dict(alpha_mean_delta=alpha_delta,retention_delta=retention_delta,
                distance_bin_delta=bin_delta,finite_realization_matched=matched,
                mean_transmittance_delta=t_delta,distance_bin_transmittance_delta=bin_t_delta,
                mean_integral_delta=e2["mean_integral"]-e3["mean_integral"],
                caveat="attenuation-only count matching is trivial; optical depth/energy can differ; no AP or tuning"))
    comparisons = {}
    for partition in config["partitions"][:2]:
        part = [s for s in scenes.values() if s["partition"] == partition]
        for metric in ("mean_effective_ratio","mean_ego_best_neighbor_gap","all_zero_fraction","ego_zero_neighbor_positive_fraction"):
            deltas = [s["protocols"]["E2"][metric]-s["protocols"]["E3"][metric] for s in part
                      if s["protocols"]["E2"][metric] is not None and s["protocols"]["E3"][metric] is not None]
            comparisons[partition+"/"+metric] = paired_scene_interval(deltas,config["bootstrap"])
    result = dict(status="complete_descriptive_prototype",scenes=scenes,fields=fields,
                  paired_scene_comparisons=comparisons,config_matching=config["matching"],
                  physical_realism_proven=False,detection_evaluated=False,
                  e0_e1_scope="compare E1 to analytic/Hahner hard component before rounding; E0 includes soft replacement",
                  marginal_weather_law_equal=True,realized_physical_statistics_equal=False)
    atomic_json(output/"COMPARISON.json",result)
    lines = ["# E0/E1/E2/E3 物理与观测审计", "", "统计按scene分组；未运行检测。所有数值均为当前生成器下的诊断。", "",
             "| scene | partition | mode | hard保留率 | 平均目标强度比例 | ego-best邻车差 | 全车零回波比例 |", "|---|---|---|---:|---:|---:|---:|"]
    for scene, data in scenes.items():
        for mode, v in data["protocols"].items():
            lines.append(f"| {scene} | {data['partition']} | {mode} | {v['physical']['hard_retention']:.6g} | {v['mean_effective_ratio']} | {v['mean_ego_best_neighbor_gap']} | {v['all_zero_fraction']} |")
        lines += ["", f"{scene} E2/E3 匹配残差：`{json.dumps(data['matching'],ensure_ascii=False)}`", ""]
    lines += ["默认衰减模式保留点坐标和行数，重点比较强度/透射率/跨车差异。RGB量化后的统计另存每云record。",
              "E0有散射替换；E1无散射，不能用E0/E1差异归因空间结构。", "",
              "样本不足5个scene时不报bootstrap区间。阈值敏感性、逐CAV连续指标、相关性曲线与匹配明细见JSON。",
              "匹配失败保留原始结果，标记不可作已匹配因果比较；不自动调参。", ""]
    atomic_text(output/"E0_E1_E2_E3_COMPARISON.md","\n".join(lines))
    return result

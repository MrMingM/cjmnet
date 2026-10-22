"""All-scene comparisons: seeds nested in scenes, residuals never filtered out."""
from __future__ import annotations

from collections import defaultdict
import math
import random
import statistics

from shared_weather_benchmark_p6.core import mean, stats, bin_name, atomic_json, atomic_text, write_csv


def subtract(a,b):
    return a-b if a is not None and b is not None else None


def matching_details(e2,e3,alpha2,alpha3,tolerance):
    delta_alpha = alpha2-alpha3
    delta_t = e2["mean_transmittance"]-e3["mean_transmittance"]
    delta_ret = e2["hard_retention"]-e3["hard_retention"]
    bins_t = [subtract(a["mean_transmittance"],b["mean_transmittance"]) for a,b in zip(e2["bins"],e3["bins"])]
    bins_ret = [subtract(a["retention"],b["retention"]) for a,b in zip(e2["bins"],e3["bins"])]
    if len(e2["bins"]) != len(e3["bins"]):
        raise ValueError("Distance-bin schema differs")
    checks = {}
    def check(name,value,limit):
        checks[name] = dict(delta=value,tolerance=limit,passed=abs(value) <= limit if value is not None else None,
                            status="empty_bin" if value is None else "tested")
    check("alpha_mean",delta_alpha,tolerance["alpha_mean_absolute_tolerance"])
    check("mean_transmittance",delta_t,tolerance["transmittance_absolute_tolerance"])
    check("raw_retention",delta_ret,tolerance["retention_absolute_tolerance"])
    for i,(a,b) in enumerate(zip(e2["bins"],e3["bins"])):
        if a["clean"] != b["clean"]:
            raise ValueError("E2/E3 original point denominator differs in a distance bin")
        if a["clean"] > 0 and (bins_t[i] is None or bins_ret[i] is None):
            raise ValueError("Nonempty distance bin has missing statistics")
        check(f"distance_bin_T/{i}",bins_t[i],tolerance["distance_bin_transmittance_tolerance"])
        check(f"distance_bin_retention/{i}",bins_ret[i],tolerance["distance_bin_absolute_tolerance"])
    failures = [k for k,v in checks.items() if v["passed"] is False]
    return dict(alpha_delta=delta_alpha,mean_transmittance_delta=delta_t,retention_delta=delta_ret,
        mean_optical_depth_delta=e2["mean_integral"]-e3["mean_integral"],
        distance_bin_transmittance_delta=bins_t,distance_bin_retention_delta=bins_ret,
        checks=checks,failed_conditions=failures,matched=not failures,
        count_matching_discriminative=False,
        interpretation="descriptive; unmatched results remain included; matching is not proof of causal isolation")


def scene_seed_results(selection, repeat, metrics, physical, fields, config):
    results = []
    for scene in sorted(metrics["by_scene"]):
        modes = metrics["by_scene"][scene]
        e2,e3 = physical[scene]["E2"],physical[scene]["E3"]
        match = matching_details(e2,e3,fields[scene]["E2"]["alpha_mean"],fields[scene]["E3"]["alpha_mean"],config["matching"])
        p = metrics["pair_analysis"].get(scene,{})
        correlations = {m:p.get(m,{}).get("all",{}).get("R_correlation",{}).get("value") for m in ("E2","E3")}
        strata = {}
        for label in sorted(set(modes["E2"]["C_by_stratum"]) | set(modes["E3"]["C_by_stratum"])):
            a = modes["E2"]["C_by_stratum"].get(label,dict(mean=None,valid=0))
            b = modes["E3"]["C_by_stratum"].get(label,dict(mean=None,valid=0))
            strata[label] = dict(delta_C=subtract(a["mean"],b["mean"]),E2=a,E3=b)
        result = dict(scene=scene,partition=modes["E2"]["partition"],repeat_index=repeat,
            frames=sum(f["scene"] == scene for f in selection["frames"]),
            cav_count=len(next(f["cavs"] for f in selection["frames"] if f["scene"] == scene)),
            E1=modes["E1"],E2=modes["E2"],E3=modes["E3"],matching=match,
            delta_C=subtract(modes["E2"]["C"]["mean"],modes["E3"]["C"]["mean"]),
            delta_R=subtract(modes["E2"]["R"]["mean"],modes["E3"]["R"]["mean"]),
            delta_intensity_weighted_T=subtract(modes["E2"]["intensity_weighted_T"],modes["E3"]["intensity_weighted_T"]),
            delta_pair_R_correlation=subtract(correlations["E2"],correlations["E3"]),
            delta_weather_added_absolute_gap=subtract(modes["E2"]["weather_added_absolute_gap"]["mean"],modes["E3"]["weather_added_absolute_gap"]["mean"]),
            E2_minus_E1_C=subtract(modes["E2"]["C"]["mean"],modes["E1"]["C"]["mean"]),
            E3_minus_E1_C=subtract(modes["E3"]["C"]["mean"],modes["E1"]["C"]["mean"]),
            field_cross_cav=fields[scene]["cross_cav"],strata=strata,
            no_matching_or_seed_selection=True)
        results.append(result)
    return results


def scene_interval(values,config):
    """values must contain at most one seed-averaged value per scene."""
    valid = {s:v for s,v in values.items() if v is not None}
    out = dict(total_scenes=len(values),valid_scenes=len(valid),missing_scenes=sorted(set(values)-set(valid)),
               mean=mean(valid.values()),ci95=None,unit="scene; all seeds/frames remain nested",
               status="insufficient_scenes_descriptive_only")
    if len(valid) >= config["minimum_scenes"]:
        rng = random.Random(config["seed"])
        observed = list(valid.values())
        boots = sorted(mean(rng.choices(observed,k=len(observed))) for _ in range(config["replicates"]))
        out.update(ci95=[boots[int(.025*len(boots))],boots[int(.975*len(boots))]],
                   status="exploratory_scene_bootstrap; no population/power guarantee")
    return out


def nested_summary(rows,metric,config):
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(row[metric])
    scenes = {s:dict(stats(v),seed_values=v) for s,v in sorted(by_scene.items())}
    aggregate = scene_interval({s:v["mean"] for s,v in scenes.items()},config)
    direction = 1 if (aggregate["mean"] or 0) > 0 else -1 if (aggregate["mean"] or 0) < 0 else 0
    valid_scene_means = [v["mean"] for v in scenes.values() if v["mean"] is not None]
    all_values = [v for values in by_scene.values() for v in values if v is not None]
    return dict(scenes=scenes,aggregate=aggregate,
        scene_direction_fraction=mean([float(v*direction > 0) for v in valid_scene_means]) if direction else None,
        seed_direction_fraction=mean([float(v*direction > 0) for v in all_values]) if direction else None,
        scene_between_std=statistics.pstdev(valid_scene_means) if valid_scene_means else None,
        count_scene_seed_results=len(rows),independent_units=len(by_scene))


def adjusted_regression(rows,bootstrap):
    """Exploratory scene-level OLS; does not eliminate physical confounding."""
    import numpy as np
    groups = defaultdict(list)
    for r in rows:
        groups[r["scene"]].append(r)
    data = [[mean(r["delta_C"] for r in rr),mean(r["matching"]["mean_transmittance_delta"] for r in rr),
             mean(r["matching"]["alpha_delta"] for r in rr)] for rr in groups.values()]
    valid = [r for r in data if all(v is not None for v in r)]
    out = dict(scene_count=len(valid),status="unsupported",coefficients=None,intercept_ci95=None,
               caveat="exploratory adjustment of global means; residual physical confounding remains")
    if len(valid) < bootstrap["minimum_scenes"]:
        return out
    a = np.asarray(valid)
    def fit(ix):
        x = np.column_stack((np.ones(len(ix)),a[ix,1:]))
        if np.linalg.matrix_rank(x) < 3:
            return None
        return np.linalg.lstsq(x,a[ix,0],rcond=None)[0].tolist()
    coefficients = fit(list(range(len(a))))
    if coefficients is None:
        return dict(out,status="unsupported_rank_deficient")
    rng = random.Random(bootstrap["seed"])
    fits = [fit(rng.choices(range(len(a)),k=len(a))) for _ in range(bootstrap["replicates"])]
    intercepts = sorted(v[0] for v in fits if v is not None)
    out.update(status="exploratory",coefficients=dict(intercept=coefficients[0],delta_T=coefficients[1],delta_alpha=coefficients[2]),
               valid_bootstrap_resamples=len(intercepts),requested_bootstrap_resamples=len(fits))
    if len(intercepts) >= .9*len(fits):
        out["intercept_ci95"] = [intercepts[int(.025*len(intercepts))],intercepts[int(.975*len(intercepts))]]
    return out


def summarize_repeats(rows,config,expected_scenes,expected_repeats):
    pairs = [(r["scene"],r["repeat_index"]) for r in rows]
    expected = {(s,k) for s in expected_scenes for k in expected_repeats}
    if set(pairs) != expected or len(set(pairs)) != len(rows):
        raise ValueError("Scene x repeat inventory incomplete or duplicated; no redraw or omission allowed")
    summaries = {}
    for partition in [*config["scene_counts"],"all_scenes"]:
        selected = rows if partition == "all_scenes" else [r for r in rows if r["partition"] == partition]
        residual_groups = defaultdict(list)
        for r in selected:
            residual_groups[bin_name(r["matching"]["mean_transmittance_delta"],config["transmittance_residual_bins"])].append(r)
        near = [r for r in selected if abs(r["matching"]["mean_transmittance_delta"]) <= config["decision"]["balanced_transmittance_absolute_delta"]]
        strata = {}
        labels = sorted({label for r in selected for label in r["strata"]})
        for label in labels:
            data = [dict(scene=r["scene"],delta_C=r["strata"].get(label,{}).get("delta_C")) for r in selected]
            strata[label] = nested_summary(data,"delta_C",config["bootstrap"])
        summaries[partition] = dict(scene_count=len({r["scene"] for r in selected}),
            scene_seed_count=len(selected),matched_scene_seed_count=sum(r["matching"]["matched"] for r in selected),
            matched_fraction=mean([float(r["matching"]["matched"]) for r in selected]),
            all_three_seeds_matched_scenes=[s for s in sorted({r["scene"] for r in selected})
                if all(r["matching"]["matched"] for r in selected if r["scene"] == s)],
            metrics={m:nested_summary(selected,m,config["bootstrap"]) for m in
                ("delta_C","delta_R","delta_intensity_weighted_T","delta_pair_R_correlation","delta_weather_added_absolute_gap")},
            C_by_stratum=strata,
            C_by_global_transmittance_residual={k:nested_summary(v,"delta_C",config["bootstrap"]) for k,v in sorted(residual_groups.items())},
            near_balanced_exploratory=nested_summary(near,"delta_C",config["bootstrap"]),
            matched_only_exploratory=nested_summary([r for r in selected if r["matching"]["matched"]],"delta_C",config["bootstrap"]),
            matched_subset_caveat="post-generation subset; not primary; selection can bias conclusions",
            regression_adjustment=adjusted_regression(selected,config["bootstrap"]))
    return dict(status="complete_exploratory_signal_proxy_comparison",partitions=summaries,
        all_scene_seed_results_included=True,scene_seed_results=rows,
        unmatched=[dict(scene=r["scene"],repeat_index=r["repeat_index"],partition=r["partition"],
                         failed_conditions=r["matching"]["failed_conditions"],matching=r["matching"])
                   for r in rows if not r["matching"]["matched"]],
        no_detection_or_causal_effect_claim=True)


def decision(result,config,smoke):
    v = result["partitions"]["physics_validation"]
    d = config["decision"]
    c = v["metrics"]["delta_C"]
    corr = v["metrics"]["delta_pair_R_correlation"]
    def nonzero(metric,margin):
        ci = metric["aggregate"]["ci95"]
        value = metric["aggregate"]["mean"]
        return bool(ci and value is not None and abs(value) >= margin and ci[0]*ci[1] > 0)
    adequate = not smoke and v["scene_count"] >= d["minimum_validation_scenes"]
    matched = adequate and (v["matched_fraction"] or 0) >= d["minimum_matching_fraction"]
    stable = ((c["scene_direction_fraction"] or 0) >= d["minimum_direction_fraction"]
              and (c["seed_direction_fraction"] or 0) >= d["minimum_direction_fraction"])
    balanced = v["near_balanced_exploratory"]
    persistent = (balanced["aggregate"]["valid_scenes"] >= d["minimum_balanced_validation_scenes"]
                  and nonzero(balanced,d["minimum_abs_C_effect"])
                  and (balanced["aggregate"]["mean"] or 0)*(c["aggregate"]["mean"] or 0) > 0)
    transfer = nonzero(corr,d["minimum_abs_correlation_effect"])
    signal = nonzero(c,d["minimum_abs_C_effect"])
    claims = dict(A_comparable_degradation=matched,B_field_design=True,C_signal_correlation_transfer=transfer,
                  D_neighbor_ratio_advantage_change=signal,E_scene_seed_direction_stability=stable,
                  F_persists_in_low_global_residual_stratum=persistent,
                  G_weather_changes_beyond_clean_ratio_baseline=signal)
    statuses = {k:"supported_exploratory" if adequate and val else "unsupported" for k,val in claims.items()}
    statuses["B_field_design"] = "supported_by_verified_construction_and_probe_checks"
    out = "INCONCLUSIVE"
    if adequate and all(claims.values()):
        out = "CONTINUE"
    elif matched:
        ci,corci = c["aggregate"]["ci95"],corr["aggregate"]["ci95"]
        if (ci and corci and max(map(abs,ci)) < d["minimum_abs_C_effect"]
                and max(map(abs,corci)) < d["minimum_abs_correlation_effect"]):
            out = "STOP_OR_REDESIGN"
    return dict(decision=out,smoke=smoke,claims=statuses,rules=d,
        interpretation="project triage using declared heuristic margins; no calibrated power/equivalence or detector claim",
        effect_direction_required="either positive or negative; E2 superiority is not a requirement",
        G_caveat="Clean R=1 and ratio C=0 by definition when valid; absolute Clean/weather gaps and E1 are also reported; this is not independent proof of cooperative benefit",
        F_caveat="stratification/OLS cannot prove complete elimination of physical confounding",
        detector_training_started=False)


def write_results(output,result,config,smoke):
    rows = result["scene_seed_results"]
    atomic_json(output/"P6C_SIGNAL_COMPLEMENTARITY.json",result)
    atomic_json(output/"P6C_MATCHING.json",dict(results=[dict(scene=r["scene"],repeat_index=r["repeat_index"],partition=r["partition"],
        **r["matching"],target_intensity_weighted_T_delta=r["delta_intensity_weighted_T"],actual_signal_R_delta=r["delta_R"])
        for r in rows],all_results_included=True))
    flat = [dict(scene=r["scene"],partition=r["partition"],repeat_index=r["repeat_index"],cav_count=r["cav_count"],
        delta_C=r["delta_C"],delta_R=r["delta_R"],delta_pair_R_correlation=r["delta_pair_R_correlation"],
        delta_intensity_weighted_T=r["delta_intensity_weighted_T"],**r["matching"]) for r in rows]
    write_csv(output/"P6C_SCENE_SEED_RESULTS.csv",flat)
    lines = ["# P6-C 全部场景和seed", "", "SMOKE："+str(smoke)+"；主单位scene，先对其固定seed求均值。未运行检测。", "",
             "| scene | partition | repeat | ΔC (E2−E3) | ΔR | ΔT | matched |", "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['scene']} | {r['partition']} | {r['repeat_index']} | {r['delta_C']} | {r['delta_R']} | {r['matching']['mean_transmittance_delta']} | {r['matching']['matched']} |")
    lines += ["", "全部结果是主结果；匹配子集与近似平衡子集仅为探索性分析，覆盖scene/seed数均在JSON中报告。", "",
              "C>0仅表示邻车相对强度保留比例更高。绝对Clean/Weather能量、有效邻车数分层、阈值和量化统计见各repeat目录。",
              "E1可因距离不同产生跨车差异；(C2−C1)−(C3−C1)=C2−C3是代数恒等式，不是额外独立证据。", ""]
    atomic_text(output/"P6C_ALL_SCENES_RESULTS.md","\n".join(lines))
    atomic_text(output/"P6C_UNMATCHED_ANALYSIS.md","# 未匹配结果\n\n全部未匹配scene/seed均保留；它们不能被解释为已经隔离空间关联的因果比较。\n\n"+
        "\n".join(f"- {r['scene']} repeat={r['repeat_index']}: {r['failed_conditions']}" for r in result["unmatched"])+
        "\n\n按总体透射率残差分层与scene级回归调整见SIGNAL_COMPLEMENTARITY；调整不能保证消除目标级物理混杂。\n")
    verdict = decision(result,config,smoke)
    atomic_json(output/"SUMMARY.json",dict(status="complete",smoke=smoke,independent_scenes=len({r["scene"] for r in rows}),
        scene_seed_pairs=len(rows),matched_pairs=sum(r["matching"]["matched"] for r in rows),decision=verdict,
        physical_realism_proven=False,detection_evaluated=False))
    atomic_text(output/"NEXT_STAGE_DECISION.md","# 下一阶段项目判断\n\n"+verdict["decision"]+"\n\n"+
        "判据为运行前声明的项目启发式，未作功效/传感器标定，不要求E2优于E3。\n\n"+
        "CONTINUE：正式8个验证scene，足够退化匹配、目标信号相关性变化、C效应与scene/seed方向稳定，并在低总体残差层有同方向信号；只建议未来AttFuse实验，不自动启动。\n\n"+
        "INCONCLUSIVE：样本/有效配对不足、seed波动、匹配质量或混杂解释仍不足。\n\n"+
        "STOP_OR_REDESIGN：匹配充分时，C与跨车信号相关性差异的探索区间都处于预声明的小效应范围；仅针对本版参数和代理指标。\n\n"+
        "逐项证据：\n\n"+"\n".join(f"- {k}: {v}" for k,v in verdict["claims"].items())+"\n")
    atomic_text(output/"RESULTS.md","# SCOR-W P6\n\n"+verdict["decision"]+"；SMOKE="+str(smoke)+
        "。完整逐scene/seed结果见P6C_ALL_SCENES_RESULTS.md，匹配残差和未匹配样本全部保留。\n\n"+
        "本轮只测signal observability proxy，不测detector recoverability，不证明实雾真实性。\n")
    return verdict

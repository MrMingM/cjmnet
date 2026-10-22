"""All-scene score-sorted AP and whole-scene uncertainty; never pool repetitions as scenes."""
from collections import defaultdict
import random
import numpy as np

from scorw_r3.core import Blocked,atomic_json,atomic_text,quantiles
from scorw_r3.metrics import ap


def group_ap(scenes,view,mode,repeat,scope,threshold=.7):
    rows=[]
    for s in scenes:
        for r in s["runs"]:
            if (r["view"],r["mode"],r["repeat"])==(view,mode,repeat) and scope in r["predictions"]:
                rows.append(r["predictions"][scope]["matches"][str(threshold)])
    return ap(rows)


def global_table(scenes):
    keys=sorted({(r["view"],r["mode"],r["repeat"],scope) for s in scenes for r in s["runs"] for scope in r["predictions"]})
    out=[]
    for view,mode,repeat,scope in keys:
        rows=[r["predictions"][scope]["matches"] for s in scenes for r in s["runs"]
              if (r["view"],r["mode"],r["repeat"])==(view,mode,repeat) and scope in r["predictions"]]
        out.append(dict(view=view,mode=mode,repeat=repeat,scope=scope,
            AP={"AP%d"%int(t*100):ap([r[str(t)] for r in rows]) for t in (.3,.5,.7)},
            counts={str(t):{key:sum(r[str(t)][key] for r in rows) for key in ("TP","FP","FN","gt")} for t in (.3,.5,.7)},
            scene_count=len(rows),definition="global score sorting over the same fixed scenes at this repeat; not mean scene AP"))
    return out


def delta(scenes,view):
    values=[]
    for j in range(20):
        b3=group_ap(scenes,view,"B3",j,"fusion");b4=group_ap(scenes,view,"B4",j,"fusion")
        e3=group_ap(scenes,view,"B3",j,"ego");e4=group_ap(scenes,view,"B4",j,"ego")
        if any(v is None for v in (b3,b4,e3,e4)):continue
        if e3!=e4:raise Blocked("R3_AGGREGATE_EGO_AP_IDENTITY_FAILED")
        values.append(dict(repeat=j,delta_AP70=b4-b3,delta_CoopGain=(b4-e4)-(b3-e3),
            B3_fusion_AP70=b3,B4_fusion_AP70=b4,ego_AP70=e3))
    return values


def summarize_delta(values):
    a=[v["delta_AP70"] for v in values]
    return dict(per_repeat=values,mean_delta_AP70=float(np.mean(a)) if a else None,
        delta_quantiles=quantiles(a),positive_repeats=sum(v>0 for v in a),negative_repeats=sum(v<0 for v in a),zero_repeats=sum(v==0 for v in a),
        descriptive_only="twenty finite-pool recombinations, not twenty independent scenes",
        leave_one_repeat_means=[float(np.mean(a[:j]+a[j+1:])) for j in range(len(a))] if len(a)>1 else [],
        leave_repeat_caveat="auxiliary sensitivity only; removing a repeat breaks the complete finite-pool balance and is not a new matched protocol")


def analyze(c,scenes,out,smoke=False):
    table=global_table(scenes);views=sorted({r["view"] for s in scenes for r in s["runs"]})
    comparisons={v:summarize_delta(delta(scenes,v)) for v in views}
    per_scene=[];target_effect=defaultdict(lambda:np.zeros((2,3),dtype=int))
    for s in scenes:
        for view in (v["name"] for v in s["views"]):
            d=summarize_delta(delta([s],view))
            cohort=[t for t in s["targets"] if t["view"]==view]
            def events(mode):
                a=[t for t in cohort if t["mode"]==mode]
                return dict(total=len(a),neighbor_detects_fusion_misses=sum(t["category"]=="B_neighbor_detects_fusion_misses" for t in a),
                    fusion_TP=sum(t["fusion_detected"] for t in a),ego_TP=sum(t["ego_detected"] for t in a),
                    recovery=sum(not t["ego_detected"] and t["fusion_detected"] for t in a),
                    lost=sum(t["ego_detected"] and not t["fusion_detected"] for t in a))
            b3,b4=events("B3"),events("B4")
            denom=b3["total"]
            per_scene.append(dict(scene=s["scene"],view=view,independent=s["independent"],unique_GT=len(s["gt_ids"]),
                all_conditions_per_repeat_AP_and_TP_FP_FN=[r for r in global_table([s]) if r["view"]==view],
                per_scene_descriptive_AP=d,B3=b3,B4=b4,
                delta_neighbor_miss_probability=(b4["neighbor_detects_fusion_misses"]-b3["neighbor_detects_fusion_misses"])/denom if denom else None,
                delta_recall=(b4["fusion_TP"]-b3["fusion_TP"])/denom if denom else None,
                new_FP={m:sum(r["new_cooperative_FP"] for r in s["runs"] if r["view"]==view and r["mode"]==m) for m in ("B3","B4")}))
            if view=="two_cav":
                for t in cohort:
                    if t["mode"] in ("B3","B4"):
                        target_effect[(s["scene"],t["object_id"])][0 if t["mode"]=="B3" else 1]+=np.asarray([
                            int(t["category"]=="B_neighbor_detects_fusion_misses"),int(not t["ego_detected"] and t["fusion_detected"]),
                            int(t["ego_detected"] and not t["fusion_detected"])])
    intervals={}
    for view in views:
        chosen=[s for s in scenes if s["independent"] and any(v["name"]==view for v in s["views"])]
        values=delta(chosen,view);nominal=float(np.mean([r["delta_AP70"] for r in values])) if values else None
        loo={s["scene"]:summarize_delta(delta([a for a in chosen if a is not s],view))["mean_delta_AP70"] for s in chosen}
        draws=[];event_draws={k:[] for k in ("neighbor_detects_fusion_misses","fusion_TP","recovery","lost","new_FP")}
        byscene={r["scene"]:r for r in per_scene if r["view"]==view}
        if len(chosen)>=c["statistics"]["minimum_scenes"]:
            rng=random.Random(c["statistics"]["seed"])
            for _ in range(c["statistics"]["bootstrap_replicates"]):
                sampled=rng.choices(chosen,k=len(chosen));ds=delta(sampled,view)
                draws.append(float(np.mean([d["delta_AP70"] for d in ds])))
                records=[byscene[s["scene"]] for s in sampled];denom=sum(r["B3"]["total"] for r in records)
                for metric in event_draws:
                    diff=sum((r["new_FP"]["B4"]-r["new_FP"]["B3"]) if metric=="new_FP" else
                        r["B4"][metric]-r["B3"][metric] for r in records)
                    event_draws[metric].append(diff/denom if denom else 0.)
        signs=[r["per_scene_descriptive_AP"]["mean_delta_AP70"] for r in per_scene if r["view"]==view and r["independent"]]
        intervals[view]=dict(independent_scenes=len(chosen),scene_ids=[s["scene"] for s in chosen],nominal=nominal,
            scene_bootstrap_CI95=np.quantile(draws,[.025,.975]).tolist() if draws else None,
            target_structure_CI95={k:np.quantile(v,[.025,.975]).tolist() if v else None for k,v in event_draws.items()},
            bootstrap_replicates=len(draws),resampling_unit="entire scene, with all twenty fields/targets/views nested",
            leave_one_scene=loo,direction_fraction=float(np.mean([v*nominal>0 for v in signs])) if signs and nominal else None,
            maximum_leave_scene_relative_change=max((abs(v-nominal)/abs(nominal) for v in loo.values() if v is not None),default=None) if nominal else None)
    effects={str(k):(v[1]-v[0]).tolist() for k,v in target_effect.items()};total=sum(sum(abs(x) for x in v) for v in effects.values())
    concentration=max((sum(abs(x) for x in v)/total for v in effects.values()),default=None) if total else None
    third_ids={s["scene"] for s in scenes if any(v["name"]=="three_cav" for v in s["views"])}
    nested=[]
    if third_ids:
        matched=[s for s in scenes if s["scene"] in third_ids]
        for j in range(20):
            nested.append(dict(repeat=j,scene_ids=sorted(third_ids),
                two_cav_delta=delta(matched,"two_cav")[j]["delta_AP70"],three_cav_delta=delta(matched,"three_cav")[j]["delta_AP70"],
                note="same GT, same shared-CAV field assignments and same scenes; auxiliary nested comparison"))
    statistical=dict(independent_unit="scene",intervals=intervals,per_scene=per_scene,
        target_event_influence=effects,maximum_target_event_share=concentration,
        target_influence_scope="vector of additive neighbor-detects/fusion-misses, recovery and lost counts; not an additive attribution of nonlinear AP",
        nested_two_three_CAV=nested,pooled_all_repetitions_AP_not_reported=True,REAL_WORLD_VALIDATION_MISSING=True)
    atomic_json(out/"R3B_B3_B4_COMPARISON.json",dict(global_AP_per_repeat=table,comparison=comparisons))
    atomic_json(out/"R3B_STATISTICAL_ANALYSIS.json",statistical)
    independent=sum(s["independent"] for s in scenes);unique=sum(len(s["gt_ids"]) for s in scenes if s["independent"])
    main=intervals.get("two_cav",{});ci=main.get("scene_bootstrap_CI95");margin=c["statistics"]["practical_ap_margin"]
    reasons=[];decision="INCONCLUSIVE"
    neighbor_targets={(t["scene"],t["object_id"]) for s in scenes if s["independent"] for t in s["targets"]
        if t["view"]=="two_cav" and any(t["neighbor_detected"].values())}
    if smoke:reasons.append("SMOKE uses previously studied development data only")
    if independent<c["selection"]["minimum_scenes_for_decision"]:reasons.append("fewer than six genuinely unused qualified scenes")
    if unique<c["statistics"]["minimum_unique_GT"]:reasons.append("too few independent fixed-scene GT instances")
    if len(neighbor_targets)<c["statistics"]["minimum_neighbor_detectable_targets"]:reasons.append("too few unique GT instances detected by a neighbor")
    nominal=main.get("nominal")
    stable=(ci is not None and (ci[0]>0 or ci[1]<0) and nominal is not None and abs(nominal)>=margin
        and (main.get("direction_fraction") or 0)>=c["statistics"]["direction_fraction"]
        and main.get("maximum_leave_scene_relative_change") is not None
        and main["maximum_leave_scene_relative_change"]<=c["statistics"]["max_loo_relative_change"])
    interpretable=concentration is not None and concentration<=c["statistics"]["max_target_effect_share"]
    simultaneous_small=(ci is not None and ci[0]>=-margin and ci[1]<=margin and
        all(v is not None and v[0]>=-c["statistics"]["practical_event_probability_margin"] and
            v[1]<=c["statistics"]["practical_event_probability_margin"] for v in main.get("target_structure_CI95",{}).values()))
    if not reasons and stable and interpretable:decision="CONTINUE_DATASET_RESEARCH"
    elif not reasons and simultaneous_small:
        decision="STOP_OR_REDESIGN";reasons.append("predeclared joint AP/target-structure scene-bootstrap equivalence margins satisfied in an adequate cohort; no broad population no-effect claim")
    else:reasons.append("requires stable scene-level effect plus distributed, interpretable target evidence; no automatic seed expansion or parameter tuning")
    # No claim that lack of significance establishes a negligible effect. An AP-equivalence
    # interval alone cannot establish negligible target-structure differences.
    result=dict(decision=decision,reasons=reasons,independent_scenes=independent,unique_GT_instances=unique,
        requirements=c["statistics"],unique_neighbor_detectable_targets=len(neighbor_targets),small_AP_interval=bool(ci and ci[0]>=-margin and ci[1]<=margin),
        simultaneous_target_structure_equivalence=simultaneous_small,
        no_effect_proven=False,formal_dataset_accepted=False,REAL_WORLD_VALIDATION_MISSING=True,
        stop_rule="STOP_OR_REDESIGN on confirmed unacceptable physics error; statistical stop requires adequate scene sample and simultaneous equivalence of both AP and target-structure effects, not absent significance")
    atomic_json(out/"R3_PROJECT_DECISION.json",result)
    lines=["# R3-B 固定检测结果","",f"独立 scene: {independent}；固定GT实例: {unique}；天气场池:20。","",
        "主指标逐repeat汇总全部固定场景并全局排序；每scene的AP仅作描述，未拿scene AP均值替代主指标。","",
        "| view | 条件 | repeat | scope | AP30 | AP50 | AP70 | TP70 | FP70 | FN70 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        a=r["AP"];v=r["counts"]["0.7"]
        lines.append(f"| {r['view']} | {r['mode']} | {r['repeat']} | {r['scope']} | {a['AP30']} | {a['AP50']} | {a['AP70']} | {v['TP']} | {v['FP']} | {v['FN']} |")
    lines.extend(["",decision,"",*reasons,"","没有真实雾天标定。所有检测为UNCALIBRATED_EXPLORATORY；有限场池的20次重组合不充当20个独立scene。"])
    atomic_text(out/"R3B_RESULTS.md","\n".join(lines)+"\n")
    return result

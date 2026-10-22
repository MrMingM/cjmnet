"""Per-scene committed Ego/Neighbour/Fusion diagnostics over all twenty fields."""
from pathlib import Path
import numpy as np

from scorw_r3.core import (Blocked,read_json,digest,identity,array_hash,load_arrays,transaction,atomic_json)
from scorw_r3.selection import fixed_gt
from scorw_r3.metrics import target_category,iou_matrix
from shared_weather_benchmark.dataset_io import read_points
from shared_weather_benchmark.audit import world_xyz
from shared_weather_benchmark.geometry import object_matrix
from scorw_p7.extract_rays import associations
from scorw_p7.balanced_assignment import assignment


def memo_predict(runtime,clouds,gt,root,parent,force_check=False):
    inputs=[array_hash(p) for p in clouds]
    request=dict(parent=parent,inputs=inputs,gt=array_hash(gt),model_state=runtime.initial_state)
    key=identity(request);folder=root/key
    transaction(folder,request,lambda code:atomic_json(folder/"prediction.json",runtime.predict(clouds,gt)))
    result=read_json(folder/"prediction.json")
    if force_check:
        again=runtime.predict(clouds,gt)
        if result["input_hash"]!=again["input_hash"] or result["output_hash"]!=again["output_hash"]:
            raise Blocked("R3_IDENTICAL_EGO_INPUT_NONDETERMINISTIC_OUTPUT")
    return result


def new_false_positives(ego,fusion,runtime,threshold):
    e=np.asarray(ego["boxes"],dtype=np.float32).reshape(-1,8,3)
    f=np.asarray(fusion["boxes"],dtype=np.float32).reshape(-1,8,3)
    e=e[ego["matches"]["0.7"]["unmatched_prediction"]]
    f=f[fusion["matches"]["0.7"]["unmatched_prediction"]]
    m=iou_matrix(f,e,runtime.audit)
    return sum(int(not len(row) or max(row)<threshold) for row in m)


def evaluate(c,old,selection,coord,wave_index,runtime,out,parent):
    rows=read_json(coord/"INPUT_MANIFEST.json")
    source={(r["scene"],r["timestamp"],r["cav"]):r for r in rows}
    waves={(r["scene"],r["timestamp"],r["cav"],r["name"]):r for r in wave_index["responses"]}
    products=[]
    for frame in selection["frames"]:
        scene,stamp,ego=frame["scene"],frame["timestamp"],frame["ego"]
        folder=out/"scenes"/scene/stamp
        def work(code):
            ids,gt=fixed_gt(frame,rows)
            if not ids:raise Blocked("R3_FIXED_CLEAN_GT_EMPTY")
            clean={};masks={}
            ego_pose=source[(scene,stamp,ego)]["pose"]
            for cav in frame["selected_cavs"]:
                r=source[(scene,stamp,cav)];raw=read_points(r["pcd"])
                if digest(r["pcd"])!=r["pcd_sha256"]:raise Blocked("R3_DETECTION_CLEAN_CHANGED")
                clean[cav]=raw;masks[cav]=associations(world_xyz(raw,r["pose"]),raw,frame["objects"])
            runs=[];targets=[];echo=[];ego_balance={}
            for mode in ("B0","B1","B2","B3","B4"):
                for repeat in range(20 if mode in ("B3","B4") else 1):
                    pc={};shown={};data={};stats={}
                    for cav in frame["selected_cavs"]:
                        schedule=selection["assignments"][scene]["schedules"]
                        name=str(schedule["E2" if mode=="B3" else "E3"][cav][repeat]) if mode in ("B3","B4") else mode
                        w=waves[(scene,stamp,cav,name)]
                        if digest(w["pcd"])!=w["pcd_sha256"] or digest(w["response"])!=w["response_sha256"]:
                            raise Blocked("R3_RESPONSE_OR_SERIALIZATION_CHANGED")
                        d=load_arrays(w["response"]);data[cav]=d
                        original=np.flatnonzero(d["emitted"])
                        pc[cav],shown[cav]=runtime.project(read_points(w["pcd"]),source[(scene,stamp,cav)]["pose"],ego_pose,original)
                        stats[cav]=w["stats"]
                    for view in frame["views"]:
                        cavs=view["cavs"];scope={"ego":[ego],"fusion":cavs}
                        scope.update({"neighbor_%d"%k:[v] for k,v in enumerate(cavs[1:],1)})
                        predictions={name:memo_predict(runtime,[pc[v] for v in vehicles],gt,out/"prediction_cache",parent,
                            force_check=(name=="ego" and mode=="B3" and repeat==c["diagnosis"]["ego_repeat_recompute"]))
                            for name,vehicles in scope.items()}
                        e=predictions["ego"]
                        if mode=="B3":ego_balance[(view["name"],repeat)]=(e["input_hash"],e["output_hash"])
                        if mode=="B4" and ego_balance[(view["name"],repeat)]!=(e["input_hash"],e["output_hash"]):
                            raise Blocked("R3_B3_B4_EGO_INPUT_OUTPUT_IDENTITY_FAILED")
                        record=dict(scene=scene,timestamp=stamp,view=view["name"],cavs=cavs,mode=mode,repeat=repeat,
                            gt_ids=ids,predictions=predictions,new_cooperative_FP=new_false_positives(e,predictions["fusion"],runtime,c["diagnosis"]["fp_overlap_iou"]))
                        runs.append(record)
                        echo.append(dict(scene=scene,view=view["name"],mode=mode,repeat=repeat,cavs={v:stats[v] for v in cavs}))
                        for k,oid in enumerate(ids):
                            counts={}
                            for cav in cavs:
                                ix=masks[cav][oid];d=data[cav]
                                visible=np.zeros(len(clean[cav]),dtype=bool);visible[shown[cav]]=True
                                center=np.asarray(object_matrix(frame["objects"][oid]))[:3,3]
                                counts[cav]=dict(clean=len(ix),clean_signal=float(clean[cav][ix,3].sum()),
                                    hard=int(d["retained"][ix].sum()),
                                    weather=int((d["emitted"][ix]&(d["origin"][ix]==1)).sum()),
                                    unknown=int((d["emitted"][ix]&(d["origin"][ix]==2)).sum()),
                                    lost=int((~d["emitted"][ix]).sum()),hard_presented=int((d["retained"][ix]&visible[ix]).sum()),
                                    distance_m=float(np.linalg.norm(center-source[(scene,stamp,cav)]["pose"][:3])))
                            flags={s:k in p["matches"]["0.7"]["matched_gt"] for s,p in predictions.items()}
                            neighbor={v:flags["neighbor_%d"%n] for n,v in enumerate(cavs[1:],1)}
                            category=target_category(flags["ego"],neighbor.values(),flags["fusion"])
                            targets.append(dict(scene=scene,timestamp=stamp,object_id=oid,view=view["name"],mode=mode,repeat=repeat,
                                counts=counts,ego_detected=flags["ego"],neighbor_detected=neighbor,fusion_detected=flags["fusion"],category=category,
                                fusion_diagnostic=predictions["fusion"]["diagnostics"][k],
                                source_scope="Weather/unknown counts follow original Clean-associated paths, not where a simulated point lands. Presented means before voxel caps. No point-count claim of detectability."))
            result=dict(scene=scene,timestamp=stamp,independent=frame["partition"]=="independent_validation",
                views=frame["views"],gt_ids=ids,targets=targets,runs=runs,echo=echo,
                ego_B3_B4_identity="passed",pool_size=20,selection_identity=selection["selection_identity"])
            atomic_json(folder/"SCENE.json",result)
        transaction(folder,dict(parent=parent,frame=frame,selection=selection["selection_identity"]),work)
        products.append(read_json(folder/"SCENE.json"))
        print("R3-B complete scene "+scene+"/"+stamp,flush=True)
    runtime.finish()
    compact=[dict(scene=s["scene"],timestamp=s["timestamp"],independent=s["independent"],views=s["views"],gt_ids=s["gt_ids"],
        ego_B3_B4_identity=s["ego_B3_B4_identity"],full_result_path=str(out/"scenes"/s["scene"]/s["timestamp"]/"SCENE.json"),
        full_result_sha256=digest(out/"scenes"/s["scene"]/s["timestamp"]/"SCENE.json"),
        run_count=len(s["runs"]),target_repeat_records=len(s["targets"]),echo_statistics=s["echo"]) for s in products]
    atomic_json(out/"R3B_SCENE_RESULTS.json",dict(status="complete",scenes=compact,independent_scenes=sum(s["independent"] for s in products),
        frames=len(products),unique_targets=sum(len(s["gt_ids"]) for s in products),
        unique_CAV_observations=sum(len(s["views"][-1]["cavs"]) for s in products),pool_size=20))
    atomic_json(out/"R3B_TARGET_DIAGNOSIS.json",dict(targets=[t for s in products for t in s["targets"]],
        independent_unit="scene",GT_source="fixed Clean; all groups use the same object list in ego coordinates"))
    atomic_json(out/"R3B_NEIGHBOR_ONLY.json",dict(status="implemented",coordinate_protocol=runtime.model_audit["neighbor_frame"],
        scene_counts={s["scene"]:{name:sum(t["category"]==name for t in s["targets"]) for name in (
            "A_neighbor_detects_fusion_recovers","B_neighbor_detects_fusion_misses","C_all_miss","D_ego_detects_fusion_loses")} for s in products},
        note="Counts contain all weather repetitions/views; unique scene/target denominators retained in scene output. Each neighbor evaluated separately; any-neighbor success is an OR, not a merged-neighbor detector."))
    return products


def audit_old_cases(c,old,runtime,out,parent):
    cases=[t for t in old.report["targets"] if t.get("neighbor_evidence_ego_weather_loss_fusion_unrecovered")]
    if len(cases)>c["budget"]["max_old_forensic_cases"]:raise Blocked("R3_OLD_CASE_FORENSIC_BUDGET_EXCEEDED")
    rows={(r["scene"],r["timestamp"],r["cav"]):r for r in old.rows}
    files={r["path"]:r["sha256"] for r in old.report["diagnostic_pcd_files"]}
    records=[];missing=[]
    for case in cases:
        scene,stamp=case["scene"],case["timestamp"]
        frame=next(f for f in old.frames if (f["scene"],f["timestamp"])==(scene,stamp))
        compute=next(f for f in old.report["compute_plan"]["frames"] if (f["scene"],f["timestamp"])==(scene,stamp))
        cavs=compute["cavs"];ego=frame["ego"];ids,gt=fixed_gt(frame,old.rows);slot=ids.index(case["object_id"])
        plan=assignment(cavs,ego);clouds=[]
        if case["mode"] not in {"B3","B4"}:raise Blocked("R3_OLD_CASE_MODE_NOT_POOL_REPLAYABLE")
        schedule=plan["schedules"]["E2" if case["mode"]=="B3" else "E3"]
        for cav in cavs:
            j=schedule[cav][case["repeat"]]
            relative=Path("diagnostic_pcd")/scene/stamp/cav/("field_"+str(j)+".pcd")
            pcd=old.r2/relative
            if not pcd.is_file():missing.append(dict(scene=scene,timestamp=stamp,repeat=case["repeat"],path=str(relative)));break
            if digest(pcd)!=files.get(relative.as_posix()):raise Blocked("R3_OLD_FORENSIC_PCD_CHANGED")
            cloud,_=runtime.project(read_points(pcd),rows[(scene,stamp,cav)]["pose"],rows[(scene,stamp,ego)]["pose"])
            clouds.append(cloud)
        if len(clouds)!=len(cavs):
            records.append(dict(old_record=case,neighbor_detected=None,fusion_cause=None,status="missing_verified_old_input"));continue
        pred={s:memo_predict(runtime,v,gt,out/"prediction_cache",parent) for s,v in
            (("ego",clouds[:1]),("neighbor",clouds[1:2]),("fusion",clouds))}
        for scope in ("ego","fusion"):
            oldscope="cooperative" if scope=="fusion" else "ego"
            reference=next(r for r in old.report["predictions"] if (r["scene"],r["timestamp"],r["mode"],r["repeat"],r["scope"])==
                (scene,stamp,case["mode"],case["repeat"],oldscope))
            a=np.asarray(pred[scope]["boxes"]).reshape(-1,8,3);b=np.asarray(reference["boxes"]).reshape(-1,8,3)
            if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=c["diagnosis"]["old_output_atol"]) or not np.allclose(
                    pred[scope]["scores"],reference["scores"],rtol=0,atol=c["diagnosis"]["old_output_atol"]):
                raise Blocked("R3_OLD_DETECTOR_ADAPTER_REPLAY_MISMATCH")
            if pred[scope]["matches"]["0.7"]["matched_gt"]!=reference["matched_gt"]:
                raise Blocked("R3_OLD_GT_MATCH_REPLAY_MISMATCH")
        center=np.asarray(object_matrix(frame["objects"][case["object_id"]]))[:3,3]
        records.append(dict(old_record=case,status="replayed_with_frozen_model",weather_fields={v:schedule[v][case["repeat"]] for v in cavs},
            ego_distance_m=float(np.linalg.norm(center-rows[(scene,stamp,ego)]["pose"][:3])),
            neighbor_detected=slot in pred["neighbor"]["matches"]["0.7"]["matched_gt"],
            fusion_diagnostic=pred["fusion"]["diagnostics"][slot],old_ego_and_fusion_reproduction="passed"))
    keys={(r["scene"],r["timestamp"],r["object_id"]) for r in cases}
    result=dict(status="complete" if not missing else "partial_missing_old_cache",weather_repeat_records=len(cases),
        unique_targets=len(keys),scene_count=len({k[0] for k in keys}),records=records,missing=missing,
        source_sha256=digest(old.root/"R2_TARGET_DIAGNOSIS.json"),interpretation="Repeated weather records of six unique targets are not fifteen independent physical targets")
    atomic_json(out/"R3B_R2_15_RECORD_AUDIT.json",result)
    runtime.finish()
    return result

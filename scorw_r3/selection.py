"""Clean-only scene qualification, then a full workload/resource freeze."""
from pathlib import Path
import shutil
import numpy as np

from scorw_r3.core import (HERE,REPO,Blocked,read_json,digest,identity,seed_for,atomic_json,atomic_text,server_train,check_inputs)
from shared_weather_benchmark.geometry import pose_matrix,inverse,box_corners,object_matrix
from shared_weather_benchmark.dataset_io import approved_path,cav_order,timestamps,read_metadata,read_points,source_paths
from shared_weather_benchmark.audit import canonical_gt,real_coordinate_audit,world_xyz
from scorw_p7.extract_rays import associations
from scorw_p7.field_pool import make_pool
from scorw_p7.balanced_assignment import assignment
from scorw_repair.range_adapter import gt_range


def allowed_candidates(history,inventory):
    forbidden=set(history["used_or_reviewed_scenes"])|set(history["future_reserved_scenes"])
    eligible=set(history["eligible_known_unused"])-forbidden
    return sorted(eligible&set(inventory)),sorted(set(inventory)-set(history["all_historical_scenes"]))


def fixed_gt(frame,rows):
    ego=next(r for r in rows if r["cav"]==frame["ego"] and r["scene"]==frame["scene"] and r["timestamp"]==frame["timestamp"])
    m=np.asarray(inverse(pose_matrix(ego["pose"])))
    ids=sorted(frame["objects"])
    boxes=np.asarray([np.asarray(box_corners(frame["objects"][oid]))@m[:3,:3].T+m[:3,3] for oid in ids],dtype=np.float32).reshape(-1,8,3)
    b=gt_range();mask=((boxes[:,:,:2]>=b[:2])&(boxes[:,:,:2]<=b[3:5])).all(axis=(1,2))
    return [oid for oid,v in zip(ids,mask) if v],boxes[mask]


def qualify(root,scene,stamp,cavs,c,consulted):
    metadata={};raw={};rows=[]
    frame=dict(scene=scene,timestamp=stamp,cavs=cavs,ego=cavs[0],partition="independent_validation")
    for cav in cavs:
        p,y=source_paths(root,frame,cav);before=(digest(p),digest(y))
        metadata[cav]=read_metadata(y,root);raw[cav]=read_points(p)
        if before!=(digest(p),digest(y)):raise Blocked("R3_SELECTION_INPUT_CHANGED_WHILE_READING")
        row=dict(scene=scene,timestamp=stamp,cav=cav,pcd=str(p),yaml=str(y),pcd_sha256=before[0],yaml_sha256=before[1],
            points=len(raw[cav]),pose=[float(x) for x in metadata[cav]["lidar_pose"]])
        rows.append(row);consulted[(scene,stamp,cav)]=row
    gt,_=canonical_gt(metadata,read_json(REPO/"shared_weather_benchmark/config/prototype.json")["audit"]["gt_pair_tolerance_m"])
    poses={r["cav"]:r["pose"] for r in rows};ego=frame["ego"]
    active=[v for v in cavs if np.linalg.norm(np.asarray(poses[v][:2])-poses[ego][:2])<=c["selection"]["communication_xy_m"]]
    if len(active)<2:return None,dict(reason="fewer_than_two_CAVs_inside_fixed70m",active_cavs=active)
    old=read_json(REPO/"shared_weather_benchmark/config/prototype.json")
    scoped={oid:obj for oid,obj in gt.items() if oid!=ego and min(
        np.linalg.norm(np.asarray(object_matrix(obj))[:3,3]-poses[v][:3]) for v in active)<=old["audit"]["max_object_range_m"]}
    frame.update(active_cavs=active,objects=scoped)
    ids,boxes=fixed_gt(frame,rows)
    if len(ids)<c["selection"]["min_clean_targets"]:return None,dict(reason="too_few_fixed_clean_GT",targets=len(ids))
    evidence={}
    for cav in active:
        masks=associations(world_xyz(raw[cav],poses[cav]),raw[cav],scoped)
        evidence[cav]={oid:int((raw[cav][masks[oid],3]>0).sum()) for oid in ids}
    minimum=c["selection"]["min_points_per_cav_target"]
    counts={v:sum(evidence[ego][oid]>=minimum and evidence[v][oid]>=minimum for oid in ids) for v in active if v!=ego}
    neighbors=sorted(counts,key=lambda v:(-counts[v],v))
    good=[v for v in neighbors if counts[v]>=c["selection"]["min_common_targets"]]
    if not good:return None,dict(reason="too_few_clean_common_targets",per_neighbor_common=counts)
    frame.update(two_cavs=[ego,good[0]],eligible_third=good[1] if len(good)>1 else None,
        clean_target_ids=ids,clean_evidence=evidence,common_target_counts=counts)
    return frame,dict(reason="qualified",targets=len(ids),per_neighbor_common=counts)


def select(c,old,out,smoke=False):
    root=server_train(c["train_root"]);history=read_json(HERE/"inputs/HISTORY_FREEZE.json")
    consulted={};tests=[];frames=[]
    inventory=[p.name for p in root.iterdir() if p.is_dir()]
    candidates,unknown=allowed_candidates(history,inventory)
    if smoke:
        source=old.report["compute_plan"]["frames"][0]
        frame=next(f for f in old.frames if (f["scene"],f["timestamp"])==(source["scene"],source["timestamp"]))
        frame=dict(frame,two_cavs=source["cavs"],eligible_third=None,partition="generator_development",clean_evidence={},common_target_counts={})
        frames=[frame];candidates=[frame["scene"]]
    else:
        order=sorted(candidates,key=lambda s:(seed_for(c["selection"]["seed"],"r3-scene-order",s),s))
        for scene in order:
            if len(frames)>=c["selection"]["target_scenes"]:break
            folder=approved_path(root/scene,root);cavs=cav_order(folder,root)
            if len(cavs)<2:tests.append(dict(scene=scene,reason="one_native_CAV"));continue
            stamps=timestamps(folder/cavs[0],root)
            if any(timestamps(folder/v,root)!=stamps for v in cavs):
                tests.append(dict(scene=scene,reason="incomplete_CAV_timestamp_inventory"));continue
            frozen_order=sorted(stamps,key=lambda t:(seed_for(c["selection"]["seed"],"r3-frame-order",scene,t),t))[:c["selection"]["max_frames_tested_per_scene"]]
            for stamp in frozen_order:
                try:qualified,why=qualify(root,scene,stamp,cavs,c,consulted)
                except FileNotFoundError:
                    tests.append(dict(scene=scene,timestamp=stamp,reason="missing_original_file"));continue
                tests.append(dict(scene=scene,timestamp=stamp,**why))
                if qualified is not None:frames.append(qualified);break
    auxiliary=0
    for f in frames:
        f["selected_cavs"]=list(f["two_cavs"])
        f["views"]=[dict(name="two_cav",cavs=f["two_cavs"])]
        if not smoke and f.get("eligible_third") and auxiliary<c["selection"]["max_three_cav_scenes"]:
            f["selected_cavs"]=[f["ego"]]+sorted([f["two_cavs"][1],f["eligible_third"]])
            f["views"].append(dict(name="three_cav",cavs=f["selected_cavs"]))
            auxiliary+=1
    plan=dict(status="qualified" if frames else "INCONCLUSIVE_NO_QUALIFIED_NEW_SCENES",frames=frames,
        requested_scenes=c["selection"]["target_scenes"],independent_scenes=0 if smoke else len(frames),
        historical_unused_candidates=candidates,unclassified_directories_not_used=unknown,
        history=history,qualification_tests=tests,consulted_inputs=list(consulted.values()),
        input_selection_uses_AP_or_weather=False,smoke_development_only=smoke,
        rule=c["selection"],three_cav_views=auxiliary,scene_shortage=len(frames)<c["selection"]["minimum_scenes_for_decision"])
    atomic_json(out/"R3B_SELECTION_FREEZE.json",plan)
    if not frames:return plan
    coord=out/"coordinates";coord.mkdir(parents=True,exist_ok=True)
    # Reuse the passed coordinate audit; no alternative transform/GT convention.
    if smoke:
        for name in ("INPUT_MANIFEST.json","CLEAN_GT.json","FIELD_DOMAINS.json","REAL_COORDINATE_AUDIT.json"):
            from scorw_r3.core import atomic_bytes
            atomic_bytes(coord/name,(old.coords/name).read_bytes())
    else:
        minimal=[{k:f[k] for k in ("scene","timestamp","cavs","ego","partition")} for f in frames]
        real_coordinate_audit(read_json(REPO/"shared_weather_benchmark/config/prototype.json"),dict(frames=minimal),root,coord)
    rows=read_json(coord/"INPUT_MANIFEST.json");audited=read_json(coord/"CLEAN_GT.json")
    for frame in frames:
        actual=next(f for f in audited if (f["scene"],f["timestamp"])==(frame["scene"],frame["timestamp"]))
        if frame["active_cavs"]!=actual["active_cavs"]:raise Blocked("R3_CONNECTIVITY_AUDIT_DISAGREES")
        frame["objects"]=actual["objects"];frame["clean_gt_sha256"]=actual["clean_gt_sha256"]
        ids,_=fixed_gt(frame,rows)
        if not smoke and ids!=frame["clean_target_ids"]:raise Blocked("R3_FIXED_GT_QUALIFICATION_CHANGED")
    domains=read_json(coord/"FIELD_DOMAINS.json");fieldconfig=read_json(REPO/"scorw_p7/config/protocol.json")
    plan["pools"]={f["scene"]:make_pool(f["scene"],fieldconfig,domains[f["scene"]]) for f in frames}
    plan["assignments"]={f["scene"]:assignment(f["selected_cavs"],f["ego"]) for f in frames}
    # All views share this scene assignment, so adding a third CAV never reassigns existing vehicles.
    plan["selection_identity"]=identity(dict(frames=frames,inputs=rows,pools=plan["pools"],assignments=plan["assignments"]))
    atomic_json(out/"R3B_SELECTION_FREEZE.json",plan)
    check_inputs(rows);check_inputs(plan["consulted_inputs"])
    return plan


def budget(c,selection,coord,out,check_space=True):
    rows=read_json(coord/"INPUT_MANIFEST.json") if coord.exists() else []
    index={(r["scene"],r["timestamp"],r["cav"]):r for r in rows}
    frames=[];total=0;peak=0
    for f in selection["frames"]:
        clouds=[index[(f["scene"],f["timestamp"],v)] for v in f["selected_cavs"]]
        n=sum(r["points"] for r in clouds);cost=21*n;total+=cost
        # Streaming one CAV/field at a time; estimates are uncompressed upper bounds, not measurements.
        mem=max(r["points"] for r in clouds)*512+1024**3;peak=max(peak,mem)
        frames.append(dict(scene=f["scene"],timestamp=f["timestamp"],cavs=f["selected_cavs"],
            original_points=n,ray_field_pairs=cost,views=f["views"],estimated_host_bytes=mem))
    cache=total*c["budget"]["bytes_per_pair_upper"]
    failures=[]
    if total>c["budget"]["max_ray_field_pairs"]:failures.append("total_ray_field_pairs")
    if any(f["ray_field_pairs"]>c["budget"]["max_pairs_per_scene"] for f in frames):failures.append("per_scene_ray_field_pairs")
    if cache>c["budget"]["max_cache_bytes"]:failures.append("disk_cache_budget")
    if peak>c["budget"]["host_memory_limit_bytes"]:failures.append("estimated_host_memory")
    free=shutil.disk_usage(out).free if check_space else None
    if free is not None and free<cache+c["budget"]["reserve_disk_bytes"]:failures.append("free_disk_space")
    r=dict(status="passed" if not failures else "BUDGET_BLOCKED",frames=frames,original_points=sum(f["original_points"] for f in frames),
        detector_forward_upper_bound=sum(43*sum(len(v["cavs"])+1 for v in f["views"]) for f in frames)+3*c["budget"]["max_old_forensic_cases"]+2*len(frames),
        weather_fields=20,constant_fields=1,ray_field_pairs=total,estimated_cache_upper_bytes=cache,
        estimated_host_upper_bytes=peak,available_disk_bytes=free,limits=c["budget"],failures=failures,
        estimation="512 bytes per ray-field includes uncompressed Unicode IDs/arrays and contract overhead; one field/CAV streamed; GPU model/activations use frozen R2 configuration and are reported at runtime",
        all_twenty_fields_preserved=True,no_scene_or_seed_removed_after_selection=True)
    atomic_json(out/"R3B_COMPUTE_PLAN.json",r)
    return r

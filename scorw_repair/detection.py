"""Fixed-weight small AttFuse diagnostic, behind explicit physical/authorization gates."""
from pathlib import Path
import copy
import hashlib
import numpy as np

from scorw_repair.core import (REPO, Blocked, atomic_json, atomic_text, read_json, digest, identity,
    save_arrays, transaction, seed_for)
from scorw_repair.generate import cached_response, original_b1
from scorw_repair.response import simulate
from shared_weather_benchmark.geometry import box_corners, inverse, pose_matrix
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.dataset_io import read_points
from scorw_p7.field_pool import make_pool, restore_field
from scorw_p7.balanced_assignment import assignment
from scorw_p7.extract_rays import associations


def model_identity(config):
    """Cross-check actual weights against the existing frozen reference evaluation."""
    model = config["model"]
    required = (model["checkpoint"],model["config"],model["reference_protocol"])
    if any(not Path(p).is_file() for p in required):
        return dict(status="BLOCKED_MODEL_IDENTITY_MISSING",missing=[Path(p).name for p in required if not Path(p).is_file()])
    old = read_json(model["reference_protocol"])["protocol"]
    sha, cfg = digest(model["checkpoint"]),digest(model["config"])
    if old.get("checkpoint_sha256") != sha or old.get("model_config_sha256") != cfg:
        raise Blocked("BLOCKED_MODEL_REFERENCE_HASH_MISMATCH")
    return dict(status="verified",checkpoint_sha256=sha,model_config_sha256=cfg,
        reference_protocol_sha256=digest(model["reference_protocol"]),
        model_class="attfuse_code.point_pillar_intermediate.PointPillarIntermediate",training=False)


def physical_checks_passed(physical):
    return (physical.get("status")=="numerical_engine_executed" and bool(physical.get("balance"))
        and physical.get("data_identity_check")=="passed"
        and physical.get("source_count_conservation")=="passed"
        and all(r.get("status")=="passed" and r.get("fields")==20 for r in physical["balance"]))


def r2_gate(source, physical, model, numerical, regression):
    if source.get("status")!="SOURCE_AND_CLEAN_SENSOR_AUDIT_PASS":
        return dict(status="BLOCKED_R0_SOURCE_AUDIT",reason="Clean fields/coordinates/source audit has not passed")
    if numerical.get("status")!="passed" or regression.get("status")!="passed":
        return dict(status="BLOCKED_R1_NUMERICAL_REGRESSION",reason="numerical or Hahner regression check failed")
    if not physical_checks_passed(physical):
        return dict(status="BLOCKED_R1_CORRECTNESS",reason="response/origin/balance gates not passed")
    if model.get("status") != "verified":
        return dict(status="BLOCKED_MODEL_IDENTITY",reason="existing model/config reference hashes unavailable")
    return dict(status="eligible",label="UNCALIBRATED_EXPLORATORY",formal_physical_validation=False,
        real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE",
        authorization="user explicitly authorized automatic exploratory R2 after numerical/regression/identity gates")


def greedy_matches(boxes, scores, gt, threshold):
    from opencood.utils import common_utils
    polygons = common_utils.convert_format(np.asarray(boxes)) if len(boxes) else []
    targets = common_utils.convert_format(np.asarray(gt)) if len(gt) else []
    if any(not p.is_valid or p.area <= 0 for p in [*polygons,*targets]):
        raise Blocked("R2_INVALID_BEV_POLYGON")
    available = list(range(len(targets))); matched, fp = [], []
    for i in np.argsort(-np.asarray(scores)):
        ious = common_utils.compute_iou(polygons[i],[targets[j] for j in available])
        if len(ious) and float(max(ious)) >= threshold:
            slot = int(np.argmax(ious)); matched.append(available.pop(slot))
        else:
            fp.append(int(i))
    return dict(matched_gt=matched,unmatched_prediction=fp,TP=len(matched),FP=len(fp),missed=len(gt)-len(matched))


def evaluate(config, coordinate_root, output, model_pin, label, parent, reuse_info=None):
    from scorw_repair.budget import plan as plan_budget
    from scorw_repair.reuse import cached as compatible_cached
    from scorw_repair.iou_audit import IoUAudit
    source_rows=read_json(coordinate_root/"INPUT_MANIFEST.json")
    frames,compute_plan=plan_budget(config,read_json(coordinate_root/"CLEAN_GT.json"),source_rows,output)
    print("R2 planned all frames: "+str(compute_plan["required_ray_field_pairs"])+
        "/"+str(compute_plan["run_limit"])+" ray-field pairs",flush=True)
    iou_audit=IoUAudit(output)
    from scorw_repair.core import install_dataset_guard
    from scorw_repair.range_adapter import literal_range_filter,mask as fixed_range_mask
    install_dataset_guard()
    import torch
    from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from opencood.data_utils.pre_processor import build_preprocessor
    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.tools.train_utils import to_device
    from opencood.utils import eval_utils, pcd_utils, common_utils, box_utils
    from shared_weather_benchmark.audit import world_xyz
    from reference_weather.common import write_pcd
    if model_identity(config) != model_pin:
        raise Blocked("R2_MODEL_CHANGED")
    if not torch.cuda.is_available():
        raise Blocked("R2_GPU_UNAVAILABLE")
    h = load_yaml(config["model"]["config"])
    if h["model"]["core_method"] != "point_pillar_intermediate":
        raise Blocked("R2_PURE_ATTFUSE_REQUIRED")
    args = h["fusion"].get("args",{})
    if isinstance(args,dict) and args.get("proj_first",True) is not True:
        raise Blocked("R2_ADAPTER_REQUIRES_PROJECT_FIRST")
    torch.manual_seed(config["seed"]); torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    model = PointPillarIntermediate(h["model"]["args"])
    weights = torch.load(config["model"]["checkpoint"],map_location="cpu",weights_only=False)
    model.load_state_dict(weights,strict=True); del weights
    model = model.eval()
    def state_hash():
        return identity({k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                         for k,v in sorted(model.state_dict().items())})
    initial_state_hash=state_hash()
    original_scatter = model.scatter

    class EmptyAwareScatter(torch.nn.Module):
        """Preserve empty CAV slots as zero BEV; no fake points or learned parameters."""
        def forward(self, batch):
            features, coords = batch["pillar_features"],batch["voxel_coords"]
            # Frozen PillarVFE.squeeze removes the voxel axis when exactly one
            # voxel survives. Restore it without changing any feature values.
            features=features.reshape(-1,original_scatter.num_bev_features)
            if len(features)!=len(coords):raise Blocked("R2_PILLAR_SOURCE_SHAPE_MISMATCH")
            count = int(batch["record_len"].sum().item())
            spatial = torch.zeros((count,original_scatter.num_bev_features,
                original_scatter.nz*original_scatter.nx*original_scatter.ny),dtype=features.dtype,device=features.device)
            for b in range(count):
                mask = coords[:,0] == b
                cc = coords[mask]
                indices = (cc[:,1]+cc[:,2]*original_scatter.nx+cc[:,3]).long()
                spatial[b,:,indices] = features[mask].t()
            batch["spatial_features"] = spatial.view(count,original_scatter.num_bev_features*original_scatter.nz,
                                                     original_scatter.ny,original_scatter.nx)
            return batch

    replacement = EmptyAwareScatter()
    # An exact nonempty equivalence gate precedes any modified empty-slot behavior.
    synthetic = dict(pillar_features=torch.arange(128,dtype=torch.float32).reshape(2,64),
        voxel_coords=torch.tensor([[0,0,0,0],[1,0,0,1]]),record_len=torch.tensor([2]))
    if not torch.equal(original_scatter(dict(synthetic))["spatial_features"],replacement(dict(synthetic))["spatial_features"]):
        raise Blocked("R2_EMPTY_SCATTER_NONEMPTY_EQUIVALENCE_FAILED")
    for feature,coords in ((torch.ones(64),torch.tensor([[0,0,0,0]])),
            (torch.empty((0,64)),torch.empty((0,4),dtype=torch.int64))):
        mapped=replacement(dict(pillar_features=feature,voxel_coords=coords,record_len=torch.tensor([2])))["spatial_features"]
        if mapped.shape[0]!=2 or torch.count_nonzero(mapped[1]).item()!=0:
            raise Blocked("R2_EMPTY_SCATTER_SLOT_PRESERVATION_FAILED")
    model.scatter = replacement
    pre,post = build_preprocessor(h["preprocess"],False),build_postprocessor(h["postprocess"],False)
    device=torch.device("cuda:0")
    anchors=torch.from_numpy(post.generate_anchor_box())
    index={(r["scene"],r["timestamp"],r["cav"]):r for r in source_rows}
    domain=read_json(coordinate_root/"FIELD_DOMAINS.json")
    field_config=read_json(REPO/"scorw_p7/config/protocol.json")
    all_results=[]; target_records=[]; evaluation={}; ablations=0; model_on_gpu=False; encoded_files=[]
    echo_statistics=[]; balance_checks=[]

    def predict(clouds):
        nonlocal model_on_gpu
        if not model_on_gpu:
            model.to(device); model_on_gpu=True
        features=[pre.preprocess(np.asarray(p,dtype=np.float32)) for p in clouds]
        data=dict(processed_lidar=to_device(pre.collate_batch(features),device),
            record_len=torch.tensor([len(clouds)],dtype=torch.int64,device=device),anchor_box=anchors.to(device),
            transformation_matrix=torch.eye(4,device=device))
        with torch.inference_mode():
            pred=model(data)
            with literal_range_filter(box_utils):
                boxes,scores=post.post_process({"ego":data},{"ego":pred})
        if boxes is None:
            return np.empty((0,8,3)),np.empty(0)
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():
            raise Blocked("R2_NONFINITE_PREDICTION")
        return boxes.cpu().numpy(),scores.cpu().numpy()

    ray_field_pairs=0
    for frame in frames:
        scene,stamp,ego=frame["scene"],frame["timestamp"],frame["ego"]
        cavs=[ego]+sorted(c for c in frame["active_cavs"] if c != ego)[:config["r2"]["cavs"]-1]
        if len(cavs) != config["r2"]["cavs"]:
            raise Blocked("R2_FROZEN_CAV_COUNT_UNAVAILABLE")
        plan=assignment(cavs,ego); pool=make_pool(scene,field_config,domain[scene])
        ego_pose=index[(scene,stamp,ego)]["pose"]
        inv=np.asarray(inverse(pose_matrix(ego_pose)))
        ids=sorted(frame["objects"])
        gt=np.asarray([np.asarray(box_corners(frame["objects"][i]))@inv[:3,:3].T+inv[:3,3] for i in ids],dtype=np.float32)
        mask=fixed_range_mask(torch.from_numpy(gt)).numpy()
        gt=gt[mask]; ids=[i for i,k in zip(ids,mask) if k]
        if not ids:
            raise Blocked("R2_FIXED_CLEAN_GT_EMPTY")
        responses={}; masks={}; raw_by_cav={}
        for cav in cavs:
            row=index[(scene,stamp,cav)]; raw=read_points(row["pcd"]); raw_by_cav[cav]=raw
            if digest(row["pcd"]) != row["pcd_sha256"] or len(raw)!=row["points"]:
                raise Blocked("R2_CLEAN_CHANGED")
            ray_field_pairs += 21*len(raw)
            if ray_field_pairs > config["r2"]["maximum_ray_field_pairs"]:
                raise Blocked("R2_COMPUTE_BUDGET_EXCEEDED")
            key=[scene,stamp,cav,row["pcd_sha256"]]; indices=np.arange(len(raw),dtype=np.int64)
            constant=WorldField(kind="constant",mean_alpha=config["field"]["mean_alpha"],bounds=domain[scene])
            zero=WorldField(kind="constant",mean_alpha=0,bounds=domain[scene])
            responses[cav]={"B0":simulate(raw,row["pose"],zero,config["receiver"],key),
                "B1":original_b1(raw,config,key,indices),
                "B2":compatible_cached(raw,row["pose"],constant,config,key,indices,output/"cloud_cache"/scene/stamp/cav/"constant",parent,
                    reuse_info,Path(scene)/stamp/cav/"constant")}
            for j,f in enumerate(pool["fields"]):
                responses[cav][j]=compatible_cached(raw,row["pose"],restore_field(f),config,key,indices,
                    output/"cloud_cache"/scene/stamp/cav/f"field_{j:02}",parent,
                    reuse_info,Path(scene)/stamp/cav/f"field_{j:02}")
            fingerprints=[identity({k:hashlib.sha256(responses[cav][j][k].tobytes()).hexdigest()
                for k in ("weather_xyzi","origin","emitted","integral","point_identity")}) for j in range(20)]
            order=plan["schedules"]["E3"][cav]
            if sorted(order)!=list(range(20)) or sorted(fingerprints)!=sorted(fingerprints[j] for j in order):
                raise Blocked("R2_FULL_RAY_BALANCE_FAILED")
            balance_checks.append(dict(scene=scene,timestamp=stamp,cav=cav,status="passed",fields=20,
                full_response_multiset_sha256=identity(sorted(fingerprints)),extra_scattering_randomness=False))
            masks[cav]=associations(world_xyz(raw,row["pose"]),raw,frame["objects"])
        for mode in ("B0","B1","B2","B3","B4"):
            for repeat in range(20 if mode in {"B3","B4"} else 1):
                selected={c:responses[c][plan["schedules"]["E2" if mode=="B3" else "E3"][c][repeat] if mode in {"B3","B4"} else mode] for c in cavs}
                from scorw_repair.response import summary
                echo_statistics.extend(dict(scene=scene,timestamp=stamp,mode=mode,repeat=repeat,cav=c,
                    stats=summary(d,config["distance_bins_m"])) for c,d in selected.items())
                projected=[]
                for cav in cavs:
                    data=selected[cav]
                    if "serialized_xyzi" not in data:
                        field_name=("field_"+str(plan["schedules"]["E2" if mode=="B3" else "E3"][cav][repeat])) if mode in {"B3","B4"} else mode
                        path=output/"diagnostic_pcd"/scene/stamp/cav/(field_name+".pcd")
                        write_pcd(path,data["weather_xyzi"][data["emitted"]])
                        written=read_points(path)
                        if written.shape!=data["weather_xyzi"][data["emitted"]].shape:
                            raise Blocked("R2_PCD_SOURCE_COUNT_CHANGED")
                        data["serialized_xyzi"]=data["weather_xyzi"].copy()
                        data["serialized_xyzi"][data["emitted"]]=written
                        encoded_files.append(dict(path=str(path.relative_to(output)),sha256=digest(path)))
                    pc=data["serialized_xyzi"][data["emitted"]]
                    original_slots=np.flatnonzero(data["emitted"])
                    if len(data["emitted"])>=2**24:
                        raise Blocked("R2_FLOAT32_INDEX_TRACKING_BUDGET_EXCEEDED")
                    # Frozen filters preserve extra columns. Carry exact source slots
                    # through them; only XYZI enters the detector.
                    pc=np.column_stack((pc,original_slots.astype(np.float32)))
                    pc=pcd_utils.mask_ego_points(pc.copy())
                    pose=np.asarray(pose_matrix(index[(scene,stamp,cav)]["pose"]))
                    transform=inv@pose
                    pc[:,:3]=pc[:,:3]@transform[:3,:3].T+transform[:3,3]
                    pc=pcd_utils.mask_points_by_range(pc,h["preprocess"]["cav_lidar_range"])
                    data["presented_mask"]=np.zeros(len(data["emitted"]),dtype=bool)
                    data["presented_mask"][pc[:,4].astype(np.int64)]=True
                    projected.append(np.ascontiguousarray(pc[:,:4]))
                predictions={scope:predict(projected[:1] if scope=="ego" else projected) for scope in ("ego","cooperative")}
                matches={}
                for scope,(boxes,scores) in predictions.items():
                    key=(mode,repeat,scope)
                    stat=evaluation.setdefault(key,{t:dict(tp=[],fp=[],gt=0,score=[]) for t in (.3,.5,.7)})
                    with iou_audit.guard(common_utils):
                        for t in stat:
                            eval_utils.caluclate_tp_fp(torch.from_numpy(boxes),torch.from_numpy(scores),torch.from_numpy(gt),stat,t)
                        matched=greedy_matches(boxes,scores,gt,.7)
                    matches[scope]=matched
                    all_results.append(dict(scene=scene,timestamp=stamp,mode=mode,repeat=repeat,scope=scope,
                        boxes=boxes.tolist(),scores=scores.tolist(),gt_ids=ids,**matched))
                ego_fp=[predictions["ego"][0][i] for i in matches["ego"]["unmatched_prediction"]]
                new_fp=0
                for i in matches["cooperative"]["unmatched_prediction"]:
                    p=common_utils.convert_format(predictions["cooperative"][0][i:i+1])[0]
                    values=iou_audit.call(common_utils.compute_iou,p,common_utils.convert_format(np.asarray(ego_fp))) if ego_fp else []
                    new_fp+=int(not len(values) or max(values)<config["r2"]["fp_overlap_iou"])
                all_results[-1]["new_cooperative_fp"]=new_fp
                for slot,oid in enumerate(ids):
                    counts={}
                    for cav in cavs:
                        indices=masks[cav][oid]; d=selected[cav]
                        counts[cav]=dict(clean=len(indices),hard=int(d["retained"][indices].sum()),
                            clean_signal=float(raw_by_cav[cav][indices,3].astype(float).sum()),
                            unknown=int((d["emitted"][indices]&(d["origin"][indices]==2)).sum()),
                            scatter=int((d["emitted"][indices]&(d["origin"][indices]==1)).sum()),
                            no_return=int((~d["emitted"][indices]).sum()),
                            clean_presented=int(responses[cav]["B0"]["presented_mask"][indices].sum()),
                            hard_presented=int((d["retained"][indices]&d["presented_mask"][indices]).sum()),
                            unknown_presented=int((d["emitted"][indices]&(d["origin"][indices]==2)&d["presented_mask"][indices]).sum()))
                    hard=[v["hard_presented"] for v in counts.values()]
                    if any(v["unknown_presented"] for v in counts.values()):
                        category="unknown_mixed_signal"
                    elif not any(hard): category="all_no_identified_original_echo"
                    elif not counts[ego]["hard_presented"] and any(v["hard_presented"] for c,v in counts.items() if c!=ego): category="neighbor_only_original_echo"
                    elif any(v["scatter"] for v in counts.values()) and any(hard): category="potential_conflict_scatter_and_hard_evidence"
                    elif all(v["clean_presented"] and v["hard_presented"]/v["clean_presented"]>=.5 for v in counts.values()): category="all_good_under_diagnostic_threshold"
                    else: category="residual_weak_or_disparate"
                    e=slot in matches["ego"]["matched_gt"]; co=slot in matches["cooperative"]["matched_gt"]
                    weather_echo_loss=(mode!="B0" and counts[ego]["clean_signal"]>0 and counts[ego]["clean_presented"]>0
                        and counts[ego]["hard_presented"]==0 and counts[ego]["unknown_presented"]==0)
                    neighbor_evidence=any(v["hard_presented"]>0 for c,v in counts.items() if c!=ego)
                    intervention=None
                    if (weather_echo_loss and category=="neighbor_only_original_echo" and co and not e
                            and ablations<config["r2"]["maximum_neighbor_ablation_targets"]):
                        counterfactual=[]
                        for cav in cavs:
                            d=selected[cav]; keep=d["emitted"].copy()
                            if cav!=ego:
                                associated=masks[cav][oid]
                                keep[associated[d["retained"][associated]]]=False
                            pc=pcd_utils.mask_ego_points(d["serialized_xyzi"][keep].copy())
                            transform=inv@np.asarray(pose_matrix(index[(scene,stamp,cav)]["pose"]))
                            pc[:,:3]=pc[:,:3]@transform[:3,:3].T+transform[:3,3]
                            counterfactual.append(pcd_utils.mask_points_by_range(pc,h["preprocess"]["cav_lidar_range"]))
                        boxes_cf,scores_cf=predict(counterfactual)
                        with iou_audit.guard(common_utils):matched_cf=greedy_matches(boxes_cf,scores_cf,gt,.7)
                        intervention=dict(target_still_detected=slot in matched_cf["matched_gt"],
                            operation="remove only neighbor model-selected original hard returns associated with this fixed Clean box; ego/scatter/unknown returns held fixed",
                            scope="small diagnostic intervention; box membership does not establish surface semantics")
                        ablations+=1
                    target_records.append(dict(scene=scene,timestamp=stamp,object_id=oid,mode=mode,repeat=repeat,
                        category=category,counts=counts,ego_detected=e,cooperative_detected=co,
                        cooperative_rescue=co and not e,cooperative_lost=e and not co,
                        ego_lost_original_echo_under_weather_model=weather_echo_loss,
                        neighbor_original_evidence=neighbor_evidence,
                        neighbor_evidence_ego_weather_loss_fusion_unrecovered=weather_echo_loss and neighbor_evidence and not co,
                        neighbor_original_echo_ablation=intervention))
        print("R2 frame complete: "+scene+"/"+stamp+"; cumulative="+str(ray_field_pairs),flush=True)
    if ray_field_pairs!=compute_plan["required_ray_field_pairs"]:
        raise Blocked("R2_COMPUTE_PLAN_EXECUTION_MISMATCH")
    iou_audit.save()
    stats=[]
    for (mode,repeat,scope),s in evaluation.items():
        stats.append(dict(mode=mode,repeat=repeat,scope=scope,AP={f"AP{int(t*100)}":float(eval_utils.calculate_ap(copy.deepcopy(s),t,True)[0]) for t in s},
            counts={str(t):dict(TP=sum(s[t]["tp"]),FP=sum(s[t]["fp"]),missed=s[t]["gt"]-sum(s[t]["tp"]),GT=s[t]["gt"]) for t in s}))
    if state_hash()!=initial_state_hash or model_identity(config)!=model_pin:
        raise Blocked("R2_FROZEN_MODEL_STATE_CHANGED")
    result=dict(status="detector_diagnostic_executed",label=label,training=False,model_identity=model_pin,
        model_state_before_after_sha256=initial_state_hash,
        runtime=dict(torch=torch.__version__,hip=torch.version.hip,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)),
        predictions=all_results,targets=target_records,metrics=stats,source_count="full original clouds, not R1 sampled rays",
        echo_statistics=echo_statistics,balance_checks=balance_checks,
        compute_plan=compute_plan,iou_audit=iou_audit.report,
        cache_reuse_policy="verified v2 numerical responses may be imported; detector predictions recomputed with finite-IoU auditing",
        diagnosis_counts={mode:{key:sum(int(t[key]) for t in target_records if t["mode"]==mode)
            for key in ("cooperative_rescue","cooperative_lost","neighbor_evidence_ego_weather_loss_fusion_unrecovered")}
            for mode in ("B0","B1","B2","B3","B4")},
        diagnostic_pcd_files=encoded_files,pcd_scope="only the frozen R2 diagnostic subset; same original RGB serializer for every condition, no full dataset",
        independent_scenes=len({f["scene"] for f in frames}),frames=len(frames),pool_size=20,
        neighbor_ablation_count=ablations,neighbor_ablation_rule="first eligible neighbor-only rescue targets in fixed scene/frame/mode/repeat/object order; maximum eight, not used to choose fields or scenes",
        primary_AP="BEV/VOC2010 global score sorting; no None frame skipped; same fixed Clean GT",
        causal_limit="rescue counts demonstrate observed detections only; they do not prove neighbor information was causally used",
        evidence_scope="hard_presented/unknown_presented count associated original paths surviving the same ego/range filters before voxelization; voxel caps can discard additional points; raw echo statistics are also retained",
        independent_validation=False,weather_dataset_accepted=False,
        physical_calibration_passed=False,real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE")
    atomic_json(output/"R2_TARGET_DIAGNOSIS.json",result)
    lines=["# R2 冻结 AttFuse 诊断", "",label,"", "不训练；固定 Clean GT；None 按零预测。开发 scene 小样本，不作独立测试或数据集创新证明。", "",
        "| 条件 | 重复 | 输入 | AP30 | AP50 | AP70 |", "|---|---:|---|---:|---:|---:|"]
    for s in stats:
        lines.append(f"| {s['mode']} | {s['repeat']} | {s['scope']} | {s['AP']['AP30']} | {s['AP']['AP50']} | {s['AP']['AP70']} |")
    lines.extend(["", "逐车完整20场回波分布平衡已核验。AP 的完整重复保留在 JSON 中，20次重组合不充当独立道路场景。",
        "", "邻车留有模型判定的原始回波、ego 在天气下失去原始回波且协同漏检的次数："])
    for mode,count in result["diagnosis_counts"].items():
        lines.append(f"- {mode}: {count['neighbor_evidence_ego_weather_loss_fusion_unrecovered']}")
    lines.extend(["", "这些次数仅描述模型行为；保留少量回波不保证可检测，不能直接判定融合缺陷。",
                  "真实雾天校准缺失，不能将该结果外推为真实天气性能。"])
    atomic_text(output/"R2_RESULTS.md","\n".join(lines)+"\n")
    return result

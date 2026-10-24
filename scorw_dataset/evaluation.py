"""D6: fixed AttFuse Clean/Fog validate, same Clean GT, never Test AP."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scorw_dataset.core import read_json,digest,identity,atomic_json,transaction,verify_input
from scorw_dataset.io import read_points
from shared_weather_benchmark.audit import canonical_gt
from shared_weather_benchmark.geometry import object_matrix,box_corners,pose_matrix,inverse
from scorw_r3.metrics import ap


def run(c,root,out,gpu,model_identity,parent):
    import os
    os.environ["ROCR_VISIBLE_DEVICES"]=str(gpu)
    os.environ.pop("HIP_VISIBLE_DEVICES",None);os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    from scorw_r3.runtime import Runtime
    data=read_json(root/"manifests/VALIDATE_SELECTION.json")
    source={(r["scene"],r["timestamp"],r["cav"]):r for r in data["rows"]}
    runtime=Runtime(dict(model=c["model"]),SimpleNamespace(config=dict(seed=c["detector_seed"]),model=model_identity),out)
    atomic_json(out/"MODEL_AUDIT.json",runtime.model_audit)
    h=runtime.h;limit=h.get("train_params",{}).get("max_cav",7);results=[]
    try:
        for scene in data["layout"]:
            for stamp in scene["timestamps"]:
                frame_root=out/"frames"/scene["scene"]/stamp
                def infer(code):
                    import yaml
                    metadata={}
                    for cav in scene["cavs"]:
                        row=source[(scene["scene"],stamp,cav)];verify_input(row)
                        with Path(row["yaml"]).open(encoding="utf-8") as f:metadata[cav]=yaml.load(f,Loader=yaml.Loader)
                    objects,_=canonical_gt(metadata,c["evaluation"]["gt_pair_tolerance_m"])
                    ego=scene["cavs"][0];pose=source[(scene["scene"],stamp,ego)]["pose"]
                    active=[v for v in scene["cavs"] if np.linalg.norm(np.asarray(source[(scene["scene"],stamp,v)]["pose"][:2])-pose[:2])<=c["evaluation"]["communication_xy_m"]][:limit]
                    selected={k:o for k,o in objects.items() if k!=ego and min(np.linalg.norm(
                        np.asarray(object_matrix(o))[:3,3]-source[(scene["scene"],stamp,v)]["pose"][:3]) for v in active)<=c["evaluation"]["max_object_range_m"]}
                    ids=sorted(selected);m=np.asarray(inverse(pose_matrix(pose)))
                    gt=np.asarray([np.asarray(box_corners(selected[i]))@m[:3,:3].T+m[:3,3] for i in ids],dtype=np.float32).reshape(-1,8,3)
                    b=c["evaluation"]["GT_RANGE"];mask=((gt[:,:,:2]>=b[:2])&(gt[:,:,:2]<=b[3:5])).all(axis=(1,2))
                    gt=gt[mask];ids=[i for i,yes in zip(ids,mask) if yes]
                    pred={}
                    for mode in ("clean","fog"):
                        clouds=[]
                        for cav in active:
                            row=source[(scene["scene"],stamp,cav)]
                            pc,_=runtime.project(read_points(root/mode/row["relative"]),row["pose"],pose)
                            clouds.append(pc)
                        pred[mode]=runtime.predict(clouds,gt,diagnose=False)
                    atomic_json(frame_root/"RESULT.json",dict(scene=scene["scene"],timestamp=stamp,cavs=active,GT_ids=ids,
                        clean_GT_sha256=identity(gt.tolist()),predictions=pred))
                transaction(frame_root,dict(parent=parent,scene=scene["scene"],stamp=stamp,selection=identity(scene)),infer)
                results.append(read_json(frame_root/"RESULT.json"))
                if len(results)%20==0 or len(results)==data["frames"]:print(f"D6 validate {len(results)}/{data['frames']} Clean/Fog pairs",flush=True)
        if len(results)!=data["frames"]:raise ValueError("Incomplete validation inference")
        metrics={mode:{"AP%d"%round(t*100):dict(AP=ap([r["predictions"][mode]["matches"][str(t)] for r in results]),
            **{k:sum(r["predictions"][mode]["matches"][str(t)][k] for r in results) for k in ("TP","FP","FN","gt")}) for t in (.3,.5,.7)} for mode in ("clean","fog")}
        if metrics["clean"]["AP70"]["gt"]==0:raise ValueError("No usable fixed Clean validation GT")
        report=dict(status="passed",label="UNCALIBRATED_EXPLORATORY",split="validate",frames=len(results),
            fixed_same_Clean_GT=True,scope=c["evaluation"],metrics=metrics,model=model_identity,
            native_dataset_GT_filter_used=False,Test_AP_computed=False,AP_decrease_required=False,
            note="synchronous frozen R3 project-first adapter; detector CAV cap/70m apply only to evaluation, generation includes all CAVs",
            frame_results_identity=identity(results),REAL_WORLD_VALIDATION_MISSING=True)
        atomic_json(out/"VALIDATION_ATTFUSE_RESULTS.json",report)
        return report
    finally:
        runtime.finish();runtime.model.cpu();runtime.torch.cuda.empty_cache()

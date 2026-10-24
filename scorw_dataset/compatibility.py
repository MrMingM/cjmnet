"""D5 isolated native-loader indexing and full real-reader/voxel compatibility."""
import argparse
from pathlib import Path
import random
import numpy as np
from scorw_dataset.core import server_only,read_json,atomic_json,digest,source_identity,SPLITS
from scorw_dataset.io import read_points


def import_check(root,out):
    import torch
    from opencood.data_utils.datasets import build_dataset
    from opencood.data_utils.pre_processor import build_preprocessor
    from opencood.data_utils.post_processor import build_postprocessor
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from attfuse_code.point_pillar_intermediate import PointPillarIntermediate
    from scorw_r3.runtime import validate_project_first
    p=read_json(Path(root)/"DATASET_PROTOCOL.json");h=load_yaml(p["config"]["model"]["config"])
    fusion=validate_project_first(h)
    if h["model"]["core_method"]!="point_pillar_intermediate":raise ValueError("Expected pure frozen AttFuse")
    for key in ("train_params","loss","optimizer","lr_scheduler","data_augment"):
        if key not in h:raise ValueError("Frozen training config lacks required key: "+key)
    pre=build_preprocessor(h["preprocess"],False)
    for pc in (np.empty((0,4),dtype=np.float32),np.asarray([[2.,2.,0.,.5]],dtype=np.float32)):
        features=pre.preprocess(pc);collated=pre.collate_batch([features])
        if any(not np.isfinite(v).all() for v in features.values()) or any(
                torch.is_tensor(v) and not torch.isfinite(v).all() for v in collated.values()):
            raise ValueError("CPU empty/nonempty voxel precheck failed")
    model=PointPillarIntermediate(h["model"]["args"]).eval()
    state=torch.load(p["config"]["model"]["checkpoint"],map_location="cpu",weights_only=False)
    model.load_state_dict(state,strict=True);del model,state
    anchors=build_postprocessor(h["postprocess"],False).generate_anchor_box()
    if not np.isfinite(anchors).all():raise ValueError("Invalid frozen anchor configuration")
    atomic_json(Path(out)/"DEPENDENCY_AUDIT.json",dict(status="passed",native_loader_import="passed",CPU_voxelizer="passed",
        empty_and_nonempty_CPU_preprocess="passed",model_class_import="passed",strict_CPU_checkpoint_load="passed",
        future_training_config_required_keys="passed",fusion=fusion,GPU_used=False,torch_cuda_check_performed=False))


def run(root,out):
    import torch
    from opencood.data_utils.datasets import build_dataset
    from opencood.data_utils.pre_processor import build_preprocessor
    from opencood.hypes_yaml.yaml_utils import load_yaml
    root=Path(root);protocol=read_json(root/"DATASET_PROTOCOL.json");c=protocol["config"]
    if source_identity()["identity"]!=protocol["sources"]["identity"]:raise ValueError("Native source changed before D5")
    records=[]
    for split in SPLITS:
        data=read_json(root/"manifests"/(split.upper()+"_SELECTION.json"))
        max_cav=max(len(s["cavs"]) for s in data["layout"])
        for mode in ("clean","fog"):
            h=load_yaml(c["model"]["config"])
            for k in ("weather_augmentation","prediction_distillation","DG_params"):h.pop(k,None)
            h["data_augment"]=[];h["root_dir"]=str(root/mode/split);h["validate_dir"]=str(root/mode/split)
            h.setdefault("train_params",{})["max_cav"]=max_cav
            dataset=build_dataset(h,visualize=False,train=False)
            if len(dataset)!=data["frames"]:raise ValueError("Native loader frame count mismatch")
            keys=[];native_paths=set()
            resolved_paths={(root/mode/r["relative"]).resolve(strict=True):r["relative"] for r in data["rows"]}
            for scene in dataset.scenario_database.values():
                for cav,frames in scene.items():
                    for stamp,f in frames.items():
                        if stamp=="ego":continue
                        p=Path(f["lidar"])
                        if p.resolve(strict=True) not in resolved_paths:raise ValueError("Native loader reads unexpected data path")
                        native_paths.add(resolved_paths[p.resolve(strict=True)])
                first=next(iter(scene.values()))
                keys.extend((Path(f["lidar"]).parent.parent.name,t) for t,f in first.items() if t!="ego")
            expected=[(s["scene"],t) for s in data["layout"] for t in s["timestamps"]]
            if keys!=expected or native_paths!={r["relative"] for r in data["rows"]}:raise ValueError("Native loader CAV/frame/path coverage differs")
            pre=build_preprocessor(h["preprocess"],False);empty=0
            for i,row in enumerate(data["rows"],1):
                pc=read_points(root/mode/row["relative"]);empty+=not len(pc)
                features=pre.preprocess(pc)
                for v in features.values():
                    if not np.isfinite(v).all():raise ValueError("Nonfinite native preprocess output")
                collated=pre.collate_batch([features])
                for v in collated.values():
                    if torch.is_tensor(v) and not torch.isfinite(v).all():raise ValueError("Nonfinite collated features")
                if i%200==0 or i==len(data["rows"]):print(f"D5 {mode}/{split}: {i}/{len(data['rows'])} clouds",flush=True)
            native_samples=[]
            if split!="test":
                cursor=0;indices=[]
                for s in data["layout"]:
                    indices.extend([cursor,cursor+len(s["timestamps"])-1]);cursor+=len(s["timestamps"])
                for index in sorted(set(indices)):
                    np.random.seed(c["master_seed"]+index);random.seed(c["master_seed"]+index)
                    sample=dataset[index]
                    if sample is None:raise ValueError("Native dataset returned None at fixed compatibility frame")
                    batch=dataset.collate_batch_test([sample])
                    if batch is None:raise ValueError("Native dataset collate returned None")
                    native_samples.append(index)
            records.append(dict(split=split,mode=mode,frames=len(dataset),all_CAV_PCDs=len(native_paths),empty_clouds=empty,
                actual_paths="native scenario_database checked for every CAV/timestamp",preprocessor=h["preprocess"]["core_method"],
                maximum_CAVs_for_catalogue=max_cav,GT_used_for_catalogue_or_voxel_checks=False,
                native_getitem_collate_indices=native_samples,Test_target_labels_interpreted=False))
    atomic_json(out/"COMPATIBILITY_AUDIT.json",dict(status="passed",records=records,GPU_used=False,
        native_loader_index_all_splits="passed",every_PCD_reader_and_voxel_preprocessing="passed",
        scope="native dataset catalogue plus per-CAV preprocessing/collation; no synthetic point insertion or model structural changes"))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True);p.add_argument("--out",required=True)
    p.add_argument("--imports-only",action="store_true");a=p.parse_args()
    server_only()
    if a.imports_only:import_check(Path(a.root),Path(a.out))
    else:run(Path(a.root),Path(a.out))

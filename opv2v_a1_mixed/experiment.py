"""Matched four-weather joint training for OPV2V, with validation-only selection."""
import argparse
from collections import Counter
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT/"opv2v_gspr_a1_v2"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import joint
import runtime
import pipeline as previous
from data import Mixed, WEATHERS, datasets, plan, scan

DATA = Path("/data/cjm/datasets")
OUT = DATA/"logs/opv2v-a1-four-weather-v1"
POINT = previous.OUT/"20260906/point/gspr_best.pth"
MAX_EPOCHS, MIN_EPOCHS, PATIENCE = 20, 8, 5
THRESHOLDS = (.3,.5,.7)


def write(path, value):
    previous.save(path,value)


def bind(path,value):
    if path.exists() and previous.read(path) != value:
        raise RuntimeError("Experiment identity changed: "+str(path))
    if not path.exists():
        write(path,value)


def loader(dataset, collate, size, seed, workers=4, sampler=None):
    return DataLoader(dataset,batch_size=size,shuffle=False,sampler=sampler,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=workers,collate_fn=collate)


def point_forward(model,batch,device):
    inputs = [batch[name].to(device) for name in ("voxel_features","voxel_num_points","voxel_coords")]
    from gspr_supervision.losses import evidential_point_loss
    return evidential_point_loss(model.gspr(*inputs),batch["point_targets"].to(device))[0]


class Residual:
    def __init__(self, model, device):
        self.sums = torch.zeros(3,device=device,dtype=torch.float64)
        self.handle = model.a1.register_forward_hook(self.add) if hasattr(model,"a1") else None

    def add(self,module,args,out):
        base = args[1].detach().float()
        delta = out.detach().float()-base
        self.sums += torch.stack((base.square().sum(),delta.square().sum(),
                                  delta.new_tensor(delta.numel()))).double()

    def close(self):
        if self.handle:
            self.handle.remove()
        base,delta,n = self.sums.tolist()
        return dict(rms=math.sqrt(delta/n) if n else 0., relative_rms=math.sqrt(delta/base) if base else 0.)


def finite_step(loss, model, optimizer):
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    a1 = torch.stack([p.grad.square().sum() for p in model.a1.parameters()
                      if p.grad is not None]).sum().sqrt() if hasattr(model,"a1") else norm.new_zeros(())
    optimizer.step()
    return float(a1)


def evaluate(models, dataset, device, seed, need_loss=None, limit=None):
    from opencood.tools import inference_utils,train_utils
    from opencood.utils import eval_utils
    stats = {name:{t:dict(tp=[],fp=[],gt=0,score=[]) for t in THRESHOLDS} for name in models}
    losses = []
    joint.seed_everything(seed)
    for model in models.values():
        model.eval()
    count = 0
    with torch.inference_mode():
        for batch in loader(dataset,dataset.collate_batch_test,1,seed):
            data = train_utils.to_device(batch,device)
            reference = None
            for name,model in models.items():
                if hasattr(model,"a1"):
                    model.enabled = name != "M1_off"
                # One detector forward, with the same historical postprocessor for AP.
                output = model(data["ego"])
                if need_loss is not None:
                    loss = need_loss(output,data["ego"]["label_dict"])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite validation loss")
                    losses.append(float(loss))
                boxes,scores,gt = dataset.post_process(data,{"ego":output})
                if reference is None:
                    reference = gt.clone()
                elif not torch.equal(reference,gt):
                    raise ValueError("Paired evaluation GT differs")
                if boxes is not None and (not torch.isfinite(boxes).all() or not torch.isfinite(scores).all()):
                    raise ValueError("Nonfinite prediction")
                for t in THRESHOLDS:
                    eval_utils.caluclate_tp_fp(boxes,scores,gt,stats[name],t)
            count += 1
            if count % 250 == 0:
                print("EVAL",count,len(dataset),flush=True)
            if count == limit:
                break
    if count != (len(dataset) if limit is None else limit):
        raise ValueError("Evaluation incomplete")
    result = {}
    if limit is None:
        result = {name:{f"ap{round(100*t)}":float(eval_utils.calculate_ap(copy.deepcopy(stat),t,False)[0])
                        for t in THRESHOLDS} for name,stat in stats.items()}
    return dict(frames=count,ap=result,loss=sum(losses)/len(losses) if losses else None)


def make_model(arm,seed,hypes,base,device):
    from attfuse_gspr.point_pillar_gspr_attfuse import PointPillarGsprAttfuse
    from opv2v_adapter import Model
    joint.seed_everything(seed)
    # runtime.install patches the module symbol; Model's base is the frozen original.
    cls = Model if arm=="M1" else Model.__bases__[0]
    model = cls(hypes["model"]["args"])
    missing,unexpected = model.load_state_dict(joint.load_state(base),strict=False)
    if unexpected or set(missing)!={k for k in model.state_dict() if k.startswith("a1.")}:
        raise ValueError("Detection initialization mismatch")
    model.gspr.load_state_dict(joint.load_state(POINT),strict=True)
    return model.to(device)


def state_hash(model):
    h = hashlib.sha256()
    for name,tensor in model.state_dict().items():
        if not name.startswith("a1."):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def smoke(root,hypes,base,device,training,validation,point_data):
    from opencood.loss.point_pillar_loss import PointPillarLoss
    from opencood.tools import train_utils
    det = PointPillarLoss(hypes["loss"]["args"]).to(device)
    models = {arm:make_model(arm,20260906,hypes,base,device) for arm in ("M0","M1")}
    if state_hash(models["M0"])!=state_hash(models["M1"]):
        raise ValueError("Shared initial weights differ")
    seed = 20260906
    index = 0
    samples = Mixed(training,[(index,w) for w in WEATHERS],seed)
    from gspr_supervision.dataset import collate_voxel_samples
    point = collate_voxel_samples([point_data[0]])
    result = {}
    for arm,model in models.items():
        model.train()
        joint.configure_detector_stage(model,True)
        opt = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)
        for pos,weather in enumerate(WEATHERS):
            batch,tags = samples.collate([samples[pos]])
            data = train_utils.to_device(batch,device)["ego"]
            residue = Residual(model,device)
            opt.zero_grad(set_to_none=True)
            # Detection loss alone isolates the gradient into A1.
            loss = det(model(data),data["label_dict"])
            before = {k:p.detach().clone() for k,p in model.named_parameters() if k.startswith("a1.")}
            gradient = finite_step(loss,model,opt)
            updated = sum(not torch.equal(before[k],p) for k,p in model.named_parameters() if k in before)
            if arm=="M1" and (gradient<=0 or updated==0):
                raise ValueError("No weather detection update: "+weather)
            row = dict(loss=float(loss),gradient=gradient,updated=updated,residual=residue.close(),counts=dict(tags))
            result[arm+"/"+weather] = row
            print("SMOKE_DETECTION",arm,weather,json.dumps(row),flush=True)
        opt.zero_grad(set_to_none=True)
        value = point_forward(model,point,device)
        value.backward()
        if hasattr(model,"a1") and any(p.grad is not None for p in model.a1.parameters()):
            raise ValueError("Point supervision reached A1")
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in model.gspr.parameters()):
            raise ValueError("No finite nonzero GSPR point gradient")
        print("SMOKE_POINT",arm,float(value),flush=True)
    for w in WEATHERS:
        evaluate(dict(M0=models["M0"],M1=models["M1"],M1_off=models["M1"]),validation[w],device,seed,limit=1)
        print("SMOKE_PAIRED",w,flush=True)
    write(root/"SMOKE.json",dict(device=str(device),detection=result,paired_weather=list(WEATHERS)))
    print("SMOKE_PASS",flush=True)


def train_epoch(model,optimizer,criterion,training,point_data,point_sampler,device,seed,epoch):
    from opencood.tools import train_utils
    from gspr_supervision.dataset import collate_voxel_samples
    joint.seed_everything(seed+epoch)
    for data in training.values():
        if hasattr(data,"set_weather_augmentation_epoch"):
            data.set_weather_augmentation_epoch(epoch-1)
    schedule,expected = plan(len(training["clean"]),seed,epoch)
    mixed = Mixed(training,schedule,seed+epoch*100003)
    points = loader(point_data,collate_voxel_samples,1,seed+epoch,sampler=point_sampler)
    point_data.set_epoch(epoch-1)
    point_sampler.set_epoch(epoch-1)
    point_iterator = iter(points)
    model.train()
    joint.configure_detector_stage(model,epoch==1)
    residue = Residual(model,device)
    actual = Counter()
    det_sum = point_sum = grad_sum = 0.
    for step,(batch,counts) in enumerate(loader(mixed,mixed.collate,2,seed+epoch),1):
        actual.update(counts)
        point = next(point_iterator,None)
        if point is None:
            point_iterator = iter(points)
            point = next(point_iterator)
        data = train_utils.to_device(batch,device)["ego"]
        optimizer.zero_grad(set_to_none=True)
        det = criterion(model(data),data["label_dict"])
        rel = point_forward(model,point,device)
        grad_sum += finite_step(det+rel,model,optimizer)
        det_sum += float(det)
        point_sum += float(rel)
        if step % 100 == 0:
            print("TRAIN",epoch,step,len(schedule)//2,dict(actual),flush=True)
    if dict(actual)!=expected:
        raise ValueError(f"Realized weather counts differ: {actual} vs {expected}")
    return dict(detection_loss=det_sum/step,point_loss=point_sum/step,a1_gradient_mean=grad_sum/step,
                residual=residue.close(),weather_counts=expected,
                schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest())

def fit(root,hypes,base,device,training,validation,point_data,point_val,seed):
    from opencood.tools import train_utils
    from gspr_supervision.dataset import StratifiedGSPRSampler,collate_voxel_samples
    from gspr_supervision.losses import BinaryHistogramMetrics,evidential_point_loss

    criterion = train_utils.create_loss(hypes).to(device)
    sampler = StratifiedGSPRSampler(point_data,num_samples=len(point_data),
                                    noise_bearing_fraction=.8,seed=seed)
    point_loader = loader(point_val,collate_voxel_samples,1,seed)
    folders = {arm:root/arm for arm in ("M0","M1")}
    for folder in folders.values():
        folder.mkdir(exist_ok=True)
    initial = {}
    for arm in folders:
        model = make_model(arm,seed,hypes,base,device)
        initial[arm] = state_hash(model)
        del model
    if initial["M0"] != initial["M1"]:
        raise ValueError("Shared parameters differ at initialization")
    bind(root/"INITIAL.json",initial)
    locked = root/"SELECTION.json"
    if not locked.exists():
        for epoch in range(1,MAX_EPOCHS+1):
            completed = {}
            for arm,folder in folders.items():
                model = make_model(arm,seed,hypes,base,device)
                optimizer = torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=1e-4)
                start,history = runtime.restore(folder,model,optimizer)
                if start < epoch:
                    if start != epoch-1:
                        raise ValueError("Epoch recovery has a gap")
                    print("START",seed,arm,epoch,flush=True)
                    metrics = train_epoch(model,optimizer,criterion,training,point_data,sampler,device,seed,epoch)
                    val = {}
                    for weather,data in validation.items():
                        val[weather] = evaluate({arm:model},data,device,seed,need_loss=criterion)
                    pv = joint.point_validation(model,point_loader,device,evidential_point_loss,
                                                BinaryHistogramMetrics)
                    score = sum(v["ap"][arm]["ap70"] for v in val.values())/4
                    path = folder/f"epoch{epoch:02d}.pth"
                    torch.save(model.state_dict(),path.with_suffix(".tmp"))
                    path.with_suffix(".tmp").replace(path)
                    row = dict(epoch=epoch,train=metrics,validation=val,point_validation=pv,
                               score=score,checkpoint_sha256=previous.sha(path))
                    history.append(row)
                    runtime.commit(folder,model,optimizer,history)
                    print("EPOCH_COMPLETE",seed,arm,json.dumps(row,allow_nan=False),flush=True)
                write(folder/"history.json",history)
                completed[arm] = history[:epoch]
                del model,optimizer
                torch.cuda.empty_cache()
            if completed["M0"][-1]["train"]["schedule_sha256"] != completed["M1"][-1]["train"]["schedule_sha256"]:
                raise ValueError("Paired sampling schedules differ")
            selected = {arm:max(rows,key=lambda row:row["score"]) for arm,rows in completed.items()}
            converged = epoch>=MIN_EPOCHS and all(epoch-row["epoch"]>=PATIENCE for row in selected.values())
            if converged or epoch==MAX_EPOCHS:
                reason = "paired_patience" if converged else "epoch_cap"
                write(locked,dict(epochs=epoch,reason=reason,selected=selected))
                break
    selection = previous.read(locked)
    models = {}
    for arm,row in selection["selected"].items():
        path = folders[arm]/f'epoch{row["epoch"]:02d}.pth'
        if previous.sha(path)!=row["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        models[arm] = make_model(arm,seed,hypes,base,device)
        models[arm].load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True)
    from opencood.data_utils.datasets import build_dataset
    results = {}
    for weather,path in previous.TESTS.items():
        result = root/f"test_{weather}.json"
        if not result.exists():
            config = copy.deepcopy(hypes)
            config["validate_dir"] = str(path)
            test = build_dataset(config,visualize=False,train=False)
            if len(test)!=2170:
                raise ValueError("Unexpected test frame count")
            metrics = evaluate(dict(**models,M1_off=models["M1"]),test,device,seed)
            for kind,left,right in (("trained_delta_pp","M1","M0"),("switch_delta_pp","M1","M1_off")):
                metrics[kind] = {k:100*(v-metrics["ap"][right][k]) for k,v in metrics["ap"][left].items()}
            write(result,metrics)
        results[weather] = previous.read(result)
    write(root/"RESULTS.json",dict(selection=selection,weather=results))
    return results


def main(root,mode,seeds):
    if mode=="smoke":
        device = torch.device("cpu")
        torch.set_num_threads(4)
    else:
        if not torch.version.hip or torch.cuda.device_count()!=1:
            raise RuntimeError("Full experiment requires one visible ROCm device")
        device = torch.device("cuda:0")
    runtime.install()
    from attfuse_gspr import loss
    sys.modules["opencood.loss.gspr_point_pillar_loss"] = loss
    from opencood.hypes_yaml.yaml_utils import load_yaml
    from gspr_supervision.dataset import GSPRPointDataset
    inputs = previous.read(previous.OUT/"inputs.json")
    for name,sha in inputs["sources"].items():
        if previous.sha(Path(name)) != sha:
            raise ValueError("Original source changed: "+name)
    base = Path(inputs["baseline"])
    if previous.sha(base)!=inputs["baseline_sha256"]:
        raise ValueError("Historical initialization changed")
    point_sha = previous.sha(POINT)
    if point_sha != previous.read(previous.OUT/"point_20260906.json")["sha256"]:
        raise ValueError("Point pretraining weight changed")
    hypes = load_yaml(str(previous.CONFIG))
    if hypes.get("weather_augmentation") or hypes.get("train_sim"):
        raise ValueError("Unexpected online augmentation configuration")
    roots = dict(train=Path(hypes["root_dir"]),validate=Path(hypes["validate_dir"]))
    weather_root = Path(os.environ.get("WEATHER_ROOT",str(DATA/"opv2v-physics-fixed-v1"))).resolve()
    if not weather_root.exists():
        if mode=="smoke":
            raise RuntimeError("Weather data missing. Full entry can materialize the existing physics recipe.")
        subprocess.run([sys.executable,"-B","-m","opencood.tools.materialize_physics_weather",
                        "--frontend-config",str(previous.CONFIG),"--output",str(weather_root)],cwd=PROJECT,check=True)
    identity = scan(roots,weather_root)
    for name,sha in inputs["point_files"].items():
        if previous.sha(Path(name))!=sha:
            raise ValueError("Point supervision input changed: "+name)
    contract = dict(data=identity,weather_root=str(weather_root),point_sha256=point_sha,
                    original_inputs_sha256=previous.sha(previous.OUT/"inputs.json"),
                    source={p.name:previous.sha(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py",".sh")},
                    selection="max mean AP70 of four validation splits, earliest tie",
                    convergence=dict(min_epochs=MIN_EPOCHS,patience=PATIENCE,max_epochs=MAX_EPOCHS),
                    target_mix=[.60,.1333,.1333,.1334])
    bind(root/"CONTRACT.json",contract)
    training = datasets(hypes,roots,weather_root,"train")
    validation = datasets(hypes,roots,weather_root,"validate")
    _,counts = plan(len(training["clean"]),20260906,1)
    print("FRAME_COUNTS",counts,flush=True)
    geometry = hypes["model"]["args"]
    def point(split,seed):
        return GSPRPointDataset([str(previous.CORPUS/(split+"_manifest.jsonl"))],
                                geometry["voxel_size"],geometry["lidar_range"],seed=seed)
    if mode=="smoke":
        smoke(root,hypes,base,device,training,validation,point("train",20260906))
        return
    all_results = {}
    for seed in seeds:
        folder = root/str(seed)
        folder.mkdir(exist_ok=True)
        all_results[str(seed)] = fit(folder,hypes,base,device,training,validation,
                                     point("train",seed),point("val",seed+1),seed)
    import statistics
    aggregate = {}
    for weather in WEATHERS:
        aggregate[weather] = {}
        for field in ("ap","trained_delta_pp","switch_delta_pp"):
            values = [r[weather][field] for r in all_results.values()]
            if field=="ap":
                aggregate[weather][field] = {
                    arm:{key:summary([v[arm][key] for v in values],statistics) for key in values[0][arm]}
                    for arm in values[0]}
            else:
                aggregate[weather][field] = {k:summary([v[k] for v in values],statistics) for k in values[0]}
    write(root/"SUMMARY.json",dict(seeds=list(seeds),aggregate=aggregate,results=all_results))
    print("COMPLETE",root/"SUMMARY.json",flush=True)


def summary(values,statistics):
    return dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=("smoke","full"))
    parser.add_argument("--seeds",nargs="+",type=int,default=[20260906])
    args = parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds) or any(s not in (20260906,20260907,20260908) for s in args.seeds):
        parser.error("Use distinct registered seeds")
    root = OUT if args.mode=="full" else OUT.with_name(OUT.name+"-smoke")
    root.mkdir(parents=True,exist_ok=True)
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main(root,args.mode,args.seeds)
        except Exception:
            (root/"FAILURE.txt").write_text(traceback.format_exc())
            raise

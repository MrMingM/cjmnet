"""Resume one registered seed and run its same-budget control."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "opv2v_gspr_a1_v2"
sys.path.insert(0, str(SOURCE))
import pipeline as original

OUT = Path("/data/cjm/datasets/logs/opv2v-gspr-a1-seed20260906-epoch10-v1")
SEED = "20260906"
OLD = original.OUT / SEED / "joint"


def replace_option(args, name, value):
    args = list(args)
    args[args.index(name)+1] = str(value)
    return args


def bind(path, value):
    if path.exists():
        if original.read(path) != value:
            raise ValueError("Input or stage identity changed: " + str(path))
    else:
        original.save(path, value)


def run(name, program, arguments, result=None):
    args = list(map(str, arguments))
    receipt = OUT / (name+".json")
    request = dict(program=program, arguments=args, contract=original.sha(OUT/"CONTRACT.json"))
    if receipt.exists():
        bind(receipt, dict(request=request, sha256=original.sha(result) if result else None))
        print("REUSE",name,flush=True)
        return
    print("START",name,flush=True)
    with (OUT/(name+".log")).open("a") as log:
        subprocess.run([sys.executable,"-u","-B",str(HERE/program),*args],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    bind(receipt,dict(request=request,sha256=original.sha(result) if result else None))
    print("COMPLETE",name,flush=True)


def seed_resume(target):
    target.mkdir(parents=True,exist_ok=True)
    if (target/"resume.pth").exists():
        return
    for epoch in range(1,6):
        name = f"net_epoch{epoch}.pth"
        shutil.copyfile(OLD/name,target/name)
    temporary = target/"seed_resume.tmp"
    shutil.copyfile(OLD/"resume.pth",temporary)
    temporary.replace(target/"resume.pth")


def verify_original():
    import torch
    print("CHECK original data/source inventory",flush=True)
    inputs = original.preflight()
    registered = original.read(original.OUT/f"joint_{SEED}.json")
    if registered["contract"]["inputs"] != original.sha(original.OUT/"inputs.json"):
        raise ValueError("Original joint receipt references other inputs")
    if original.sha(OLD/"net_best_validation.pth") != registered["sha256"]:
        raise ValueError("Original joint completion weight changed")
    arguments = registered["contract"]["args"]
    point = Path(arguments[arguments.index("--gspr-checkpoint")+1])
    baseline = Path(arguments[arguments.index("--baseline-checkpoint")+1])
    if original.sha(point) != original.read(original.OUT/f"point_{SEED}.json")["sha256"]:
        raise ValueError("Point pretraining checkpoint mismatch")
    if original.sha(baseline) != inputs["baseline_sha256"]:
        raise ValueError("Detection initialization checkpoint mismatch")
    state = torch.load(OLD/"resume.pth",map_location="cpu",weights_only=False)
    history = original.read(OLD/"history.json")
    fifth = torch.load(OLD/"net_epoch5.pth",map_location="cpu",weights_only=True)
    if ([row["epoch"] for row in history] != list(range(1,6)) or history != state["history"]
            or state["model"].keys() != fifth.keys()
            or any(not torch.equal(t,fifth[k]) for k,t in state["model"].items())
            or not state["optimizer"]["state"]):
        raise ValueError("Original recovery file does not match the completed fifth epoch")
    pins = [OLD/"resume.pth",OLD/"history.json",baseline,point,original.OUT/"inputs.json"]
    pins += [OLD/f"net_epoch{i}.pth" for i in range(1,6)]
    pins += sorted(HERE.glob("*.py")) + [HERE/"run_all.sh"]
    bind(OUT/"CONTRACT.json",dict(seed=SEED,args=arguments,epochs=10,checkpoints=[5,10],
                                 files={str(p):original.sha(p) for p in pins}))
    print("CHECK complete epoch 5 recovery and matching initialization",flush=True)
    return arguments,baseline


def main():
    args,baseline = verify_original()
    args = replace_option(args,"--epochs",10)
    a1 = OUT/"a1"
    gs = OUT/"gspr_only"
    seed_resume(a1)
    smoke_a1 = OUT/"smoke_a1"
    seed_resume(smoke_a1)
    for model,folder in (("a1",smoke_a1),("gspr",OUT/"smoke_gspr")):
        run("smoke_"+model,"train.py",["--model",model,"--smoke",*replace_option(args,"--run-dir",folder)])
    pair_smoke = OUT/"PAIR_SMOKE.json"
    run("smoke_pair","pair_eval.py",["--a1",a1/"net_epoch5.pth","--gspr",baseline,
         "--config",original.CONFIG,"--weather","clean","--smoke","--output",pair_smoke],pair_smoke)
    print("ACCEPTANCE passed",flush=True)
    for model,folder in (("a1",a1),("gspr",gs)):
        run("train_"+model,"train.py",["--model",model,*replace_option(args,"--run-dir",folder)],
            folder/"history.json")
        history = original.read(folder/"history.json")
        if [row["epoch"] for row in history] != list(range(1,11)):
            raise ValueError("Training did not complete ten epochs")
    results = {}
    for epoch in (5,10):
        results[str(epoch)] = {}
        for weather in original.TESTS:
            result = OUT / f"AP70_epoch{epoch}_{weather}.json"
            run(f"eval_{epoch}_{weather}","pair_eval.py",["--a1",a1/f"net_epoch{epoch}.pth",
                "--gspr",gs/f"net_epoch{epoch}.pth","--config",original.CONFIG,
                "--weather",weather,"--output",result],result)
            item = original.read(result)
            if item["hashes"] != {"a1":original.sha(a1/f"net_epoch{epoch}.pth"),
                                  "gspr":original.sha(gs/f"net_epoch{epoch}.pth")}:
                raise ValueError("Paired result checkpoint mismatch")
            results[str(epoch)][weather] = item
    original.save(OUT/"AP70_REPORT.json",results)
    table = ["epoch\tweather\tA1_on_%\tA1_off_%\tGSPR_only_%\tswitch_pp\ttrained_pp"]
    for epoch,conditions in results.items():
        for weather,row in conditions.items():
            values = [100*row["ap70"][key] for key in ("a1_on","a1_off","gspr_only")]
            values += [row["switch_gain_pp"],row["trained_gain_pp"]]
            table.append("\t".join([epoch,weather,*[f"{x:.6f}" for x in values]]))
    (OUT/"AP70_REPORT.tsv").write_text("\n".join(table)+"\n")
    print("COMPLETE",OUT/"AP70_REPORT.tsv",flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main()
        except Exception:
            (OUT/"FAILURE.txt").write_text(traceback.format_exc())
            raise

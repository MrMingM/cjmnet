import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = Path("/data/cjm/datasets/logs/opv2v-gspr-a1-v2")
CORPUS = Path("/data/cjm/datasets/gspr_opv2v_weather_full_v1")
BASE = Path("/data/cjm/datasets/logs/gspr_attfuse_opv2v_20260904_154404/net_epoch11.pth")
CONFIG = PROJECT / "gspr_attfuse_config.yaml"
OPV2V = Path("/data/scd/datasets/opv2v_official_data_dumping")
TESTS = {"clean": OPV2V / "test", **{w: Path("/data/cjm/datasets/opv2v-w") / w / "test" for w in ("fog","rain","snow")}}
SEEDS = (20260906, 20260907, 20260908)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(folder):
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Empty dataset {folder}")
    h = hashlib.sha256()
    for path in files:
        info = path.stat()
        h.update(json.dumps([str(path.relative_to(folder)), info.st_size, info.st_mtime_ns]).encode())
    return dict(path=str(folder), files=len(files), inventory_sha256=h.hexdigest(), identity_scope="relative path, byte size and mtime_ns; no detection payload hash")


def preflight():
    import torch
    import numpy as np
    import yaml
    if not torch.version.hip or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible ROCm GPU")
    config = yaml.safe_load(CONFIG.read_text())
    if Path(config["root_dir"]) != OPV2V/"train" or Path(config["validate_dir"]) != OPV2V/"validate":
        raise ValueError("Historical OPV2V roots changed")
    if config.get("weather_augmentation") or config.get("train_sim"):
        raise ValueError("Unexpected detection weather augmentation")
    sources = list(ROOT.glob("*.py")) + [ROOT/"run_all.sh", ROOT/"historical.json", CONFIG]
    for package in ("attfuse_gspr", "attfuse_code", "gspr_supervision"):
        sources.extend((PROJECT/package).rglob("*.py"))
    runtime = next(p for p in (PROJECT, PROJECT.parent) if (p/"opencood").is_dir())
    sources.extend((runtime/"opencood").rglob("*.py"))
    pins = {str(p): sha(p) for p in sorted(sources)}
    existing = OUT/"inputs.json"
    if existing.exists():
        result = read(existing)
        if result["sources"] != pins or result["baseline_sha256"] != sha(BASE):
            raise ValueError("Source or baseline changed since preflight")
        for name, expected in result["point_files"].items():
            if sha(Path(name)) != expected:
                raise ValueError("Changed point input: " + name)
        for name, item in result["datasets"].items():
            print("RECHECK DATA INVENTORY", name, flush=True)
            if inventory(Path(item["path"])) != item:
                raise ValueError("Changed detection input: " + name)
        if (result["torch"], result["hip"], result["gpu"]) != (torch.__version__, torch.version.hip, torch.cuda.get_device_name(0)):
            raise ValueError("ROCm environment changed")
        return result
    point_files = {}
    groups = {}
    seen = set()
    for split, expected in (("train",4644), ("val",972)):
        manifest = CORPUS/(split+"_manifest.jsonl")
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        if len(rows) != expected:
            raise ValueError(f"Unexpected {split} manifest count {len(rows)}")
        point_files[str(manifest)] = sha(manifest)
        groups[split] = {}
        for index, row in enumerate(rows):
            path = Path(row["path"]).resolve(strict=True)
            if CORPUS.resolve() not in path.parents or path in seen:
                raise ValueError(f"Duplicate or external point input {path}")
            seen.add(path)
            with np.load(path, allow_pickle=False) as data:
                points, labels = data["points"], data["reliability"]
                if (points.shape != (row["output_points"],4) or labels.size != len(points)
                        or not np.isfinite(points).all() or not np.isin(labels,[0,1]).all()
                        or int((labels==0).sum()) != row["noise_points"]):
                    raise ValueError(f"Invalid point supervision {path}")
            point_files[str(path)] = sha(path)
            key = row["weather"]+"/"+row["severity"]
            groups[split][key] = groups[split].get(key,0)+1
            if index % 250 == 0:
                print("POINT AUDIT", split, index, len(rows), flush=True)
        if len(groups[split]) != 9:
            raise ValueError("Expected nine weather/severity groups")
    datasets = {}
    for key, path in {"train":OPV2V/"train","validation":OPV2V/"validate", **TESTS}.items():
        print("DATA INVENTORY", key, path, flush=True)
        datasets[key] = inventory(path)
    result = dict(sources=pins, baseline=str(BASE), baseline_sha256=sha(BASE),
                  point_files=point_files, point_groups=groups, datasets=datasets,
                  torch=torch.__version__, hip=torch.version.hip, gpu=torch.cuda.get_device_name(0))
    save(existing,result)
    return result


def stage(name, script, args, artifact, env=None):
    print("START",name,flush=True)
    receipt = OUT/(name+".json")
    contract = dict(args=[str(x) for x in args], inputs=sha(OUT/"inputs.json"))
    if receipt.exists():
        if read(receipt) != dict(contract=contract, sha256=sha(artifact)):
            raise ValueError("Stage identity changed: "+name)
        print("REUSE",name,flush=True)
        return
    os.environ.pop("A1_SMOKE", None)
    with (OUT/(name+".log")).open("a") as stream:
        subprocess.run([sys.executable,"-B",str(ROOT/script), *contract["args"]],
                       stdout=stream,stderr=subprocess.STDOUT,env=env,check=True)
    save(receipt,dict(contract=contract,sha256=sha(artifact)))
    print("COMPLETE",name,flush=True)


def main():
    preflight()
    train = CORPUS/"train_manifest.jsonl"
    val = CORPUS/"val_manifest.jsonl"
    common = ["--train-manifest",train,"--val-manifest",val,"--epochs",5,
              "--num-workers",4,"--noise-bearing-fraction",.8]
    joint = ["--config",CONFIG,"--baseline-checkpoint",BASE,*common,
             "--detection-batch-size",2,"--point-batch-size",1,"--max-train-steps",0,
             "--lr",.00002,"--lambda-point",1,"--point-sampling","stratified",
             "--freeze-detector-epochs",1,"--min-point-macro-auroc",.995]
    smoke = OUT/"smoke"
    stage("smoke","joint.py",[*joint,"--run-dir",smoke,"--gspr-checkpoint",BASE,
                            "--seed",SEEDS[0]],smoke/"SMOKE.json",dict(os.environ,A1_SMOKE="1"))
    if os.environ.get("A1_MODE") == "smoke":
        return
    runs = {}
    for seed in SEEDS:
        point = OUT/str(seed)/"point"
        model = OUT/str(seed)/"joint"
        stage(f"point_{seed}","point.py",[*common,"--output-dir",point,
              "--init-attfuse-checkpoint",BASE,"--batch-size",1,"--lr",.0002,
              "--max-voxels",32000,"--sampling","stratified","--seed",seed],point/"gspr_best.pth")
        weights = model/"net_best_validation.pth"
        stage(f"joint_{seed}","joint.py",[*joint,"--run-dir",model,
              "--gspr-checkpoint",point/"gspr_best.pth","--seed",seed],weights)
        results = {}
        for weather in TESTS:
            dest = OUT/str(seed)/"evaluation"/weather
            stage(f"eval_{seed}_{weather}","evaluate.py",["--config",CONFIG,"--checkpoint",weights,
                  "--output-root",dest,"--conditions",weather],dest/"summary.json")
            results[weather] = read(dest/"summary.json")["results"][weather]
            if results[weather]["samples"] != 2170:
                raise ValueError("Unexpected OPV2V-W frame count")
        runs[str(seed)] = dict(results=results,weights=str(weights),sha256=sha(weights),
                              selection=read(model/"best_selection.json"),history=read(model/"history.json"))
    historical = read(ROOT/"historical.json")
    lines = ["# OPV2V GSPR＋A1","",
             "历史基线来自 AI_CONTEXT.md §5、§7。20260906 历史 pilot 的种子设置与后两次不同。本轮统一入口，差值是与历史均值的描述性比较。","",
             "| 天气 | 指标 | AttFuse | GSPR 均值 ± 标准差 | GSPR＋A1 均值 ± 标准差 | Δ GSPR（百分点） |",
             "|---|---|---:|---:|---:|---:|"]
    for w in TESTS:
        for i,key in enumerate(("ap30","ap_50","ap_70")):
            values = [run["results"][w][key] for run in runs.values()]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            old, old_sd = historical[w]["gspr"][i]
            lines.append(f"| {w} | AP{(30,50,70)[i]} | {historical[w]['attfuse'][i]:.4f} | {old:.6f} ± {old_sd:.6f} | {mean:.6f} ± {sd:.6f} | {100*(mean-old):+.3f} |")
    (OUT/"FINAL_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    save(OUT/"FINAL_RESULTS.json",dict(runs=runs,historical=historical,inputs=read(OUT/"inputs.json")))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            main()
        except Exception:
            (OUT/"FAILURE.txt").write_text(traceback.format_exc())
            raise

"""D0-D7 serial entry; CPU generation, isolated native IO check, validate-only GPU."""
import argparse
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import os
import platform
import signal
import subprocess
import sys
import traceback
from scorw_dataset.core import (HERE,REPO,SPLITS,server_only,source_identity,load_config,read_json,
    atomic_json,atomic_text,atomic_bytes,digest,identity,lock,begin,transaction)


def dependencies():
    result={}
    for name in ("numpy","scipy","PyYAML","open3d","torch","spconv","cumm"):
        try:result[name]=version(name)
        except PackageNotFoundError:result[name]=None
    return result


def publish(folder,root):
    for p in folder.iterdir():
        if p.is_file() and p.suffix in {".json",".md"} and p.name not in {"CONTRACT.json","COMMIT.json"}:
            # Full D0 input manifests stay in the stage directory; their hashes/paths
            # are in SOURCE_INVENTORY. Avoid copying large inventories to small reports.
            if p.name.endswith("_SOURCE.json"):continue
            atomic_bytes(root/p.name,p.read_bytes())


def run(args):
    server_only()
    from scorw_dataset import inventory,performance,tests,generation,report
    from scorw_repair.detection import model_identity
    c=load_config();code=source_identity();deps=dependencies()
    if args.workers<1 or args.workers>min(c["performance"]["max_workers"],os.cpu_count() or 1):raise ValueError("WORKERS exceeds available/frozen CPU limit")
    expected=Path("/data/scd/datasets/opv2v_official_data_dumping").resolve(strict=True)
    if Path(c["source_root"]).resolve(strict=True)!=expected:raise ValueError("Unexpected original OPV2V root")
    root=Path(c["output_root"]+({"full":"","smoke":"-smoke","preflight":"-preflight"}[args.mode]))
    if root.is_symlink() or expected==root or expected in root.resolve().parents or REPO in root.resolve().parents:raise ValueError("Unsafe output root")
    model=model_identity(c)
    if model.get("status")!="verified":raise ValueError("Frozen AttFuse model/config unavailable: "+str(model))
    # All CPU stages, including imports and native compatibility, see no GPU.
    os.environ["ROCR_VISIBLE_DEVICES"]="";os.environ.pop("HIP_VISIBLE_DEVICES",None);os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    request=dict(config=c,config_sha256=digest(HERE/"config.json"),sources=code,mode=args.mode,workers=args.workers,
        requested_inference_GPU=args.gpu,model=model,runtime=dict(python=platform.python_version(),dependencies=deps,
            threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}))
    with lock(root):
        parent=begin(root,request,args.resume=="1")
        journal=root/"PUBLICATION_IN_PROGRESS.json"
        if journal.exists() and read_json(journal).get("contract")!=parent:raise ValueError("Publication contract changed")
        if not journal.exists() and (root/"REPORT_MANIFEST.json").exists():
            for n,r in read_json(root/"REPORT_MANIFEST.json").items():
                if digest(root/n)!=r["sha256"]:raise ValueError("Published report changed: "+n)
        atomic_json(journal,dict(contract=parent))
        atomic_json(root/"DATASET_PROTOCOL.json",request)
        gates=dict(status="RUNNING",mode=args.mode,contract=parent,stages={},REAL_WORLD_VALIDATION_MISSING=True)
        active="D0"
        def phase(name,callback):
            folder=root/"stages"/name
            if source_identity()["identity"]!=code["identity"]:raise ValueError("Source changed before "+name)
            print("START "+name,flush=True)
            def checked(k):
                callback(folder,k)
                if source_identity()["identity"]!=code["identity"]:raise ValueError("Source changed during "+name)
            transaction(folder,dict(parent=parent,stage=name),checked)
            publish(folder,root)
            gates["stages"][name]=dict(status="passed",commit_sha256=digest(folder/"COMMIT.json"))
            atomic_json(root/"VALIDATION_GATES.json",gates)
            print("PASS "+name,flush=True)
            return folder
        try:
            def preflight(p,k):
                subprocess.run([sys.executable,"-u","-B","-m","scorw_dataset.compatibility","--root",str(root),"--out",str(p),"--imports-only"],check=True)
                inventory.run(c,p)
                budget=performance.run(c,p,p,args.workers)
                if budget["status"]!="passed":raise ValueError("RESOURCE_BLOCKED: "+"; ".join(budget["reasons"])+"; see GENERATION_BUDGET.json; no dataset size or physics reduced")
            d0=phase("D0",preflight)
            inventory.verify_source_tree(c,d0)
            if args.mode=="preflight":gates["status"]="PREFLIGHT_READY";return 0
            smoke=args.mode=="smoke"
            for split in SPLITS:
                d=inventory.selected(read_json(d0/(split.upper()+"_SOURCE.json")),c,smoke)
                dest=root/"manifests"/(split.upper()+"_SELECTION.json")
                if dest.exists() and read_json(dest)!=d:raise ValueError("Frozen split selection changed")
                atomic_json(dest,d)
            active="D1"
            phase(active,lambda p,k:tests.run(c,d0,p))
            for name,split in zip(("D2","D3","D4"),SPLITS):
                active=name;data=read_json(root/"manifests"/(split.upper()+"_SELECTION.json"))
                def generate(p,k):
                    m=generation.generate_split(c,root,data,parent,args.workers,smoke)
                    atomic_json(p/"SPLIT_COMMIT.json",dict(split=split,manifest_sha256=digest(root/(split.upper()+"_MANIFEST.json")),pcds=m["pcds"]))
                phase(name,generate)
                # Existing stage commits never bypass full external output validation.
                generation.audit_split(c,root,data,parent)
            active="D5"
            def compatibility(p,k):
                subprocess.run([sys.executable,"-u","-B","-m","scorw_dataset.compatibility","--root",str(root),"--out",str(p)],check=True)
                if read_json(p/"COMPATIBILITY_AUDIT.json")["status"]!="passed":raise ValueError("D5 failed")
            phase(active,compatibility)
            active="D6"
            def evaluate(p,k):
                env=dict(os.environ,ROCR_VISIBLE_DEVICES=str(args.gpu));env.pop("HIP_VISIBLE_DEVICES",None);env.pop("CUDA_VISIBLE_DEVICES",None)
                subprocess.run([sys.executable,"-u","-B","-m","scorw_dataset.evaluate_cli","--root",str(root),"--out",str(p),"--gpu",str(args.gpu),"--parent",k],env=env,check=True)
                if read_json(p/"VALIDATION_ATTFUSE_RESULTS.json")["status"]!="passed":raise ValueError("D6 failed")
            phase(active,evaluate)
            active="D7"
            inventory.verify_source_tree(c,d0)
            for split in SPLITS:generation.audit_split(c,root,read_json(root/"manifests"/(split.upper()+"_SELECTION.json")),parent)
            def summarize(p,k):
                qa=report.run(c,root,smoke,p)
                atomic_json(p/"FINAL_COMMIT.json",dict(status=qa["status"],dataset_QA_sha256=digest(p/"DATASET_QA.json"),
                    README_sha256=digest(p/"README.md"),config_identity_sha256=digest(root/"configs/CONFIG_IDENTITIES.json")))
            final=phase(active,summarize);receipt=read_json(final/"FINAL_COMMIT.json")
            for name,key in (("DATASET_QA.json","dataset_QA_sha256"),("README.md","README_sha256"),("configs/CONFIG_IDENTITIES.json","config_identity_sha256")):
                if digest(root/name)!=receipt[key]:raise ValueError("Final publication identity changed: "+name)
            for name,h in read_json(root/"configs/CONFIG_IDENTITIES.json")["generated"].items():
                if digest(root/name)!=h:raise ValueError("Derived model configuration changed: "+name)
            gates["status"]="SMOKE_COMPLETE" if smoke else "DATASET_COMPLETE"
            return 0
        except (Exception,KeyboardInterrupt) as error:
            gates.update(status="NOT_COMPLETE",failed_stage=active,error_type=type(error).__name__,reason=str(error),
                locations=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
            print("STOP "+active+": "+type(error).__name__+": "+str(error),flush=True)
            current=root/"stages"/active
            if current.is_dir():publish(current,root)
            atomic_text(root/"FINAL_RESULTS.md","# SCOR-W-Fog v1\n\nNOT_COMPLETE\n\n"+str(error)+"\n\nCompleted valid clouds retained; no failure skipped.\n")
            return 3
        finally:
            try:
                if source_identity()["identity"]!=code["identity"] or model_identity(c)!=model:
                    gates.update(status="NOT_COMPLETE",reason="SOURCE_OR_MODEL_CHANGED_AT_END")
            except Exception as error:
                gates.update(status="NOT_COMPLETE",reason="SOURCE_OR_MODEL_END_CHECK_FAILED: "+str(error))
            atomic_json(root/"VALIDATION_GATES.json",gates)
            if gates["status"] not in {"DATASET_COMPLETE","SMOKE_COMPLETE"}:
                atomic_json(root/"DATASET_QA.json",dict(status=gates["status"],reason=gates.get("reason"),REAL_WORLD_VALIDATION_MISSING=True))
            report.archive(root);journal.unlink()
            logs=Path(c["logs_root"]);logs.mkdir(parents=True,exist_ok=True)
            atomic_json(logs/(args.mode+"_LATEST.json"),dict(root=str(root),status=gates["status"]))
            print("Reports: "+str(root)+"; status="+gates["status"],flush=True)
            if gates["status"]=="NOT_COMPLETE" and "failed_stage" not in gates:raise ValueError(gates["reason"])


def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=("preflight","smoke","full"),required=True)
    p.add_argument("--resume",choices=("0","1"),default="0");p.add_argument("--workers",type=int,default=4);p.add_argument("--gpu",type=int,default=0)
    a=p.parse_args()
    if a.gpu<0:raise ValueError("GPU must be a physical nonnegative device number")
    signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(InterruptedError("terminated")))
    try:code=run(a)
    except Exception as error:print(type(error).__name__+": "+str(error),file=sys.stderr);code=1
    raise SystemExit(code)


if __name__=="__main__":main()

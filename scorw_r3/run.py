"""One serial, hash-checked R3-A -> R3-B entry; real IO is designated-server only."""
import argparse
from contextlib import redirect_stderr,redirect_stdout
import io
import os
from pathlib import Path
import platform
import signal
import shutil
import sys
import traceback
import unittest

from scorw_r3.core import (HERE,REPO,Blocked,read_json,digest,identity,sources,atomic_json,
    atomic_text,atomic_bytes,lock,begin,transaction,server_only,server_train,check_inputs)
from scorw_repair.run import archive,dependencies


REQUIRED_A=("R3A_PHYSICS_AUDIT.md","R3A_INTENSITY_DEPENDENCE.json","R3A_NO_RETURN_DIAGNOSIS.json",
    "R3A_HAHNER_DIFFERENCES.json","R3A_SENSITIVITY.json","R3A_RESULTS.md","R3A_GATE.json")
REQUIRED_B=("R3B_SELECTION_FREEZE.json","R3B_COMPUTE_PLAN.json","R3B_MODEL_AUDIT.json",
    "R3B_SCENE_RESULTS.json","R3B_B3_B4_COMPARISON.json","R3B_NEIGHBOR_ONLY.json",
    "R3B_TARGET_DIAGNOSIS.json","R3B_STATISTICAL_ANALYSIS.json","R3B_RESULTS.md")


def numerical_checks(out):
    from scorw_r3.tests import test_r3,test_argsfix
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_r3,test_argsfix)])
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
        errors=len(result.errors),failures=len(result.failures),log=stream.getvalue(),
        synthetic_only=True,detector_executed=False,real_world_validation=False)
    atomic_json(out/"NUMERICAL_TESTS.json",report)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise Blocked("R3_NUMERICAL_GATE_FAILED")
    return report


def publish(folder,out):
    # Only stage-level small reports; arrays, PCD and per-scene prediction caches stay on disk.
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in {".json",".md"} and p.name not in {"CONTRACT.json","COMMIT.json"}:
            atomic_bytes(out/p.name,p.read_bytes())


def validate_config(c):
    old=read_json(REPO/"scorw_repair/config.json")
    if any(c[k]!=old[k] for k in ("receiver","field","b1","model")) or c["pool_size"]!=20:
        raise Blocked("R3_FROZEN_R2_PHYSICS_OR_MODEL_CHANGED")
    if c["automatic_seed_expansion"] or c["automatic_weather_tuning"] or not c["REAL_WORLD_VALIDATION_MISSING"]:
        raise Blocked("R3_FORBIDDEN_TUNING_OR_CALIBRATION_PROMOTION")
    return c


def preflight(c,old,dep,out):
    from scorw_r3.selection import allowed_candidates
    from scorw_repair.regression import table_resource
    from scorw_r3.runtime import validate_project_first
    from opencood.hypes_yaml.yaml_utils import load_yaml
    root=server_train(c["train_root"])
    candidates,unknown=allowed_candidates(read_json(HERE/"inputs/HISTORY_FREEZE.json"),[p.name for p in root.iterdir() if p.is_dir()])
    _,table=table_resource(production=True)
    fusion_config=validate_project_first(load_yaml(c["model"]["config"]))
    problems=[]
    if not dep["required_r0_r1"] or not all(dep["modules"].get(n,False) for n in ("torch","spconv","cumm")):
        problems.append("DEPENDENCIES_MISSING")
    if not list((old.r1/"rays").rglob("rays.npz")):problems.append("R1_FIXED_SAMPLE_CACHE_MISSING")
    if not list((old.r2/"cloud_cache").rglob("response.npz")):problems.append("R2_RESPONSE_CACHE_MISSING")
    report=dict(status="ready" if not problems else "blocked",problems=problems,
        known_fresh_scene_candidates=candidates,unclassified_not_recruited=unknown,
        possible_independent_scenes_upper_bound=len(candidates),requested_scenes=c["selection"]["target_scenes"],
        source_and_old_report_hashes="verified",old_stage_receipts="resolved and verified",
        old_resolved_stages={k:str(getattr(old,k)) for k in ("coords","r1","r2")},
        old_cache_arrays_not_read=True,new_scene_clouds_not_read=True,Hahner_table=table,model=old.model,
        missing_optional_old_forensic_PCD=old.optional_missing,
        fusion_config=fusion_config,
        budget_limits=c["budget"],actual_budget="computed from Clean-qualified inputs before R3-B response generation",
        numerical_tests="deferred to smoke/full",network_requests=0,downloads=False,
        real_world_validation="MISSING",runtime_dependencies=dep)
    atomic_json(out/"PREFLIGHT.json",report)
    return report


def selection_external(stage):
    if not (stage/"R3B_SELECTION_FREEZE.json").is_file():return
    p=read_json(stage/"R3B_SELECTION_FREEZE.json")
    check_inputs(p.get("consulted_inputs",[]))
    if (stage/"coordinates/INPUT_MANIFEST.json").is_file():check_inputs(read_json(stage/"coordinates/INPUT_MANIFEST.json"))


def placeholder(out,names,reason):
    for name in names:
        if (out/name).exists():continue
        if name.endswith(".json"):
            atomic_json(out/name,dict(status="NOT_EXECUTED",reason=reason,actual_results_available=False,
                REAL_WORLD_VALIDATION_MISSING=True))
        else:atomic_text(out/name,"# "+name+"\n\nNOT_EXECUTED\n\n"+reason+"\n")


def finalize(out,c,gates):
    if gates["mode"]!="preflight":
        placeholder(out,REQUIRED_A+REQUIRED_B,gates.get("reason","prerequisite has not passed"))
        if not (out/"R3_PROJECT_DECISION.json").exists() or gates["status"] in {"failed","blocked"}:
            decision="STOP_OR_REDESIGN" if read_json(out/"R3A_GATE.json").get("status")=="BLOCKED_PHYSICS_ERROR" else "INCONCLUSIVE"
            atomic_json(out/"R3_PROJECT_DECISION.json",dict(decision=decision,reasons=[gates.get("reason","not yet evaluated")],
                formal_dataset_accepted=False,REAL_WORLD_VALIDATION_MISSING=True))
        decision=read_json(out/"R3_PROJECT_DECISION.json")
    else:decision=dict(decision="INCONCLUSIVE",reasons=["PREFLIGHT_ONLY; no research experiment executed"])
    evidence=dict(ENGINEERING_PASS=gates["status"].endswith("COMPLETE"),
        PHYSICS_NUMERICAL_PASS=gates["stages"].get("R3A",{}).get("status")=="passed",
        REAL_WORLD_VALIDATION_MISSING=True,DETECTOR_EXPLORATORY_RESULT=gates["stages"].get("detector",{}).get("status")=="passed",
        SMOKE_IS_INDEPENDENT_VALIDATION=False,formal_dataset_accepted=False,
        decision=decision["decision"],status=gates["status"])
    atomic_json(out/"EVIDENCE_LEVELS.json",evidence)
    atomic_json(out/"VALIDATION_GATES.json",gates)
    lines=["# R3 最终结果","",gates["status"],"",decision["decision"],"",*decision.get("reasons",[]),"",
        "真实雾天物理标定始终缺失。数学正确性、引擎执行和检测行为证据分别记录；检测只用于UNCALIBRATED_EXPLORATORY机制诊断。",
        "R3-A报告给出接收器假设及原版差异；R3-B报告给出全部场池、逐scene/目标结果及三种检测范围。"]
    for key in ("independent_scenes","unique_GT_instances","unique_neighbor_detectable_targets"):
        if key in decision:lines.append(f"{key}: {decision[key]}")
    if (out/"R3B_B3_B4_COMPARISON.json").exists():
        comparison=read_json(out/"R3B_B3_B4_COMPARISON.json")
        for v,d in comparison.get("comparison",{}).items():lines.append(f"{v}: 平均ΔAP70={d['mean_delta_AP70']}，正/负/零重复={d['positive_repeats']}/{d['negative_repeats']}/{d['zero_repeats']}")
    if gates.get("reason"):lines.extend(["",gates["reason"]])
    atomic_text(out/"FINAL_RESULTS.md","\n\n".join(lines)+"\n")
    atomic_text(out/"NEXT_STAGE_DECISION.md","# 下一阶段\n\n"+decision["decision"]+"\n\n"+
        "\n\n".join(decision.get("reasons",[]))+"\n\nREAL_WORLD_VALIDATION_MISSING\n\n"
        "仅在场景数与样本质量足够、对照公平、差异有跨scene稳定性且目标诊断可解释时考虑继续。"
        "未显著不能证明效应很小；不扩大天气seed、不选有利场景、不调参数寻找正结果。\n")
    archive(out)
    logs=Path(c["logs_root"]);logs.mkdir(parents=True,exist_ok=True)
    atomic_json(logs/(gates["mode"]+"_LATEST.json"),dict(output=str(out),status=gates["status"],
        archive_sha256=digest(out/"SMALL_RESULTS.tar.gz")))


def execute(args):
    server_only()
    c=validate_config(read_json(HERE/"config.json"));code=sources();dep=dependencies()
    from scorw_r3.old import OldRun,readonly_old
    from scorw_r3 import receiver_audit,selection,waves,detection,analysis,recovery
    from scorw_repair.detection import model_identity
    out=Path(c["output_root"])/args.mode;smoke=args.mode=="smoke"
    # Hardware visibility is fixed before any detector/GPU initialization.
    os.environ["ROCR_VISIBLE_DEVICES"]=str(c["gpu"])
    os.environ.pop("HIP_VISIBLE_DEVICES",None);os.environ.pop("CUDA_VISIBLE_DEVICES",None)
    with readonly_old(c["legacy_root"]),recovery.readonly_previous(c,args.mode) as previous,lock(out):
        old=OldRun(c,deep=True)
        request=dict(config=c,config_sha256=digest(HERE/"config.json"),sources=code["identity"],
            source_snapshot_sha256=code["snapshot_sha256"],mode=args.mode,old_report_manifest_sha256=old.manifest,
            old_contract=old.request["identity"],model=old.model,
            history_freeze_sha256=digest(HERE/"inputs/HISTORY_FREEZE.json"),
            missing_optional_old_forensic_PCD=old.optional_missing,
            runtime=dict(python=platform.python_version(),dependencies=dep,
                threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}),
            R3B_label="UNCALIBRATED_EXPLORATORY",REAL_WORLD_VALIDATION_MISSING=True)
        reuse=recovery.audit(previous,c,request)
        request["verified_previous_r3"]=reuse
        parent=begin(out,request,args.resume=="1")
        journal=out/"PUBLICATION_IN_PROGRESS.json"
        if journal.exists() and read_json(journal).get("contract")!=parent:raise Blocked("R3_PUBLICATION_JOURNAL_CHANGED")
        if not journal.exists() and (out/"REPORT_MANIFEST.json").exists():
            for n,r in read_json(out/"REPORT_MANIFEST.json").items():
                if digest(out/n)!=r["sha256"]:raise Blocked("R3_PUBLISHED_REPORT_CHANGED: "+n)
        atomic_json(journal,dict(contract=parent,role="mutable publication; committed stages are authoritative"))
        gates=dict(status="running",mode=args.mode,contract=parent,stages={},REAL_WORLD_VALIDATION_MISSING=True)
        atomic_json(out/"R3_PROTOCOL_FREEZE.json",request);atomic_json(out/"SOURCE_AUDIT.json",code)
        atomic_json(out/"DEPENDENCIES.json",dep)
        atomic_json(out/"OLD_CACHE_AUDIT.json",dict(required_bytes="verified",missing_optional_forensic_PCD=old.optional_missing,
            policy="missing original forensic PCD disclosed; existing altered bytes/required arrays always stop; availability frozen in new contract"))
        atomic_json(out/"R3_CACHE_REUSE_AUDIT.json",dict(status="verified" if reuse else "fresh_run",
            previous=reuse,old_contract_modified=False,old_detector_predictions_reused=False,
            additional_waveform_array_bytes=0 if reuse else None))
        active="preflight"

        def phase(name,callback):
            folder=out/"stages"/name
            if sources()["identity"]!=code["identity"]:raise Blocked("R3_SOURCE_CHANGED_BEFORE_STAGE")
            selection_external(out/"stages/selection")
            recovery.verify(reuse)
            print("START "+name,flush=True)
            def checked(stage_parent):
                if not recovery.import_stage(reuse,name,folder):callback(folder,stage_parent)
                if sources()["identity"]!=code["identity"]:raise Blocked("R3_SOURCE_CHANGED_DURING_STAGE")
                selection_external(out/"stages/selection")
            record=transaction(folder,dict(parent=parent,stage=name),checked)
            publish(folder,out)
            gates["stages"][name]=dict(status="passed",commit_sha256=digest(folder/"COMMIT.json"),contract=record["contract"])
            atomic_json(out/"VALIDATION_GATES.json",gates)
            print("PASS "+name,flush=True)
            return folder

        try:
            flight=preflight(c,old,dep,out)
            if flight["status"]!="ready":raise Blocked("R3_PREFLIGHT_FAILED: "+str(flight["problems"]))
            if args.mode=="preflight":gates["status"]="PREFLIGHT_READY";return 0
            active="numerical"
            phase(active,lambda p,k:numerical_checks(p))
            active="R3A"
            r3a=phase(active,lambda p,k:receiver_audit.run(c,old,p,smoke))
            if read_json(r3a/"R3A_GATE.json")["status"]!="PASS_WITH_DISCLOSED_APPROXIMATIONS":
                raise Blocked("R3A_NOT_ELIGIBLE_FOR_DETECTION")
            active="selection"
            def choose(p,k):
                selected=selection.select(c,old,p,smoke)
                selection.budget(c,selected,p/"coordinates",p)
            selected_root=phase(active,choose)
            selected=read_json(selected_root/"R3B_SELECTION_FREEZE.json")
            plan=read_json(selected_root/"R3B_COMPUTE_PLAN.json")
            if plan["status"]!="passed":raise Blocked("R3B_BUDGET_BLOCKED_BEFORE_RESPONSE_GENERATION: "+str(plan["failures"]))
            if not selected["frames"]:
                active="analysis"
                phase(active,lambda p,k:analysis.analyze(c,[],p,smoke))
                gates["status"]="NO_QUALIFIED_NEW_SCENES_COMPLETE";return 0
            coords=selected_root/"coordinates"
            active="waves"
            # On resume count bytes already occupying the new wave stage; the frozen
            # total budget is unchanged. Cache identities are still checked by transactions.
            existing=Path(reuse["stages"]["waves"]["root"]) if reuse else out/"stages/waves"
            occupied=sum(p.stat().st_size for p in existing.rglob("*") if p.is_file()) if existing.exists() else 0
            needed=0 if reuse else max(0,plan["estimated_cache_upper_bytes"]-occupied)
            if shutil.disk_usage(out).free<needed+c["budget"]["reserve_disk_bytes"]:
                raise Blocked("R3B_DISK_BUDGET_CHANGED_BEFORE_RESPONSE_GENERATION")
            wave_root=phase(active,lambda p,k:waves.build(c,old,selected,coords,p,k,smoke))
            wave_index=read_json(wave_root/"R3B_WAVE_INDEX.json")
            active="detector"
            def infer(p,k):
                from scorw_r3.runtime import Runtime
                if model_identity(c)!=old.model:raise Blocked("R3_MODEL_CHANGED_BEFORE_DETECTION")
                runtime=Runtime(c,old,p)
                atomic_json(p/"R3B_MODEL_AUDIT.json",runtime.model_audit)
                try:
                    detection.audit_old_cases(c,old,runtime,p,k)
                    detection.evaluate(c,old,selected,coords,wave_index,runtime,p,k)
                    atomic_bytes(p/"R3B_IOU_AUDIT.json",(p/"R2_IOU_AUDIT.json").read_bytes())
                finally:
                    runtime.finish();runtime.model.cpu();runtime.torch.cuda.empty_cache()
            detected=phase(active,infer)
            active="analysis"
            def summarize(p,k):
                refs=read_json(detected/"R3B_SCENE_RESULTS.json")["scenes"]
                results=[]
                for r in refs:
                    if digest(r["full_result_path"])!=r["full_result_sha256"]:raise Blocked("R3_SCENE_RESULTS_CHANGED")
                    results.append(read_json(r["full_result_path"]))
                analysis.analyze(c,results,p,smoke)
            phase(active,summarize)
            gates["status"]=("SMOKE_" if smoke else "FULL_")+"UNCALIBRATED_EXPLORATORY_COMPLETE"
            return 0
        except (Blocked,ValueError,AssertionError) as error:
            gates.update(status="blocked",blocked_stage=active,reason=str(error))
            if active in {"numerical","R3A"}:
                # Failed algebra/regression is a hard stop. Missing immutable artifacts are
                # reported as missing evidence, not silently promoted to a physics pass.
                physics=not isinstance(error,FileNotFoundError) and any(s in str(error) for s in
                    ("LINEARITY","ARGMAX","REGRESSION_FAILED","WAVEFORM","ZERO_ENCODING","NUMERICAL_GATE","BASELINE_REAL_RAY","Prefix/target","CONSERVATION"))
                atomic_json(out/"R3A_GATE.json",dict(status="BLOCKED_PHYSICS_ERROR" if physics else "INCONCLUSIVE_MODEL_ASSUMPTIONS",
                    reason=str(error),downstream_allowed=False,REAL_WORLD_VALIDATION_MISSING=True))
            print("STOP "+active+": "+str(error),flush=True)
            return 3
        except (Exception,KeyboardInterrupt) as error:
            gates.update(status="failed",blocked_stage=active,reason=type(error).__name__+": "+str(error),
                locations=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)])
            print("STOP "+active+": "+gates["reason"],flush=True)
            return 1
        finally:
            # Preserve completed reports and any diagnostic written immediately before a gate failed.
            explicit_a_failure=read_json(out/"R3A_GATE.json") if (out/"R3A_GATE.json").exists() and active in {"numerical","R3A"} and gates["status"] in {"blocked","failed"} else None
            for name in gates["stages"]:publish(out/"stages"/name,out)
            current=out/"stages"/active
            if current.is_dir() and active not in gates["stages"]:publish(current,out)
            if explicit_a_failure is not None:atomic_json(out/"R3A_GATE.json",explicit_a_failure)
            final_error=None
            try:
                if sources()["identity"]!=code["identity"]:raise Blocked("R3_SOURCE_CHANGED_DURING_RUN")
                old.verify();recovery.verify(reuse);selection_external(out/"stages/selection")
                if model_identity(c)!=old.model:raise Blocked("R3_MODEL_CHANGED_DURING_RUN")
            except Exception as error:
                final_error=error;gates.update(status="failed",reason="FINAL_IDENTITY_CHECK_FAILED: "+str(error))
            finalize(out,c,gates)
            journal.unlink()
            print("Reports: "+str(out)+"; status="+gates["status"],flush=True)
            if final_error:raise final_error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",required=True,choices=("preflight","smoke","full"))
    p.add_argument("--resume",default="0",choices=("0","1"))
    args=p.parse_args()
    signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(InterruptedError("terminated")))
    try:result=execute(args)
    except Exception as error:
        print("STOP before contract/final publication: "+type(error).__name__+": "+str(error),file=sys.stderr);result=1
    raise SystemExit(result)


if __name__=="__main__":main()

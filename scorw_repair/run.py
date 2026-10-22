"""Offline-reference R0 -> numerical/Hahner R1 -> uncalibrated exploratory R2."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import io
import gzip
import os
from pathlib import Path
import platform
import signal
import sys
import tarfile
import traceback
import unittest

from scorw_repair.core import (HERE, REPO, Blocked, load_config, sources, server_only, atomic_json,
    atomic_text, read_json, digest, identity, lock, begin, transaction, verify_inventory)


def numerical_checks(output):
    from scorw_repair.tests import test_repair,test_budgetfix
    stream=io.StringIO()
    with redirect_stdout(stream),redirect_stderr(stream):
        suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_repair,test_budgetfix)])
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(status="passed" if result.wasSuccessful() else "failed",tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),log=stream.getvalue(),
        real_data=False,detector_executed=False,physical_realism_proven=False)
    atomic_json(output/"NUMERICAL_TESTS.json",report)
    print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():
        raise Blocked("R1_NUMERICAL_GATE_FAILED: no tolerance/seed/parameter automatically changed")
    return report


def dependencies():
    names={n:importlib.util.find_spec(n) is not None for n in ("numpy","scipy","yaml","open3d","torch","spconv","cumm")}
    versions={}
    for name in ("numpy","scipy","PyYAML","torch","open3d","spconv","cumm"):
        try: versions[name]=version(name)
        except PackageNotFoundError: versions[name]=None
    return dict(modules=names,required_r0_r1=all(names[n] for n in ("numpy","scipy","yaml","open3d")),
        versions=versions,
        installation_policy="no automatic install/upgrade; existing Torch/ROCm preserved")


def external_check(output,config):
    input_manifest=output/"stages/coordinates/INPUT_MANIFEST.json"
    if input_manifest.exists():
        for row in read_json(input_manifest):
            if (not Path(row["pcd"]).is_file() or not Path(row["yaml"]).is_file()
                    or digest(row["pcd"])!=row["pcd_sha256"] or digest(row["yaml"])!=row["yaml_sha256"]):
                raise Blocked("ORIGINAL_OPV2V_CHANGED: "+row["scene"]+"/"+row["timestamp"]+"/"+row["cav"])


def preflight(config,dep,model):
    from scorw_repair.regression import table_resource
    problems=[]
    try:
        _,table=table_resource(production=True)
    except Blocked as error:
        table=dict(status="missing_or_changed",reason=str(error));problems.append(str(error))
    if not dep["required_r0_r1"]:problems.append("R0_R1_DEPENDENCIES_MISSING")
    if not all(dep["modules"].get(n,False) for n in ("torch","spconv","cumm")):
        problems.append("R2_DEPENDENCIES_MISSING")
    if model.get("status")!="verified":problems.append("FROZEN_ATTFUSE_IDENTITY_MISSING")
    if not Path(config["train_root"]).is_dir():problems.append("ORIGINAL_TRAIN_ROOT_MISSING")
    return dict(status="ready" if not problems else "blocked",problems=problems,dependencies=dep,model=model,
        hahner_table=table,network_requests=0,downloads_enabled=False,extraction_enabled=False,
        real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE",
        r2_policy="automatic UNCALIBRATED_EXPLORATORY after strict R0/R1 gates; SMOKE includes a minimal detector diagnostic")


def placeholder(output,stage,status,reason):
    name={"R0":"R0_SENSOR_AUDIT.json","R1":"R1_PHYSICS_AUDIT.json","R2":"R2_TARGET_DIAGNOSIS.json"}[stage]
    if not (output/name).exists():
        atomic_json(output/name,dict(status=status,reason=reason,physical_calibration_passed=False,
            real_data_compared=False,detector_executed=False))
        atomic_text(output/(stage+"_RESULTS.md"),f"# {stage}\n\n{status}\n\n{reason}\n\n尚无该阶段有效真实实验结果。\n")


def publish(stage_root,output,names):
    from scorw_repair.core import atomic_bytes
    for name in names:
        source=stage_root/name
        if source.is_file():
            atomic_bytes(output/name,source.read_bytes())


def archive(output):
    # No arrays, raw samples, credentials or private files enter the result package.
    files=[p for p in sorted(output.iterdir()) if p.is_file() and p.suffix in {".json",".md"}
           and p.name not in {"REPORT_MANIFEST.json","PUBLICATION_IN_PROGRESS.json"}]
    rows={p.name:dict(sha256=digest(p),bytes=p.stat().st_size) for p in files}
    if sum(r["bytes"] for r in rows.values())>64*1024*1024:
        raise Blocked("SMALL_RESULT_ARCHIVE_BUDGET")
    atomic_json(output/"REPORT_MANIFEST.json",rows)
    from scorw_repair.core import atomic_bytes
    payload=io.BytesIO()
    with gzip.GzipFile(fileobj=payload,mode="wb",mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode="w") as package:
            for p in [*files,output/"REPORT_MANIFEST.json"]:
                data=p.read_bytes(); info=tarfile.TarInfo(p.name)
                info.size=len(data);info.mtime=0;info.mode=0o644
                package.addfile(info,io.BytesIO(data))
    atomic_bytes(output/"SMALL_RESULTS.tar.gz",payload.getvalue())


def run(args):
    server_only()
    config=load_config(); code=sources(); dep=dependencies()
    from scorw_repair import source_audit, regression, generate, detection, reuse
    if args.download!="0":
        raise Blocked("DOWNLOAD_DISABLED: this revision never reads URLs or downloads real data; set DOWNLOAD=0")
    smoke=args.mode=="smoke"
    label=args.mode
    output=Path(config["output_root"])/label
    opv2v=generate.opv2v_freeze(config,smoke)
    model=detection.model_identity(config)
    request=dict(config=config,source_identity=code["identity"],snapshot=code["snapshot_sha256"],mode=args.mode,
        exploratory_authorized=True,real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE",
        real_reference_enabled=False,opv2v_selection=opv2v,model=model,
        runtime=dict(python=platform.python_version(),dependencies=dep,
            threads={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")}),
        frozen_before_weather=True,URL_tokens_recorded=False)
    with reuse.readonly_previous(config,args.mode) as previous,lock(output):
        reuse_info=reuse.audit(previous,config,opv2v,model,request["runtime"])
        request["verified_previous_run"]=reuse_info
        contract=begin(output,request,args.resume=="1")
        journal=output/"PUBLICATION_IN_PROGRESS.json"
        if journal.exists() and read_json(journal).get("contract")!=contract:
            raise Blocked("PUBLICATION_JOURNAL_CONTRACT_CHANGED")
        if not journal.exists() and (output/"REPORT_MANIFEST.json").exists():
            for name,row in read_json(output/"REPORT_MANIFEST.json").items():
                if not (output/name).is_file() or digest(output/name)!=row["sha256"]:
                    raise Blocked("PUBLISHED_REPORT_CHANGED: "+name)
        # Root reports are derived publications. During an interrupted publication, stage commits remain authoritative.
        atomic_json(journal,dict(contract=contract,role="mutable report publication; immutable stage commits still strictly verified"))
        gates=dict(status="running",contract=contract,mode=args.mode,stages={},r2_authorized=True,
            real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE",real_data_required=False)
        atomic_json(output/"R_PROTOCOL_FREEZE.json",request)
        atomic_json(output/"SOURCE_AUDIT.json",code)
        atomic_json(output/"DEPENDENCIES.json",dep)
        atomic_json(output/"CACHE_REUSE_AUDIT.json",dict(status="verified" if reuse_info else "not_available_fresh_run",
            previous=reuse_info,old_contract_modified=False,old_detector_predictions_reused=False))
        def phase(name,callback):
            folder=output/"stages"/name
            print("START "+name,flush=True)
            if sources()["identity"]!=code["identity"]:
                raise Blocked("SOURCE_CHANGED_BEFORE_STAGE: "+name)
            reuse.verify(reuse_info)
            def checked(stage_code):
                if not reuse.import_stage(reuse_info,name,folder):callback(folder,stage_code)
                required={"numerical":("NUMERICAL_TESTS.json","passed"),
                    "coordinates":("REAL_COORDINATE_AUDIT.json","passed"),
                    "R0":("R0_SENSOR_AUDIT.json","SOURCE_AND_CLEAN_SENSOR_AUDIT_PASS"),
                    "hahner_regression":("R1_HAHNER_REGRESSION.json","passed"),
                    "R1":("R1_PHYSICS_AUDIT.json","numerical_engine_executed"),
                    "R2":("R2_TARGET_DIAGNOSIS.json","detector_diagnostic_executed")}
                report,expected=required[name]
                if not (folder/report).is_file() or read_json(folder/report).get("status")!=expected:
                    raise Blocked("FAILED_STAGE_REPORT: "+name)
                if name=="R2" and read_json(folder/report).get("label")!="UNCALIBRATED_EXPLORATORY":
                    raise Blocked("R2_MUST_REMAIN_UNCALIBRATED_EXPLORATORY")
                if sources()["identity"]!=code["identity"]:
                    raise Blocked("SOURCE_CHANGED_INSIDE_STAGE: "+name)
            record=transaction(folder,dict(parent=contract,stage=name),checked)
            gates["stages"][name]=dict(status="passed",root=str(folder),commit_sha256=digest(folder/"COMMIT.json"),hashes=record["hashes"])
            atomic_json(output/"VALIDATION_GATES.json",gates)
            print("PASS "+name,flush=True)
            return folder
        active="preflight"
        try:
            # Existing committed external inputs are verified even when their stages are reused.
            external_check(output,config)
            if args.mode=="preflight":
                report=preflight(config,dep,model)
                atomic_json(output/"PREFLIGHT.json",report)
                gates.update(status=report["status"],large_data_downloaded=False)
                print("Preflight: "+report["status"]+"; network requests=0",flush=True)
                for problem in report["problems"]:
                    print("Problem: "+problem,flush=True)
                return 0 if report["status"]=="ready" and dep["required_r0_r1"] else 3
            active="numerical"
            numeric=phase(active,lambda p,c:numerical_checks(p))
            publish(numeric,output,["NUMERICAL_TESTS.json"])
            if not dep["required_r0_r1"]:
                raise Blocked("BLOCKED_DEPENDENCIES: see DEPENDENCIES.json; no environment upgrades performed")
            active="coordinates"
            coords=phase(active,lambda p,c:generate.coordinates(config,opv2v,p,c))
            publish(coords,output,["COORDINATE_AUDIT.md"])
            active="R0"
            r0=phase(active,lambda p,c:source_audit.audit(config,coords,p))
            publish(r0,output,["R0_RESULTS.md","R0_SENSOR_AUDIT.json"])
            active="hahner_regression"
            reg=phase(active,lambda p,c:regression.check(config,p))
            publish(reg,output,["R1_HAHNER_REGRESSION.json","R1_HAHNER_REGRESSION.md"])
            active="R1"
            r1=phase(active,lambda p,c:generate.execute_r1(config,opv2v,coords,p,c))
            publish(r1,output,["R1_RESULTS.md","R1_PHYSICS_AUDIT.json","R1_FIELD_POOL_FREEZE.json"])
            source,physical=read_json(output/"R0_SENSOR_AUDIT.json"),read_json(output/"R1_PHYSICS_AUDIT.json")
            active="R2"
            external_check(output,config)
            if detection.model_identity(config)!=model:raise Blocked("MODEL_CHANGED_BEFORE_R2")
            decision=detection.r2_gate(source,physical,model,read_json(numeric/"NUMERICAL_TESTS.json"),
                read_json(reg/"R1_HAHNER_REGRESSION.json"))
            atomic_json(output/"R2_GATE.json",decision)
            if decision["status"]!="eligible":
                gates["stages"]["R2"]=decision
                placeholder(output,"R2",decision["status"],decision["reason"])
                gates["status"]=decision["status"]
                return 3
            os.environ["ROCR_VISIBLE_DEVICES"]=str(config["r2"]["gpu"])
            os.environ.pop("HIP_VISIBLE_DEVICES",None);os.environ.pop("CUDA_VISIBLE_DEVICES",None)
            r2=phase(active,lambda p,c:detection.evaluate(config,coords,p,model,decision["label"],c,reuse_info))
            publish(r2,output,["R2_RESULTS.md","R2_TARGET_DIAGNOSIS.json","R2_COMPUTE_PLAN.json","R2_IOU_AUDIT.json"])
            gates["status"]=("SMOKE_" if smoke else "FULL_")+"UNCALIBRATED_EXPLORATORY_COMPLETE"
            return 0
        except Blocked as error:
            gates.update(status="blocked",blocked_stage=active,reason=str(error))
            print("STOP "+active+": "+str(error),flush=True)
            return 3
        except Exception as error:
            # Unknown external exceptions are logged by type only to prevent token leakage.
            gates.update(status="failed",failed_stage=active,error_type=type(error).__name__,
                locations=[dict(file=Path(f.filename).name,line=f.lineno,function=f.name) for f in traceback.extract_tb(error.__traceback__)],
                reason="Unexpected error; no downstream stage executed; private error strings suppressed")
            print("STOP "+active+": "+type(error).__name__,flush=True)
            return 1
        finally:
            publish(output/"stages/R2",output,["R2_COMPUTE_PLAN.json","R2_IOU_AUDIT.json"])
            for stage in ("R0","R1","R2"):
                placeholder(output,stage,"NOT_EXECUTED",gates.get("reason","prerequisite or physical gate has not passed"))
            # A placeholder from an earlier blocked attempt may be replaced by a committed real report on resume.
            final_error=None
            try:
                source_changed=sources()["identity"]!=code["identity"]
            except Blocked:
                source_changed=True
            if source_changed:
                gates.update(status="failed",reason="SOURCE_CHANGED_DURING_RUN")
                final_error="SOURCE_CHANGED_DURING_RUN"
            try:
                external_check(output,config)
                reuse.verify(reuse_info)
                if detection.model_identity(config)!=model:raise Blocked("MODEL_CHANGED_DURING_RUN")
            except Blocked as error:
                gates.update(status="failed",reason=str(error))
                final_error=str(error)
            atomic_json(output/"VALIDATION_GATES.json",gates)
            stage_pass=lambda name:gates["stages"].get(name,{}).get("status")=="passed"
            valid_run=not final_error
            physics_ok=(valid_run and all(stage_pass(n) for n in ("numerical","coordinates","R0","hahner_regression","R1"))
                and detection.physical_checks_passed(read_json(output/"R1_PHYSICS_AUDIT.json")))
            detector_ok=physics_ok and stage_pass("R2")
            evidence=dict(mathematical_correctness=valid_run and stage_pass("numerical"),
                simulator_executed=gates["stages"].get("R1",{}).get("status")=="passed",
                grounded_physical_approximation=True,real_statistical_comparison=False,
                physical_calibration_passed=False,detector_impact_verified=detector_ok,
                detector_label="UNCALIBRATED_EXPLORATORY",real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE",
                ENGINEERING_PASS=physics_ok and detector_ok,PHYSICS_NUMERICAL_PASS=physics_ok,
                REAL_WORLD_VALIDATION_MISSING=True,DETECTOR_EXPLORATORY_RESULT=detector_ok,
                smoke_replaces_formal_results=False,
                formal_dataset_accepted=False)
            atomic_json(output/"EVIDENCE_LEVELS.json",evidence)
            atomic_text(output/"RESULTS.md","# SCOR-W 修复结果\n\n"+gates["status"]+"\n\n"+
                gates.get("reason","")+"\n\n"+"\n".join(f"- {k}: {v}" for k,v in evidence.items())+
                "\n\n数值正确、引擎执行、物理近似、真实统计比较和检测影响分别记录。"
                "本版没有真实雾天数据比较，接收阈值、波形和 OPV2V 强度功率映射均未标定。"
                "只有数值、来源、身份和 Hahner 回归全部通过才运行 R2，所有检测均标为未标定探索。"
                "不会自动增加场池、修改参数或扩大数据集。\n")
            atomic_text(output/"NEXT_STAGE_DECISION.md","# 后续判断\n\n"+
                ("EXPLORATORY_REVIEW_REQUIRED" if detector_ok else "STOP_DOWNSTREAM_UNTIL_GATES_PASS")+
                "\n\n本轮自动流程不会批准正式 SCOR-W 数据集。真实验证缺失。"
                "只有回波替换/失效/距离统计有可解释的模型依据，B3/B4 平衡通过且检测诊断显示值得研究的结构，"
                "才考虑获取独立真实参考、校准接收模型并预注册后续验证。"
                "若仍缺乏稳定且有实际意义的结构差异，应停止扩大全量数据集；不自动增加 seed 或调参寻找结果。\n")
            archive(output)
            journal.unlink()
            logs=Path(config["logs_root"]);logs.mkdir(parents=True,exist_ok=True)
            atomic_json(logs/(label+"_LATEST.json"),dict(output=str(output),status=gates["status"],
                archive_sha256=digest(output/"SMALL_RESULTS.tar.gz")))
            print("Reports: "+str(output)+"; status="+gates["status"],flush=True)
            if final_error:
                raise Blocked(final_error)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("preflight","smoke","full"),required=True)
    parser.add_argument("--download",choices=("0","1"),default="0")
    parser.add_argument("--resume",choices=("0","1"),default="0")
    args=parser.parse_args()
    signal.signal(signal.SIGTERM,lambda *a:(_ for _ in ()).throw(InterruptedError("terminated")))
    try:
        result=run(args)
    except Exception as error:
        # All URL-bearing paths are behind sanitized boundaries; still avoid unexpected text leakage.
        print("STOP before stage contract: "+type(error).__name__+"; "+(str(error) if isinstance(error,Blocked) else "details suppressed"),file=sys.stderr)
        result=1
    raise SystemExit(result)


if __name__=="__main__":main()

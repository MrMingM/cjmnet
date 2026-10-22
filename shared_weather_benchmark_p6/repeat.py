"""P6-C CPU materialization through immutable v1 kernels; all repeats retained."""
from __future__ import annotations

import itertools
from pathlib import Path
import random
import tempfile
import platform

from shared_weather_benchmark_p6.core import (
    read_json, digest, identity, seed_for, ensure_json, verify_external, verify_hashes,
    safe_child, assert_close, atomic_json, atomic_text)
from shared_weather_benchmark_p6.selection import server_freeze, scope_selection, frozen_selection_identity
from shared_weather_benchmark_p6.metrics import pearson, write_analysis
from shared_weather_benchmark_p6.legacy import scan_clouds
from shared_weather_benchmark_p6.comparison import scene_seed_results,summarize_repeats,write_results


def field_checks(cfg,selection,domains):
    import numpy as np
    from shared_weather_benchmark.generator import make_field
    from shared_weather_benchmark.compare import field_diagnostics
    result = field_diagnostics(cfg,selection,domains)
    for scene in result:
        frame = next(f for f in selection["frames"] if f["scene"] == scene)
        low,high = domains[scene][:3],domains[scene][3:]
        rng = random.Random(seed_for(20261021,"p6-fixed-cross-cav-probes",scene))
        probes = np.asarray([[rng.uniform(a,b) for a,b in zip(low,high)] for _ in range(256)])
        cross = {}
        for mode in ("E2","E3"):
            arrays = {c:make_field(cfg,scene,c,mode,domains[scene]).alpha_numpy(probes) for c in frame["cavs"]}
            pairs = {}
            for a,b in itertools.combinations(frame["cavs"],2):
                delta = float(np.max(np.abs(arrays[a]-arrays[b])))
                if mode == "E2" and delta != 0:
                    raise ValueError("E2 shared-world field disagreement")
                pairs[a+"/"+b] = dict(alpha_correlation=pearson(arrays[a].tolist(),arrays[b].tolist()),max_alpha_delta=delta)
            cross[mode] = pairs
        if len(set(result[scene]["E3"]["field_hashes"])) != len(frame["cavs"]):
            raise ValueError("Independent E3 field identities collided")
        result[scene]["cross_cav"] = dict(design_verified=True,world_probe_pairs=cross,
            interpretation="field correlations at common positions; not target signal correlations; independent seeds do not imply finite-sample Pearson exactly zero")
    return result


def verify_generated(root,config,contract):
    from shared_weather_benchmark.generator import validate_record
    rows = read_json(root/"INPUT_MANIFEST.json")
    index = {(r["scene"],r["timestamp"],r["cav"]):r for r in rows}
    manifest = read_json(root/"DATA_MANIFEST.json")
    if manifest["contract"] != contract or manifest["protocols"] != config["protocols"]:
        raise ValueError("P6 generated dataset identity differs")
    if manifest["input_manifest_sha256"] != digest(root/"INPUT_MANIFEST.json") or manifest["clean_gt_sha256"] != digest(root/"CLEAN_GT.json"):
        raise ValueError("P6 generated input/GT changed")
    if len(set(manifest["records"])) != len(manifest["records"]) or set(manifest["records"]) != set(manifest["record_hashes"]):
        raise ValueError("P6 duplicate or unregistered cloud record")
    verify_hashes(root,manifest["record_hashes"])
    found = set()
    records = []
    for name in manifest["records"]:
        record = read_json(safe_child(root,name))
        key = (record["scene"],record["timestamp"],record["cav"])
        full = (record["mode"],*key)
        if key not in index or full in found:
            raise ValueError("P6 cloud identity/duplicate record mismatch")
        expected = Path(record["mode"])/"train"/key[0]/key[2]/(key[1]+".pcd")
        if record["pcd"] != expected.as_posix() or record["sidecar"] != (Path("sidecars")/expected.with_suffix(".npz")).as_posix():
            raise ValueError("P6 cloud path identity mismatch")
        verify_record_metadata(root,record,index[key],config,contract)
        validate_record(root,record,index[key],contract)
        found.add(full)
        records.append(record)
    if found != {(m,*k) for m in config["protocols"] for k in index}:
        raise ValueError("Incomplete scene/frame/CAV/protocol Cartesian product")
    if {p.relative_to(root).as_posix() for p in (root/"records").rglob("*.json")} != set(manifest["records"]):
        raise ValueError("Unregistered P6 cloud record files")
    for mode in config["protocols"]:
        if {p.relative_to(root).as_posix() for p in (root/mode).rglob("*.pcd")} != {r["pcd"] for r in records if r["mode"] == mode}:
            raise ValueError("Unregistered or missing P6 weather PCD")
    return records


def verify_record_metadata(root,record,row,config,contract):
    mode = record["mode"]
    if mode not in config["protocols"] or record["partition"] != row["partition"] or record["contract"] != contract:
        raise ValueError("P6 record protocol/partition/contract mismatch")
    relative = Path(mode)/"train"/row["scene"]/row["cav"]/(row["timestamp"]+".pcd")
    sidecar = (Path("sidecars")/relative.with_suffix(".npz")).as_posix()
    if (record["pcd"] != relative.as_posix() or record["sidecar"] != sidecar
            or set(record["hashes"]) != {relative.as_posix(),relative.with_suffix(".yaml").as_posix(),sidecar}):
        raise ValueError("P6 record artifact inventory differs")
    safe_child(root,record["pcd"]);safe_child(root,sidecar)
    annotation = root/relative.with_suffix(".yaml")
    if annotation.resolve(strict=True) != Path(row["yaml"]).resolve(strict=True):
        raise ValueError("P6 YAML no longer points to original input")
    fields = read_json(root/"SCENE_WEATHER_FIELD_MANIFEST.json")["fields"]
    key = "/".join([mode,row["scene"],row["cav"] if mode == "E3" else "shared"])
    if (record["field_sha256"] != fields[key]["sha256"]
            or record["seed"] != seed_for(config["seed"],mode,row["scene"],row["timestamp"],row["cav"]) % (2**32)):
        raise ValueError("P6 field/seed metadata differs from frozen design")


def verify_partial_records(root,config,contract):
    from shared_weather_benchmark.generator import validate_record
    inputs = {(r["scene"],r["timestamp"],r["cav"]):r for r in read_json(root/"INPUT_MANIFEST.json")}
    for file in (root/"records").rglob("*.json"):
        record = read_json(file)
        key = (record["scene"],record["timestamp"],record["cav"])
        if key not in inputs:
            raise ValueError("Unexpected partial cloud source identity")
        verify_record_metadata(root,record,inputs[key],config,contract)
        expected = root/"records"/Path(record["pcd"]).with_suffix(".json")
        if file != expected:
            raise ValueError("Partial record stored under wrong identity")
        validate_record(root,record,inputs[key],contract)


def prepare_inputs(cfg,selection,output,resume):
    """Always recheck source geometry into a temp tree before reusing commits."""
    from shared_weather_benchmark.audit import real_coordinate_audit
    with tempfile.TemporaryDirectory(prefix=".coordinate-",dir=output) as temp:
        work = Path(temp)
        real_coordinate_audit(cfg,selection,Path(cfg["train_root"]),work)
        for name in ("INPUT_MANIFEST.json","CLEAN_GT.json","REAL_COORDINATE_AUDIT.json","FIELD_DOMAINS.json"):
            ensure_json(output/name,read_json(work/name))
        if not (output/"COORDINATE_AUDIT.md").exists():
            atomic_text(output/"COORDINATE_AUDIT.md",(work/"COORDINATE_AUDIT.md").read_text("utf-8"))


def run_repeats(config,legacy,output,contract,smoke,resume):
    import numpy as np
    from importlib.metadata import version
    from shared_weather_benchmark.generator import generate,materialize_fields
    from shared_weather_benchmark.compare import aggregate_clouds
    output.mkdir(parents=True,exist_ok=True)
    old_selection_sha = digest(Path(config["legacy_root"])/"SELECTION.json")
    try:
        freeze = server_freeze(legacy["selection"],config,old_selection_sha)
    except Exception as error:
        atomic_json(output/"P6C_SELECTION_FAILURE.json",dict(status="stopped",error=str(error),future_reservation_relaxed=False))
        raise
    # Both SMOKE and full materialize the complete 12-scene plan before any new
    # point is simulated. SMOKE merely chooses a declared subset of that plan.
    ensure_json(output/"P6C_SELECTION_FREEZE.json",freeze)
    selection = scope_selection(freeze,config,smoke)
    ensure_json(output/"P6C_FROZEN_CONFIG.json",dict(config=config,weather_seeds=freeze["weather_seeds"],
        formal_plan_identity=frozen_selection_identity(freeze),selection=selection,smoke=smoke))
    cfg = dict(legacy["config"],protocols=config["protocols"],field=config["field"],integration=config["integration"],
               train_root=config["train_root"],output_root=str(output),frames_per_scene=2)
    cfg["integration_for_generation"] = {k:config["integration"][k] for k in ("step_m","batch")}
    cfg["return_policy"] = dict(name="attenuation_only",threshold=0,status="no_sensor_threshold_assumed")
    prepare_inputs(cfg,selection,output,resume)
    ensure_json(output/"SELECTION.json",selection)
    runtime = dict(python=platform.python_version(),numpy=np.__version__,
                   dependencies={name:version(name) for name in ("open3d","PyYAML")})
    repeats = config["smoke_repeat_indices"] if smoke else config["repeat_indices"]
    all_results = []
    for repeat in repeats:
        repeat_root = output/"repeats"/f"repeat_{repeat}"
        if output.resolve() not in repeat_root.resolve().parents or repeat_root.is_symlink():
            raise ValueError("Repeat output aliases a different/frozen tree")
        repeat_root.mkdir(parents=True,exist_ok=True)
        repeat_cfg = dict(cfg,seed=freeze["weather_seeds"][str(repeat)],output_root=str(repeat_root))
        for name in ("INPUT_MANIFEST.json","CLEAN_GT.json","FIELD_DOMAINS.json"):
            ensure_json(repeat_root/name,read_json(output/name))
        ensure_json(repeat_root/"SELECTION.json",selection)
        domains = read_json(output/"FIELD_DOMAINS.json")
        fields = materialize_fields(repeat_cfg,selection,domains)
        diagnostics = field_checks(repeat_cfg,selection,domains)
        ensure_json(repeat_root/"SCENE_WEATHER_FIELD_MANIFEST.json",fields)
        ensure_json(repeat_root/"P6C_FIELD_CHECKS.json",diagnostics)
        request = dict(parent_contract=contract,repeat_index=repeat,config=repeat_cfg,runtime=runtime,
            input_sha256=digest(output/"INPUT_MANIFEST.json"),gt_sha256=digest(output/"CLEAN_GT.json"),
            selection_identity=identity(selection),field_identity=fields["identity"])
        sub_contract = identity(request)
        ensure_json(repeat_root/"CONTRACT.json",dict(identity=sub_contract,request=request))
        ensure_json(repeat_root/"RESOLVED_CONFIG.json",repeat_cfg)
        verify_partial_records(repeat_root,repeat_cfg,sub_contract)
        if (repeat_root/"DATA_MANIFEST.json").exists():
            verify_generated(repeat_root,repeat_cfg,sub_contract)
        generate(repeat_cfg,repeat_root,sub_contract,resume)
        records = verify_generated(repeat_root,repeat_cfg,sub_contract)
        metric_root = repeat_root/"metrics"
        metric_root.mkdir(exist_ok=True)
        rows = scan_clouds(repeat_root,repeat_cfg,repeat_root/"metric_cache",repeat,replay_integrals=False)
        analysis = write_analysis(metric_root,rows,config,config["protocols"])
        physical = {}
        for scene in diagnostics:
            physical[scene] = {mode:aggregate_clouds([r for r in records if r["scene"] == scene and r["mode"] == mode],
                len(cfg["distance_bins_m"])-1) for mode in config["protocols"]}
        result = scene_seed_results(selection,repeat,analysis,physical,diagnostics,config)
        ensure_json(repeat_root/"P6C_SCENE_SEED_RESULT.json",result)
        all_results.extend(result)
        print(f"P6-C repeat={repeat} completed; no redraw",flush=True)
    result = summarize_repeats(all_results,config,{f["scene"] for f in selection["frames"]},repeats)
    verdict = write_results(output,result,config,smoke)
    return result,verdict

"""Read-only P0-P5 evidence checks, fresh reductions, and small audit exports."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
import tempfile

from shared_weather_benchmark_p6.core import (
    read_json, read_jsonl, digest, identity, verify_hashes, safe_child, verify_external,
    assert_close, atomic_json, atomic_text, write_csv, write_jsonl, mean, stats)
from shared_weather_benchmark_p6.metrics import cloud_rows, check_denominators, row_key


def verify_contract(root, expected):
    contract = read_json(root/"CONTRACT.json")
    if contract["identity"] != expected or identity(contract["request"]) != expected:
        raise ValueError("Legacy contract identity mismatch")
    gates = read_json(root/"VALIDATION_GATES.json")
    if gates["contract"] != expected or gates["status"] != "complete":
        raise ValueError("Legacy run not complete or gate contract differs")
    if set(gates["stages"]) != {"P0","P1","P2","P3","P4","P5"}:
        raise ValueError("Legacy stage inventory mismatch")
    for stage, data in gates["stages"].items():
        if data["status"] != "passed" or not data["hashes"]:
            raise ValueError("Legacy stage not passed: " + stage)
        verify_hashes(root,data["hashes"])
    return contract,gates


def verify_legacy(root, expected, train, historical):
    """All registered artifacts and raw inputs are rehashed on each resume."""
    from shared_weather_benchmark.dataset_io import approved_path, source_paths, cav_order, timestamps
    root,train = Path(root).resolve(strict=True),approved_path(train)
    contract,gates = verify_contract(root,expected)
    cfg = read_json(root/"RESOLVED_CONFIG.json")
    if contract["request"]["config"] != cfg or cfg["return_policy"]["name"] != "attenuation_only":
        raise ValueError("Unexpected legacy configuration/return policy")
    if Path(cfg["train_root"]).resolve(strict=True) != train or not contract["request"]["smoke"]:
        raise ValueError("Legacy source train or SMOKE identity differs")
    if digest(root/"CONFIG_INPUT.json") != contract["request"]["config_file_sha256"]:
        raise ValueError("Legacy input config changed")
    source_audit = read_json(root/"SOURCE_AUDIT.json")
    if source_audit["status"] != "passed" or source_audit["source_identity"] != contract["request"]["source_identity"]:
        raise ValueError("Legacy source audit identity mismatch")
    selection = read_json(root/"SELECTION.json")
    if {p.name for p in train.iterdir() if p.is_dir()} != set(selection["profiles"]):
        raise ValueError("Original scene inventory changed since legacy selection")
    for scene,expected_inventory in selection["profile_timestamp_inventory_sha256"].items():
        cavs = cav_order(train/scene,train)
        if identity(timestamps(train/scene/cavs[0],train)) != expected_inventory:
            raise ValueError("Original timestamp inventory changed: " + scene)
    frames = selection["frames"]
    if set(f["scene"] for f in frames) != set(historical):
        raise ValueError("Legacy four-scene identity differs")
    keys = [(f["scene"],f["timestamp"],c) for f in frames for c in f["cavs"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate legacy frame/CAV identity")
    inputs = read_json(root/"INPUT_MANIFEST.json")
    index = {(r["scene"],r["timestamp"],r["cav"]):r for r in inputs}
    if len(index) != len(inputs) or set(index) != set(keys):
        raise ValueError("Legacy input manifest/frame inventory differs")
    manifest = read_json(root/"DATA_MANIFEST.json")
    if manifest["contract"] != expected or manifest["status"] != "complete_generation":
        raise ValueError("Legacy cloud manifest incomplete")
    modes = ["E0","E1","E2","E3"]
    if cfg["protocols"] != modes or manifest["protocols"] != modes:
        raise ValueError("Legacy protocol inventory differs")
    verify_hashes(root,manifest["record_hashes"])
    if (set(manifest["records"]) != set(manifest["record_hashes"])
            or len(set(manifest["records"])) != len(manifest["records"])):
        raise ValueError("Duplicate/missing legacy record entries")
    if (manifest["clean_gt_sha256"] != digest(root/"CLEAN_GT.json")
            or manifest["input_manifest_sha256"] != digest(root/"INPUT_MANIFEST.json")):
        raise ValueError("Legacy frozen GT/input identity mismatch")
    artifacts = {}
    def register(path, sha=None, role="legacy_artifact"):
        path = Path(path)
        observed = digest(path)
        if sha is not None and observed != sha:
            raise ValueError("Read-only source/artifact hash differs: " + str(path))
        key = str(path.absolute())
        if key in artifacts and artifacts[key]["sha256"] != observed:
            raise ValueError("File changed during audit: " + key)
        artifacts[key] = dict(path=key,sha256=observed,bytes=path.stat().st_size,role=role)
    for name in ("CONTRACT.json","VALIDATION_GATES.json"):
        register(root/name)
    for data in gates["stages"].values():
        for name, sha in data["hashes"].items():
            register(safe_child(root,name),sha)
    freeze = read_json(root/"GENERATION_FREEZE.json")
    register(Path(freeze["e0_table"]["path"]),freeze["e0_table"]["sha256"],"frozen_E0_lookup_asset")
    for frame in frames:
        for cav in frame["cavs"]:
            row = index[(frame["scene"],frame["timestamp"],cav)]
            pcd,yaml = source_paths(train,frame,cav)
            if Path(row["pcd"]).resolve() != pcd or Path(row["yaml"]).resolve() != yaml:
                raise ValueError("Original PCD/YAML path identity mismatch")
            register(pcd,row["pcd_sha256"],"original_clean_pcd")
            register(yaml,row["yaml_sha256"],"original_clean_yaml")
    for row in selection["selection_input_hashes"]:
        register(approved_path(row["yaml"],train),row["sha256"],"selection_layout_yaml")
    records = []
    observed_keys = set()
    for name in manifest["records"]:
        path = safe_child(root,name)
        record = read_json(path)
        key = (record["scene"],record["timestamp"],record["cav"])
        full = (record["mode"],*key)
        if key not in index or full in observed_keys or record["mode"] not in modes:
            raise ValueError("Unpaired/duplicate cloud identity")
        observed_keys.add(full)
        row = index[key]
        relative = Path(record["mode"])/"train"/record["scene"]/record["cav"]/(record["timestamp"]+".pcd")
        expected_sidecar = (Path("sidecars")/relative.with_suffix(".npz")).as_posix()
        if (record["contract"] != expected or record["pcd"] != relative.as_posix()
                or record["sidecar"] != expected_sidecar
                or record["input_pcd_sha256"] != row["pcd_sha256"]
                or record["input_yaml_sha256"] != row["yaml_sha256"]
                or record["partition"] != row["partition"]):
            raise ValueError("Cloud provenance/partition mismatch")
        needed = {relative.as_posix(),relative.with_suffix(".yaml").as_posix(),expected_sidecar}
        if set(record["hashes"]) != needed:
            raise ValueError("Cloud artifact inventory mismatch")
        for child,sha in record["hashes"].items():
            if child.endswith(".yaml"):
                link = root/child
                if not link.is_symlink() or link.resolve(strict=True) != Path(row["yaml"]).resolve(strict=True):
                    raise ValueError("Weather YAML no longer links exact original")
                register(link,sha,"original_yaml_link")
            else:
                register(safe_child(root,child),sha)
        register(path,manifest["record_hashes"][name])
        records.append(record)
    if observed_keys != {(m,*k) for m in modes for k in keys}:
        raise ValueError("A protocol or CAV was skipped")
    actual_records = {p.relative_to(root).as_posix() for p in (root/"records").rglob("*.json")}
    if actual_records != set(manifest["records"]):
        raise ValueError("Unregistered or missing cloud records")
    for mode in modes:
        actual = {p.relative_to(root).as_posix() for p in (root/mode).rglob("*.pcd")}
        if actual != {r["pcd"] for r in records if r["mode"] == mode}:
            raise ValueError("Unregistered/missing weather PCDs: " + mode)
    return dict(contract=contract,config=cfg,selection=selection,inputs=inputs,records=records,
                artifacts=sorted(artifacts.values(),key=lambda r:r["path"]))


def legacy_targets(rows, settings):
    from shared_weather_benchmark.observability import classify
    groups = defaultdict(list)
    for r in rows:
        if r["active"]:
            groups[(r["mode"],r["scene"],r["timestamp"],r["object_id"])].append(r)
    result = []
    for key,values in sorted(groups.items()):
        ego = next(r for r in values if r["is_ego"])
        peers = [r for r in values if not r["is_ego"]]
        available = [r["R"] for r in peers if r["R"] is not None]
        best = max(available) if available else None
        result.append(dict(mode=key[0],scene=key[1],timestamp=key[2],object_id=key[3],partition=ego["partition"],
            category=classify(values,settings["min_returns"],settings["good_ratio"],settings["disparity"]),
            sensitivity={f"count={n},ratio={q}":classify(values,n,q,settings["disparity"])
                         for n in settings["sensitivity_min_returns"] for q in settings["sensitivity_good_ratio"]},
            all_zero_hard_returns=all(r["weather_hard_count"] == 0 for r in values),
            ego_zero_neighbor_positive=ego["weather_hard_count"] == 0 and any(r["weather_hard_count"] > 0 for r in peers),
            all_zero_clean_returns=all(r["clean_hard_count"] == 0 for r in values),
            ego_best_neighbor_intensity_gap=best-ego["R"] if best is not None and ego["R"] is not None else None,
            ego_effective_ratio=ego["R"],best_neighbor_effective_ratio=best,
            per_cav_retention={r["cav"]:r["retention_ratio"] for r in values}))
    return result


def scan_clouds(root, config, cache, repeat, replay_integrals=True):
    """Server-only PCD/sidecar validation and deterministic per-cloud metric commits."""
    import numpy as np
    from shared_weather_benchmark.dataset_io import server_only
    from shared_weather_benchmark.observability import inspect_cloud
    from shared_weather_benchmark.generator import cloud_stats, load_field
    from shared_weather_benchmark.ray_integrator import integrate_numpy
    from shared_weather_benchmark.audit import world_xyz
    server_only()
    manifest = read_json(root/"DATA_MANIFEST.json")
    inputs = {(r["scene"],r["timestamp"],r["cav"]):r for r in read_json(root/"INPUT_MANIFEST.json")}
    frames = {(f["scene"],f["timestamp"]):f for f in read_json(root/"CLEAN_GT.json")}
    domains = read_json(root/"FIELD_DOMAINS.json")
    all_rows,chunk_hashes = [],{}
    for name in manifest["records"]:
        record = read_json(root/name)
        row = inputs[(record["scene"],record["timestamp"],record["cav"])]
        frame = frames[(row["scene"],row["timestamp"])]
        path = cache/Path(name)
        bound = dict(record_sha256=digest(root/name),config_sha256=identity(config),repeat_index=repeat)
        if path.exists():
            chunk = read_json(path)
            if chunk["bound"] != bound or identity(chunk["rows"]) != chunk["rows_identity"]:
                raise ValueError("Metric cache identity mismatch")
        else:
            data = inspect_cloud(root,record,row)
            if any(data[k].shape != (len(data["original_xyzi"]),) for k in ("emitted","retained","origin","integral","original_index","output_index")):
                raise ValueError("Sidecar array rank mismatch")
            if data["emitted"].dtype != np.dtype(bool) or data["retained"].dtype != np.dtype(bool):
                raise ValueError("Sidecar mask dtype mismatch")
            raw,weather = data["original_xyzi"],data["weather_xyzi"]
            if record["mode"] in {"E1","E2","E3"}:
                if not data["retained"].all() or not data["emitted"].all() or data["origin"].any():
                    raise ValueError("attenuation_only changed original point retention")
                if replay_integrals:
                    field = load_field(config,row,record["mode"],domains)
                    tau = integrate_numpy(field,row["pose"][:3],world_xyz(raw,row["pose"]),**config["integration_for_generation"])
                    if not np.allclose(tau,data["integral"],atol=1e-10,rtol=1e-10):
                        raise ValueError("Stored optical depths differ from world-path replay")
                expected = (raw[:,3]*np.exp(-2*data["integral"])).astype(raw.dtype)
                if not np.array_equal(expected,weather[:,3]):
                    raise ValueError("Pre-encoding weather intensities violate attenuation formula")
            for field_name,cloud in (("statistics",weather),("serialized_statistics",data["serialized_xyzi"])):
                recomputed = cloud_stats(raw,cloud,data["emitted"],data["retained"],data["origin"],data["integral"],config["distance_bins_m"])
                assert_close(record[field_name],recomputed)
            rows = cloud_rows(data,row,frame,record["mode"],repeat,config["observability"]["intensity_threshold_sensitivity"])
            chunk = dict(bound=bound,rows_identity=identity(rows),rows=rows)
            atomic_json(path,chunk)
            print("P6 metric cloud committed: "+name,flush=True)
        all_rows.extend(chunk["rows"])
        chunk_hashes[path.relative_to(cache).as_posix()] = digest(path)
    atomic_json(cache/"CACHE_MANIFEST.json",dict(hashes=chunk_hashes,rows=len(all_rows),config_sha256=identity(config)))
    return all_rows


def audit_legacy(config, output, evidence):
    from shared_weather_benchmark.dataset_io import select_scenes
    from shared_weather_benchmark.audit import real_coordinate_audit
    from shared_weather_benchmark.compare import compare,aggregate_clouds
    root = Path(config["legacy_root"])
    old = evidence["config"]
    if select_scenes(old,smoke=True) != evidence["selection"]:
        raise ValueError("Legacy selection no longer matches frozen rule/source metadata")
    with tempfile.TemporaryDirectory(prefix=".recheck-",dir=output) as temporary:
        work = Path(temporary)
        coord = real_coordinate_audit(old,evidence["selection"],Path(config["train_root"]),work)
        for name in ("INPUT_MANIFEST.json","CLEAN_GT.json","FIELD_DOMAINS.json","REAL_COORDINATE_AUDIT.json"):
            assert_close(read_json(root/name),read_json(work/name))
        atomic_json(output/"P6A_COORDINATE_RECHECK.json",coord)
    rows = scan_clouds(root,old,output/"cache",repeat="legacy")
    denominator = check_denominators(rows,old["protocols"])
    original_rows = list(read_jsonl(root/"OBSERVABILITY_ROWS.jsonl"))
    indexed = {row_key(r):r for r in rows}
    if len(indexed) != len(rows) or len(original_rows) != len(rows) or len({row_key(r) for r in original_rows}) != len(rows):
        raise ValueError("Legacy observation row count/identity mismatch")
    tol = config["recompute_tolerance"]
    for before in original_rows:
        computed = indexed[row_key(before)]
        assert_close(before,{k:computed[k] for k in before},**tol)
    targets = legacy_targets(rows,old["observability"])
    assert_close(read_json(root/"OBSERVABILITY_TARGETS.json"),targets,**tol)
    summary = read_json(root/"OBSERVABILITY_SUMMARY.json")
    if summary["rows"] != len(rows) or summary["target_instances"] != len(targets):
        raise ValueError("Legacy summary denominator changed")
    old_denominator = sorted((r["scene"],r["timestamp"],r["object_id"],r["cav"],r["clean_hard_count"],r["clean_gt_sha256"])
                             for r in rows if r["mode"] == "E0")
    if summary["gt_cav_denominator_sha256"] != identity(old_denominator) or summary["contract"] != config["legacy_contract"]:
        raise ValueError("Legacy summary GT/CAV identity mismatch")
    recomputed_summary = {}
    import statistics
    for scene in sorted({r["scene"] for r in targets}):
        recomputed_summary[scene] = {}
        for mode in old["protocols"]:
            subset = [r for r in targets if r["scene"] == scene and r["mode"] == mode]
            gaps = [r["ego_best_neighbor_intensity_gap"] for r in subset if r["ego_best_neighbor_intensity_gap"] is not None]
            recomputed_summary[scene][mode] = dict(target_instances=len(subset),
                categories=dict(Counter(r["category"] for r in subset)),
                all_zero_hard_fraction=sum(r["all_zero_hard_returns"] for r in subset)/len(subset),
                ego_zero_neighbor_positive_fraction=sum(r["ego_zero_neighbor_positive"] for r in subset)/len(subset),
                all_zero_clean_fraction=sum(r["all_zero_clean_returns"] for r in subset)/len(subset),
                mean_ego_best_neighbor_intensity_gap=mean(gaps),
                median_ego_best_neighbor_intensity_gap=statistics.median(gaps) if gaps else None)
    assert_close(summary["by_scene"],recomputed_summary,**tol)
    # Run the frozen comparison formulas on freshly reduced target rows, in an
    # isolated tree. Only record/selection JSON is copied; no legacy writes.
    with tempfile.TemporaryDirectory(prefix=".comparison-",dir=output) as temporary:
        work = Path(temporary)
        for name in ("DATA_MANIFEST.json","SELECTION.json","FIELD_DOMAINS.json"):
            atomic_json(work/name,read_json(root/name))
        manifest = read_json(root/"DATA_MANIFEST.json")
        for name in manifest["records"]:
            atomic_json(work/name,read_json(root/name))
        write_jsonl(work/"OBSERVABILITY_ROWS.jsonl",({k:r[k] for k in original_rows[0]} for r in rows))
        atomic_json(work/"OBSERVABILITY_TARGETS.json",targets)
        recomputed = compare(old,work)
        assert_close(read_json(root/"COMPARISON.json"),recomputed,**tol)
    from shared_weather_benchmark_p6.comparison import matching_details
    scene_table,failures = [],{}
    for scene, data in recomputed["scenes"].items():
        frames = [f for f in evidence["selection"]["frames"] if f["scene"] == scene]
        gt = [f for f in read_json(root/"CLEAN_GT.json") if f["scene"] == scene]
        active_rows = [r for r in rows if r["mode"] == "E2" and r["scene"] == scene and r["active"]]
        details = matching_details(data["protocols"]["E2"]["physical"],data["protocols"]["E3"]["physical"],
            recomputed["fields"][scene]["E2"]["alpha_mean"],recomputed["fields"][scene]["E3"]["alpha_mean"],old["matching"])
        if details["matched"] != data["matching"]["finite_realization_matched"]:
            raise ValueError("Independent matching predicate differs")
        poses = [r["pose"][:2] for r in evidence["inputs"] if r["scene"] == scene]
        entry = dict(scene=scene,partition=data["partition"],frame_count=len(frames),
            cav_count=len(set(c for f in frames for c in f["cavs"])),
            cav_counts_by_frame=[len(f["cavs"]) for f in frames],
            active_cav_counts_by_frame=[len(f["active_cavs"]) for f in gt],
            fixed_clean_target_instances=sum(len(f["objects"]) for f in gt),
            fixed_unique_object_ids=len(set(oid for f in gt for oid in f["objects"])),
            active_target_cav_pairs=len(active_rows),valid_signal_pairs=sum(r["R"] is not None for r in active_rows),
            initial_xy_span_m=evidence["selection"]["profiles"][scene]["initial_xy_span_m"],
            selected_origin_max_xy_span_m=max(math.dist(a,b) for a in poses for b in poses),
            mean_path_length_m=data["protocols"]["E1"]["physical"]["mean_integral"]/old["field"]["mean_alpha"],
            E2_alpha_mean=recomputed["fields"][scene]["E2"]["alpha_mean"],
            E3_alpha_mean=recomputed["fields"][scene]["E3"]["alpha_mean"],
            path_length_distribution=dict(edges_m=old["distance_bins_m"],
                clean_points=[b["clean"] for b in data["protocols"]["E2"]["physical"]["bins"]]),
            **details)
        scene_table.append(entry)
        failures[scene] = entry
    from shared_weather_benchmark_p6.metrics import pearson
    associations = {key:pearson([r[key] for r in scene_table],
        [abs(r["mean_transmittance_delta"]) for r in scene_table])
        for key in ("cav_count","initial_xy_span_m","mean_path_length_m")}
    associations["absolute_alpha_drift"] = pearson([abs(r["alpha_delta"]) for r in scene_table],
        [abs(r["mean_transmittance_delta"]) for r in scene_table])
    write_csv(output/"P6A_SCENE_TABLE.csv",scene_table)
    atomic_json(output/"P6A_MATCH_FAILURES.json",dict(scenes=failures,
        grouped_by_cav_count={str(n):dict(total=sum(r["cav_count"] == n for r in scene_table),
            unmatched=sum(r["cav_count"] == n and not r["matched"] for r in scene_table))
            for n in sorted({r["cav_count"] for r in scene_table})},
        explanation_status="possible explanations only; four scenes cannot identify causes",
        descriptive_associations_with_absolute_T_residual=associations,
        possible_explanations=["CAV layout", "path-length sampling", "finite-domain field mean drift", "finite seed realization"],
        no_scene_or_seed_dropped=True))
    missing = Counter(r["missing_reason"] or "valid" for r in rows)
    report = dict(status="passed",legacy_contract=config["legacy_contract"],
        file_identity="all registered artifacts and original input bytes rehashed",
        statistic_recomputation="fresh signal reductions; frozen cloud/comparison formulas replayed in isolated output",
        physical_realism_validation="not_performed",rows=len(rows),target_instances=len(targets),
        denominator_sha256=denominator,missing_R=dict(missing),scenes=scene_table,
        selected_scenes=evidence["selection"]["selected"],
        complete_legacy_partitions=evidence["selection"]["all_scene_partition"],
        per_cav_physical={scene:{mode:{cav:aggregate_clouds(
            [r for r in evidence["records"] if r["scene"] == scene and r["mode"] == mode and r["cav"] == cav],
            len(old["distance_bins_m"])-1)
            for cav in sorted({r["cav"] for r in evidence["records"] if r["scene"] == scene})}
            for mode in ("E2","E3")} for scene in recomputed["scenes"]},
        coordinate_recheck=coord,read_only_legacy=True,
        issues=["Legacy G header reversed; actual best-neighbor minus ego",
                "Raw-count A/B inherit Clean under attenuation_only",
                "Legacy effective neighbor counts were not explicitly saved; P6 recomputes them",
                "Box membership is a point association proxy, not surface truth"],
        nonfinite_or_skipped_records=0,old_matching_failures_are_experimental_results=True)
    atomic_json(output/"P6A_AUDIT.json",report)
    atomic_json(output/"P6A_ARTIFACT_MANIFEST.json",dict(legacy_root=str(root),
        contract=config["legacy_contract"],artifacts=evidence["artifacts"],
        large_files="referenced by path/size/SHA256 only; not copied into audit exports"))
    atomic_text(output/"P6A_AUDIT.md","# P6-A 独立核验\n\n合同、原始输入和登记产物逐字节核验；统计从原始sidecar重新归约，旧公式在独立目录重放。实雾真实性未验证。\n\n"+
        "逐scene人数/目标分母、实际alpha、路径分布、每项匹配条件和失败原因见P6A_SCENE_TABLE.csv及P6A_MATCH_FAILURES.json。\n\n"+
        "旧G表头方向有误，实际为最佳邻车−ego；计数A/B沿用Clean结构，零强度比例为null。P6另存邻车数量、绝对能量与量化敏感性。\n\n"+
        "四个scene仅支持描述布局/均值漂移/路径与未匹配的共现；不能据此确定因果。匹配失败样本全部保留。\n")
    verify_external(evidence["artifacts"])
    return report

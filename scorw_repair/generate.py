"""Reuse Clean coordinate audit/ray sampler; cache each ray-field response once."""
from pathlib import Path
import numpy as np

from scorw_repair.core import (HERE, REPO, atomic_json, atomic_text, read_json, digest, identity,
    seed_for, save_arrays, transaction, Blocked, server_train)
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.e0_adapter import simulate as hahner, table_identity
from scorw_p7.field_pool import make_pool, restore_field
from scorw_p7.balanced_assignment import assignment
from scorw_p7.extract_rays import extract_scene
from scorw_repair.response import simulate, summary, validate


def opv2v_freeze(config, smoke):
    archived = read_json(HERE/"inputs/OPV2V_DEVELOPMENT_SELECTION.json")
    selected = sorted(archived["generator_development"])[:config["opv2v"]["smoke_scenes" if smoke else "full_scenes"]]
    if not selected or set(selected) & set(archived["excluded_future"]):
        raise Blocked("Invalid frozen development/future scene selection")
    frames = [f for f in archived["frames"] if f["scene"] in selected]
    if any(f["partition"] != "generator_development" for f in frames):
        raise Blocked("Repair scope must use historical development scenes only")
    # Reuse previously audited frame keys, never select on any new weather outcome.
    return dict(frames=frames,scenes=selected,parent_selection_sha256=archived["parent_selection_sha256"],
                independent_validation=False,choice="first lexical P7 development scenes; original two frames")


def coordinates(config, frozen, output, parent):
    from shared_weather_benchmark.audit import real_coordinate_audit
    from shared_weather_benchmark.dataset_io import source_paths
    root = server_train(config["train_root"])
    inputs = [dict(scene=f["scene"],timestamp=f["timestamp"],cav=c,pcd_sha256=digest(p),yaml_sha256=digest(y))
        for f in frozen["frames"] for c in f["cavs"] for p,y in [source_paths(root,f,c)]]
    # The caller already owns the stage transaction. Never lock/contract the same
    # directory recursively: that fails with flock on the Linux server.
    old = read_json(REPO/"shared_weather_benchmark/config/prototype.json")
    real_coordinate_audit(old,frozen,root,output)
    after=read_json(output/"INPUT_MANIFEST.json")
    if [(r["scene"],r["timestamp"],r["cav"],r["pcd_sha256"],r["yaml_sha256"]) for r in inputs] != [
            (r["scene"],r["timestamp"],r["cav"],r["pcd_sha256"],r["yaml_sha256"]) for r in after]:
        raise Blocked("COORDINATE_INPUT_CHANGED_DURING_AUDIT")
    return inputs


def original_b1(raw, config, point_key, indices):
    legacy = read_json(REPO/"shared_weather_benchmark/config/prototype.json")
    legacy["e0"] = config["b1"]
    weather, emitted, retained, kinds, tau, _ = hahner(raw,legacy,seed_for(config["seed"],"repair-b1",*point_key) % (2**32))
    data=dict(original_xyzi=raw,weather_xyzi=weather,emitted=emitted,retained=retained,origin=kinds,
        integral=tau,original_index=indices,
        point_identity=np.asarray([identity([*point_key,int(i)]) for i in indices]),frame_identity=np.asarray(point_key))
    validate(data)
    return data


def cached_response(raw, pose, field, config, key, indices, path, parent):
    def work(code):
        data = simulate(raw,pose,field,config["receiver"],key,indices)
        save_arrays(path/"response.npz",**data)
        atomic_json(path/"STATS.json",summary(data,config["distance_bins_m"]))
    transaction(path,dict(parent=parent,field=field.manifest(),indices=indices.tolist(),receiver=config["receiver"]),work)
    with np.load(path/"response.npz",allow_pickle=False) as stored:
        result = {k:stored[k] for k in stored.files}
    validate(result)
    if (not np.array_equal(result["original_index"],indices)
            or not np.array_equal(result["original_xyzi"],raw)):
        raise Blocked("Cached response changed Clean identity")
    return result


def execute_r1(config, frozen, coordinate_root, output, parent):
    sampling = read_json(REPO/"scorw_p7/config/protocol.json")
    sampling["sampling"] = dict(sampling["sampling"],max_per_stratum=config["opv2v"]["rays_per_stratum"],
        max_memberships_per_cloud=config["opv2v"]["max_memberships_per_cloud"])
    domains = read_json(coordinate_root/"FIELD_DOMAINS.json")
    inputs = read_json(coordinate_root/"INPUT_MANIFEST.json")
    original_index = {(r["scene"],r["timestamp"],r["cav"]):r for r in inputs}
    pools = {s:make_pool(s,sampling,domains[s]) for s in frozen["scenes"]}
    plans = {}
    for scene in frozen["scenes"]:
        frames = [f for f in frozen["frames"] if f["scene"] == scene]
        plans[scene] = assignment(frames[0]["cavs"],frames[0]["ego"])
    # Freeze all response assignments and budgets before field response computation.
    atomic_json(output/"R1_FIELD_POOL_FREEZE.json",dict(pools=pools,assignments=plans,
        response_randomness="none; deterministic continuum scattering; field identity is the complete stochastic input",
        receiver=config["receiver"],sampling=sampling["sampling"],selection=frozen))
    b1_table = table_identity(read_json(REPO/"shared_weather_benchmark/config/prototype.json"))
    rows, balance, target_rows, uniform_comparison = [], [], [], []
    for scene in frozen["scenes"]:
        clouds = extract_scene(scene,coordinate_root,output/"rays"/scene,sampling,parent)
        for cloud in clouds:
            audit = read_json(cloud/"SAMPLING.json")
            key = (scene,audit["timestamp"],audit["cav"])
            source = original_index[key]
            with np.load(cloud/"rays.npz",allow_pickle=False) as arrays:
                raw, indices = arrays["original_xyzi"].copy(),arrays["original_index"].copy()
                expected_point_ids=arrays["point_id"].copy()
            point_key = [*key,source["pcd_sha256"]]
            here = output/"responses"/scene/key[1]/key[2]
            constant = WorldField(kind="constant",mean_alpha=config["field"]["mean_alpha"],bounds=domains[scene])
            b0 = simulate(raw,source["pose"],WorldField(kind="constant",mean_alpha=0,bounds=domains[scene]),config["receiver"],point_key,indices)
            b1 = original_b1(raw,config,point_key,indices)
            b2 = cached_response(raw,source["pose"],constant,config,point_key,indices,here/"constant",parent)
            uniform_comparison.append(dict(scene=scene,timestamp=key[1],cav=key[2],
                b1=summary(b1,config["distance_bins_m"]),b2=summary(b2,config["distance_bins_m"]),
                equivalence_claim=False,reason="B1 selects the stronger separate hard/soft lookup peak and adds noise; B2 uses the total pulse waveform and marks shifted overlapping peaks unknown"))
            for mode,data in (("B0",b0),("B1",b1),("B2",b2)):
                if not np.array_equal(data["point_identity"],expected_point_ids):
                    raise Blocked("R1_ORIGINAL_POINT_IDENTITY_MISMATCH")
                save_arrays(here/(mode+".npz"),**data)
                rows.append(dict(scene=scene,timestamp=key[1],cav=key[2],mode=mode,stats=summary(data,config["distance_bins_m"])))
            signals=[]
            targets = read_json(cloud/"targets.json")
            metrics = {t["object_id"]:[] for t in targets}
            for j, manifest in enumerate(pools[scene]["fields"]):
                field = restore_field(manifest)
                data = cached_response(raw,source["pose"],field,config,point_key,indices,here/f"field_{j:02}",parent)
                if not np.array_equal(data["point_identity"],expected_point_ids):
                    raise Blocked("R1_POOL_POINT_IDENTITY_MISMATCH")
                st = summary(data,config["distance_bins_m"])
                signals.append([st["original_hard"],st["replaced_by_weather"],st["no_return"],st["unknown"],float(data["integral"].sum())])
                for t in targets:
                    slots = np.asarray(t["ray_slots"],dtype=int)
                    weights = np.asarray(t["weights"],dtype=float)
                    mask = data["retained"][slots]
                    numerator = float(np.sum(weights*data["weather_xyzi"][slots,3]*mask))
                    denom = t["fixed_weighted_clean_denominator"]
                    metrics[t["object_id"]].append(dict(field=j,hard=int(mask.sum()),
                        replaced=int((data["emitted"][slots]&(data["origin"][slots]==1)).sum()),
                        unknown=int((data["emitted"][slots]&(data["origin"][slots]==2)).sum()),
                        missing=int((~data["emitted"][slots]).sum()),signal_ratio=numerator/denom if denom else None,
                        hard_component_ratio=float(np.sum(weights*data["hard_signal"][slots]*mask))/denom if denom else None))
                # B3/B4 reuse the identical response; only protocol assignment changes.
                for mode,schedule in (("B3","E2"),("B4","E3")):
                    repeat = plans[scene]["schedules"][schedule][key[2]].index(j)
                    rows.append(dict(scene=scene,timestamp=key[1],cav=key[2],mode=mode,repeat=repeat,field=j,stats=st))
            array = np.asarray(signals,dtype=float)
            order = plans[scene]["schedules"]["E3"][key[2]]
            if not np.array_equal(np.sort(array,axis=0),np.sort(array[order],axis=0)):
                raise Blocked("R1_BALANCE_FAILED: response marginals differ")
            # Verify each original ray's encoded signal, origin and dropout distribution, not only means.
            paths = [here/f"field_{j:02}/response.npz" for j in range(20)]
            fingerprints = []
            for path in paths:
                with np.load(path,allow_pickle=False) as d:
                    fingerprints.append(identity({k:identity(d[k].tolist()) for k in ("weather_xyzi","origin","emitted","integral")}))
            if sorted(fingerprints) != sorted(fingerprints[j] for j in order):
                raise Blocked("R1_PER_RAY_BALANCE_FAILED")
            balance.append(dict(scene=scene,timestamp=key[1],cav=key[2],status="passed",fields=20,
                all_ray_distribution_identity=identity(sorted(fingerprints)),response_evaluations=20,
                extra_scattering_randomness=False,physical_threshold_calibrated=False))
            target_rows.extend(dict(target=t,fields=metrics[t["object_id"]],sampling="fixed weighted sample, not full counts") for t in targets)
    result = dict(status="numerical_engine_executed",rows=rows,balance=balance,target_rows=target_rows,
        pool_size=20,source_count_conservation="passed",b1_table=b1_table,uniform_comparison=uniform_comparison,
        data_identity_check="passed",physical_calibration_passed=False,real_comparison_performed_in_this_stage=False,
        real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE",
        engine="independent single-scattering pulse-response approximation with strongest total-peak selection",
        limitations=["not calibrated for OPV2V", "strongest peak selector is a model assumption", "multiple scattering omitted",
                     "shifted overlapping hard/soft peaks have unknown origin; target-centred hard-dominated peaks retain a model hard label", "output quantization is not a hardware threshold",
                     "clear-return-conditioned paths only", "sampled GT box associations are not surface truth"])
    atomic_json(output/"R1_PHYSICS_AUDIT.json",result)
    text="# R1 回波修复\n\n数值引擎已执行；真实物理标定未通过。\n\n"
    text+=(
        "B0/B1/B2及20场 B3/B4 使用同一组固定 Clean 光路。B3/B4 复用完整逐场接收结果，"
        "新增散射不含随机数，逐车及逐光路完整循环分布严格平衡。\n\n"
        "所有真实回波纯峰、替换、丢失、混合未知来源和距离统计见 R1_PHYSICS_AUDIT.json。"
        "原位置、硬分量占优的峰作为模型选中的原始目标回波；偏移混合峰保留 unknown。"
        "保留上下界是模型条件下的计数界限，不是实测光子来源保证；硬/软分量另行记录。\n")
    text+="\n| scene | 条件 | 原始回波保留下界 | 替换比例 | 无返回比例 | 未知来源比例 |\n|---|---|---:|---:|---:|---:|\n"
    for scene in frozen["scenes"]:
        for mode in ("B0","B1","B2","B3","B4"):
            rr=[r["stats"] for r in rows if r["scene"]==scene and r["mode"]==mode]
            total=sum(r["original"] for r in rr)
            values=[sum(r[k] for r in rr)/total if total else None for k in ("original_hard","replaced_by_weather","no_return","unknown")]
            text+=f"| {scene} | {mode} | "+" | ".join(str(v) for v in values)+" |\n"
    text+="\nB3/B4 的分母包含同一固定光路的20次重组合，不能当作20个独立 scene。"
    text+="所有数字来自固定采样光路，不能冒充全量点云丢失率。\n"
    atomic_text(output/"R1_RESULTS.md",text)
    return result

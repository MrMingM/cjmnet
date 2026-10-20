"""Small, sequential CPU materialization with per-cloud transactions."""
from __future__ import annotations

import io
import math
from pathlib import Path

from shared_weather_benchmark.audit import world_xyz
from shared_weather_benchmark.dataset_io import read_points, ensure_link
from shared_weather_benchmark.e0_adapter import simulate as simulate_e0
from shared_weather_benchmark.provenance import (atomic_bytes,atomic_json,read_json,digest,
                                               seed_for,identity,check_hashes)
from shared_weather_benchmark.ray_integrator import integrate_numpy
from shared_weather_benchmark.world_field import WorldField


def make_field(config, scene, cav, protocol, bounds):
    scope = [scene,"shared"] if protocol in {"E1","E2"} else [scene,"independent",cav]
    return WorldField(**config["field"],seed=seed_for(config["seed"],"world-field",*scope),
                      kind="constant" if protocol == "E1" else "smooth",bounds=bounds)


def materialize_fields(config,selection,domains):
    fields = {}
    for frame in selection["frames"]:
        for protocol in ("E1","E2","E3"):
            for cav in frame["cavs"]:
                key = "/".join([protocol,frame["scene"],cav if protocol == "E3" else "shared"])
                if key not in fields:
                    fields[key] = make_field(config,frame["scene"],cav,protocol,domains[frame["scene"]]).manifest()
    return dict(status="frozen_before_generation",algorithm="bounded-random-fourier-tanh-v1",
                fields=fields,identity=identity(fields),selection_sha256=identity(selection),
                marginal_matching="same ensemble law for E2/E3, no realized-statistic/AP fit")


def load_field(config,row,mode,domains):
    return make_field(config,row["scene"],row["cav"],mode,domains[row["scene"]])


def sidecar_validate(raw, weather, emitted, retained, origin, tau):
    import numpy as np
    n = len(raw)
    if (weather.shape != raw.shape or any(len(a) != n for a in (emitted,retained,origin,tau))
            or not np.isfinite(weather).all() or not np.isfinite(tau).all()
            or (tau < 0).any() or (weather[:,3] < 0).any() or (weather[:,3] > 1+1e-6).any()
            or not np.isin(origin,[0,1,2]).all() or (retained & ~emitted).any()
            or not np.array_equal(retained,emitted & (origin == 0))):
        raise ValueError("Invalid/lost return provenance")
    if retained.any() and not np.array_equal(weather[retained,:3],raw[retained,:3]):
        raise ValueError("A retained hard return moved in local coordinates")


def cloud_stats(raw, weather, emitted, retained, origin, tau, edges):
    import numpy as np
    ranges = np.linalg.norm(raw[:,:3],axis=1)
    emitted_ranges = np.linalg.norm(weather[emitted,:3],axis=1)
    bins = []
    for low,high in zip(edges,edges[1:]):
        ix = (ranges >= low) & (ranges < high)
        bins.append(dict(low_m=low,high_m=high,clean=int(ix.sum()),
                         retained_hard=int((ix & retained).sum()),
                         integral_sum=float(tau[ix].sum()),
                         t_sum=float(np.exp(-2*tau[ix]).sum()),
                         clean_intensity_sum=float(raw[ix,3].sum()),
                         weather_hard_intensity_sum=float(weather[ix & retained,3].sum())))
    invalid = emitted & ((np.linalg.norm(weather[:,:3],axis=1) <= 0)
                        | (np.linalg.norm(weather[:,:3],axis=1) > ranges+1e-5)
                        | ((weather[:,:3]*raw[:,:3]).sum(axis=1) <= 0))
    # 'Beyond source return' is a diagnostic, not a proof of physical invalidity.
    hist_edges = np.linspace(0,1,21)
    return dict(clean=len(raw),emitted=int(emitted.sum()),retained_hard=int(retained.sum()),
                removed_original=int((~retained).sum()),simulated=int((emitted & (origin==1)).sum()),
                unknown=int((emitted & (origin==2)).sum()),range_anomalies=int(invalid.sum()),
                mean_integral=float(tau.mean()),mean_transmittance=float(np.exp(-2*tau).mean()),
                clean_intensity_mean=float(raw[:,3].mean()),
                weather_intensity_mean=float(weather[emitted,3].mean()) if emitted.any() else None,
                intensity_histogram_edges=hist_edges.tolist(),
                clean_intensity_histogram=np.histogram(raw[:,3],hist_edges)[0].tolist(),
                weather_intensity_histogram=np.histogram(weather[emitted,3],hist_edges)[0].tolist(),
                weather_output_range_histogram=np.histogram(emitted_ranges,edges)[0].tolist(),bins=bins)


def validate_record(output, record, row, contract):
    if (record["contract"] != contract or record["input_pcd_sha256"] != row["pcd_sha256"]
            or record["input_yaml_sha256"] != row["yaml_sha256"]):
        raise ValueError("Cloud record identity mismatch")
    check_hashes(output,record["hashes"])
    if digest(row["pcd"]) != row["pcd_sha256"] or digest(row["yaml"]) != row["yaml_sha256"]:
        raise ValueError("Original input changed")


def generate(config, output, contract, resume):
    import numpy as np
    from reference_weather.common import write_pcd
    rows = read_json(output/"INPUT_MANIFEST.json")
    domains = read_json(output/"FIELD_DOMAINS.json")
    records = []
    for mode in config["protocols"]:
        for row in rows:
            relative = Path(mode)/"train"/row["scene"]/row["cav"]/(row["timestamp"]+".pcd")
            record_path = output/"records"/relative.with_suffix(".json")
            if record_path.exists():
                record = read_json(record_path)
                validate_record(output,record,row,contract)
                if not resume:
                    raise ValueError("Existing cloud commit requires RESUME=1")
                records.append(str(record_path.relative_to(output)))
                continue
            raw = read_points(row["pcd"])
            if digest(row["pcd"]) != row["pcd_sha256"] or digest(row["yaml"]) != row["yaml_sha256"]:
                raise ValueError("Input identity changed after coordinate audit")
            seed = seed_for(config["seed"],mode,row["scene"],row["timestamp"],row["cav"]) % (2**32)
            extra = {}
            if mode == "E0":
                weather,emitted,retained,origin,tau,extra = simulate_e0(raw,config,seed)
                field_hash = None
            else:
                field = load_field(config,row,mode,domains)
                xyz = world_xyz(raw,row["pose"])
                tau = integrate_numpy(field,row["pose"][:3],xyz,**config["integration_for_generation"])
                weather = raw.copy()
                weather[:,3] = raw[:,3]*np.exp(-2*tau)
                policy = config["return_policy"]
                emitted = np.ones(len(raw),dtype=bool) if policy["name"] == "attenuation_only" else weather[:,3] >= policy["threshold"]
                retained = emitted.copy()
                origin = np.zeros(len(raw),dtype=np.uint8)
                field_hash = field.manifest()["sha256"]
            sidecar_validate(raw,weather,emitted,retained,origin,tau)
            # A transaction is committed only when PCD, sidecar, and GT link verify.
            target = output/relative
            ensure_link(Path(row["yaml"]),target.with_suffix(".yaml"))
            write_pcd(target,weather[emitted])
            written = read_points(target)
            serialized = weather.copy()
            serialized[emitted] = written
            # Compare RGB serialization separately from pre-encoding physical values.
            if not np.array_equal(written[:,:3],weather[emitted,:3]) or not np.allclose(written[:,3],weather[emitted,3],atol=1/255+1e-7,rtol=0):
                raise ValueError("Written PCD differs from sidecar mapping")
            output_index = np.full(len(raw),-1,dtype=np.int64)
            output_index[emitted] = np.arange(int(emitted.sum()))
            payload = io.BytesIO()
            np.savez_compressed(payload,original_index=np.arange(len(raw),dtype=np.int64),
                original_xyzi=raw,weather_xyzi=weather,serialized_xyzi=serialized,
                emitted=emitted,retained=retained,origin=origin,integral=tau,
                output_index=output_index,
                frame_identity=np.array([row["scene"],row["timestamp"],row["cav"],row["pcd_sha256"]]))
            side_relative = Path("sidecars")/relative.with_suffix(".npz")
            atomic_bytes(output/side_relative,payload.getvalue())
            meta_relative = relative.with_suffix(".yaml")
            record = dict(contract=contract,mode=mode,scene=row["scene"],timestamp=row["timestamp"],
                cav=row["cav"],partition=row["partition"],seed=seed,field_sha256=field_hash,
                input_pcd_sha256=row["pcd_sha256"],input_yaml_sha256=row["yaml_sha256"],
                sidecar=side_relative.as_posix(),pcd=relative.as_posix(),
                point_id_rule="SHA256([scene,timestamp,CAV,input_pcd_sha256,original_index]) canonical JSON",
                origin_codes={"0":"original_hard_target_return","1":"simulated_weather_return","2":"unknown"},
                statistics=cloud_stats(raw,weather,emitted,retained,origin,tau,config["distance_bins_m"]),
                serialized_statistics=cloud_stats(raw,serialized,emitted,retained,origin,tau,config["distance_bins_m"]),
                upstream_statistics=extra,
                hashes={p.as_posix():digest(output/p) for p in (relative,side_relative,meta_relative)})
            validate_record(output,record,row,contract)
            atomic_json(record_path,record)
            records.append(str(record_path.relative_to(output)))
            print(f"P3 {mode} {row['scene']}/{row['timestamp']}/{row['cav']} committed",flush=True)
    atomic_json(output/"DATA_MANIFEST.json",dict(status="complete_generation",audit_required="P4/P5; consult VALIDATION_GATES.json",contract=contract,
        records=records,record_hashes={r:digest(output/r) for r in records},
        input_manifest_sha256=digest(output/"INPUT_MANIFEST.json"),
        clean_gt_sha256=digest(output/"CLEAN_GT.json"),protocols=config["protocols"],
        original_inputs_preserved=True,point_semantics="clean returns in GT boxes are association proxies"))

"""Integrate each unique frozen ray-field pair once; E2/E3 only reindex caches."""
import numpy as np

from shared_weather_benchmark.ray_integrator import integrate_numpy
from scorw_p7.core import read_json, digest, identity, transaction, save_arrays, atomic_json
from scorw_p7.field_pool import restore_field


def strict_balance(values, plan, tolerance):
    """Check every fixed-domain column, including original rays, not just means."""
    maximum = 0.0
    for cav in plan["cavs"]:
        a = values[np.asarray(plan["schedules"]["E2"][cav])]
        b = values[np.asarray(plan["schedules"]["E3"][cav])]
        if not np.array_equal(np.sort(a,axis=0),np.sort(b,axis=0)):
            raise ValueError("Per-CAV fixed-domain marginal distribution differs")
        error = float(np.max(np.abs(a.mean(axis=0)-b.mean(axis=0)))) if a.size else 0.0
        maximum = max(maximum,error)
        if not np.allclose(a.mean(axis=0),b.mean(axis=0),atol=tolerance["absolute"],rtol=tolerance["relative"]):
            raise ValueError("Per-CAV full-cycle mean differs")
    return maximum


def integrate_cloud(cloud, output, pool, plan, config, contract):
    with np.load(cloud/"rays.npz", allow_pickle=False) as data:
        rays = {k:data[k] for k in data.files}
    targets = read_json(cloud/"targets.json")
    sampling = read_json(cloud/"SAMPLING.json")
    n = len(rays["original_index"])
    if len(set(rays["point_id"].tolist())) != n or len(set(rays["original_index"].tolist())) != n:
        raise ValueError("Duplicate original ray identity")
    all_tau, diagnostics = [], []
    for j, manifest in enumerate(pool["fields"]):
        target = output/f"field_{j:02d}"
        request = dict(parent=contract,rays_sha256=digest(cloud/"rays.npz"),targets_sha256=digest(cloud/"targets.json"),
                       field_sha256=manifest["sha256"], integration=config["integration"])
        def calculate(code):
            field = restore_field(manifest)
            tau = integrate_numpy(field, rays["origin"], rays["world_endpoints"],
                                  config["integration"]["step_m"], config["integration"]["batch"])
            count = min(n,config["integration"]["probes_per_cloud"])
            probe = rays["world_endpoints"][:count]
            again = integrate_numpy(field,rays["origin"],probe,config["integration"]["step_m"],config["integration"]["batch"])
            fine = integrate_numpy(field,rays["origin"],probe,config["integration"]["step_m"]/2,config["integration"]["batch"])
            if not np.array_equal(tau[:count],again):
                raise ValueError("Identical ray/field integration is not bitwise reproducible")
            error = float(np.max(np.abs(again-fine))) if count else 0.0
            if error > config["integration"]["probe_step_halving_tolerance"]:
                raise ValueError("Step-halving numerical gate failed")
            save_arrays(target/"tau.npz",tau=tau)
            atomic_json(target/"NUMERICAL.json",dict(primary_ray_integrations=n,diagnostic_ray_integrations=2*count,
                step_halving_max_error=error,same_input_reproducible=True,field_sha256=manifest["sha256"]))
        transaction(target,request,calculate)
        with np.load(target/"tau.npz",allow_pickle=False) as data:
            tau = data["tau"]
        if tau.shape != (n,) or not np.isfinite(tau).all() or (tau < 0).any():
            raise ValueError("Invalid cached optical depths")
        all_tau.append(tau)
        diagnostics.append(read_json(target/"NUMERICAL.json"))
        print(f"P7 field {j+1}/20 {sampling['scene']}/{sampling['timestamp']}/{sampling['cav']} verified",flush=True)
    tau = np.stack(all_tau)
    T = np.exp(-2*tau)
    balance = dict(per_ray_tau_mean_max_error=strict_balance(tau,plan,config["tolerance"]),
                   per_ray_T_mean_max_error=strict_balance(T,plan,config["tolerance"]),
                   status="passed",columns=n,pool_size=20,ray_identity=sampling["rays_identity"])
    rows = []
    for row in targets:
        slots, weights = np.asarray(row["ray_slots"], dtype=np.int64), np.asarray(row["weights"],dtype=float)
        count = float(weights.sum())
        I = rays["original_xyzi"][slots,3].astype(np.float64)
        denom = float(np.dot(I,weights))
        if not np.isclose(denom,row["fixed_weighted_clean_denominator"],atol=1e-12,rtol=1e-12):
            raise ValueError("Frozen Clean signal denominator changed")
        signal = ((T[:,slots]*I[None,:])*weights[None,:]).sum(axis=1)
        R = signal/denom if denom > 0 else None
        if R is not None:
            if (R < 0).any() or (R > 1+1e-12).any():
                raise ValueError("Sampled retention ratio outside physical attenuation bounds")
            strict_balance(R[:,None],plan,config["tolerance"])
        item = {k:v for k,v in row.items() if k not in {"ray_slots","weights"}}
        rows.append(dict(item,R=R.tolist() if R is not None else None,
            tau=(tau[:,slots]@weights/count).tolist() if count else None,
            T=(T[:,slots]@weights/count).tolist() if count else None,
            sampled_weather_signal=signal.tolist(), sample_estimated_intensity_mean=(signal/count).tolist() if count else None,
            estimator=config["sampling"]["estimator"],full_cloud_exact=False))
    atomic_json(output/"MATRIX.json",dict(rows=rows,pool_identity=pool["sha256"],ray_identity=sampling["rays_identity"],
                matrix_identity=identity(rows),sampling=sampling,balance=balance,numerical=diagnostics,
                primary_ray_field_pairs=20*n,read_only_permutation_composition=True))
    return rows

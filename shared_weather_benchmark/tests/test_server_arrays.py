"""Mandatory P3 synthetic batch/PCD/E0 checks, run only on the Linux server."""
from shared_weather_benchmark.dataset_io import server_only,read_points
from shared_weather_benchmark.provenance import identity


def run_server_tests(config):
    server_only()
    import math
    from pathlib import Path
    import tempfile
    import numpy as np
    from shared_weather_benchmark.world_field import WorldField
    from shared_weather_benchmark.ray_integrator import integrate,integrate_numpy
    from shared_weather_benchmark.e0_adapter import simulate
    from shared_weather_benchmark.generator import sidecar_validate
    from reference_weather.common import write_pcd
    end = np.array([[3.,4.,0.],[20,8,2],[40,-12,3],[0,0,0]])
    origin = np.array([1.,2.,3.])
    field = WorldField(seed=53,length_m=8)
    batch = integrate_numpy(field,origin,end,.5,batch=1)
    batch2 = integrate_numpy(field,origin,end,.5,batch=128)
    scalar = np.array([integrate(field,origin.tolist(),p.tolist(),.5) for p in end])
    if not np.array_equal(batch,batch2) or not np.allclose(batch,scalar,atol=1e-12,rtol=0):
        raise ValueError("Synthetic scalar/batch/order integration mismatch")
    reverse = integrate_numpy(field,origin,end[::-1],.5)[::-1]
    if not np.array_equal(reverse,batch):
        raise ValueError("Batch point order changed path integrals")
    raw = np.array([[r,1.,.5,i] for r in (2.,10.,40.,100.,210.) for i in (.1,.5,1.)],dtype=np.float32)
    a = simulate(raw,config,42)
    b = simulate(raw,config,42)
    if any(not np.array_equal(x,y) for x,y in zip(a[:5],b[:5])):
        raise ValueError("E0 same seed did not reproduce")
    sidecar_validate(raw,*a[:5])
    # Exact E1 reduction to the frozen hard-target kernel, before/after rounding.
    from reference_weather.vendor.fog import fog_simulation as fog
    hard = raw.astype(float).copy(); hard[:,3] *= 255
    p = fog.ParameterSet(alpha=config["field"]["mean_alpha"])
    actual = fog.P_R_fog_hard(p,hard.copy())[:,3]/255
    f = WorldField(mean_alpha=p.alpha,kind="constant")
    tau = integrate_numpy(f,[0,0,0],raw[:,:3],.5)
    expected = raw[:,3]*np.exp(-2*tau)
    if not np.allclose(actual,expected,atol=.5/255+1e-10,rtol=0):
        raise ValueError("E1/frozen Hahner hard-component reduction failed")
    with tempfile.TemporaryDirectory(prefix="scor-w-synthetic-") as temporary:
        path = Path(temporary)/"sample.pcd"
        write_pcd(path,a[0][a[1]])
        written = read_points(path)
        if not np.array_equal(written[:,:3],a[0][a[1],:3]):
            raise ValueError("PCD row identity changed")
    return dict(status="passed",synthetic_checks=7,backend="NumPy + frozen E0 + PCD; CPU",
                scalar_batch_max_error=float(np.max(np.abs(batch-scalar))),
                e0_reproduction_array_sha256=identity([v.tolist() for v in a[:5]]),
                e1_rounding_max_error=float(np.max(np.abs(actual-expected))))

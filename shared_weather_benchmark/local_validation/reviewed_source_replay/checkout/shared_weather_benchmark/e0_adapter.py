"""Call the frozen Hahner kernel; preserved fifth feature tracks exact origins."""
from __future__ import annotations

from pathlib import Path
import hashlib

from shared_weather_benchmark.provenance import digest, read_json, REPO


def table_identity(config):
    # The exact alpha and tau used by the frozen reference control.
    alpha = config["e0"]["alpha"]
    table = Path(config["assets_root"])/"fog"/"integral_lookup_tables"/"original"/(
        f"integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_{alpha}.pickle")
    actual = digest(table)
    lock = read_json(REPO/"reference_weather"/"assets_lock.json")
    # Validate against the pinned public-resource digest, not an arbitrary pickle.
    entry = next((e for e in lock["assets"] if e["relative_path"].endswith(table.name)),None)
    content = table.read_bytes()
    blob = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
    if entry is None or len(content) != entry["size"] or blob != entry["git_blob_sha1"]:
        raise ValueError("E0 integral table does not match the frozen resource lock")
    return dict(path=str(table),sha256=actual,upstream_content_blob=blob,
                note="content hash calculation only; no Git executable or repository")


def simulate(raw, config, seed):
    import numpy as np
    from reference_weather.adapters import ReferenceSimulator
    sim = ReferenceSimulator("fog",{"fog":config["e0"]},config["assets_root"])
    fog = sim.backend
    output, stats = sim.augment(raw,seed)
    # Call the same frozen kernel with stable index as an extra preserved feature.
    indexed = np.column_stack((raw.astype(np.float64),np.arange(len(raw),dtype=np.float64)))
    indexed[:,3] *= 255
    fog.RNG = np.random.default_rng(int(seed))
    full,soft,info = fog.simulate_fog(sim.parameters,indexed,int(config["e0"]["noise"]),
        gain=bool(config["e0"]["gain"]),noise_variant=config["e0"]["noise_variant"])
    if not np.array_equal(full[:,4],np.arange(len(raw))):
        raise ValueError("E0 lost input row identities")
    soft_mask = np.zeros(len(raw),dtype=bool)
    if soft is not None:
        ids = soft[:,4].astype(np.int64)
        if not np.array_equal(ids.astype(float),soft[:,4]) or len(np.unique(ids)) != len(ids) or (ids < 0).any() or (ids >= len(raw)).any():
            raise ValueError("E0 soft return source indices invalid")
        soft_mask[ids] = True
    if int(soft_mask.sum()) != info["num_fog_responses"]:
        raise ValueError("E0 origin count mismatch")
    emitted = full[:,3] > 0
    full[:,3] /= 255
    weather = full[:,:4].astype(np.float32)
    if output.shape != weather[emitted].shape or not np.array_equal(output,weather[emitted]):
        raise ValueError("Instrumented E0 differs from frozen ReferenceSimulator.augment")
    origin = soft_mask.astype(np.uint8)  # 0 hard, 1 simulated; 2 reserved unknown.
    retained = emitted & ~soft_mask
    tau = config["e0"]["alpha"]*np.linalg.norm(raw[:,:3].astype(np.float64),axis=1)
    return weather,emitted,retained,origin,tau,stats

"""The unchanged v1 world field; all twenty identities frozen before integration."""
from shared_weather_benchmark.world_field import WorldField, ALGORITHM
from scorw_p7.core import identity, seed_for


def make_pool(scene, config, bounds):
    if config["pool_size"] != 20 or config["algorithm"] != ALGORITHM:
        raise ValueError("P7 requires the fixed twenty-field v1 algorithm")
    seeds = [seed_for(config["weather_root_seed"], "scorw-p7-pool-v1", scene, j) for j in range(20)]
    fields = [WorldField(**config["field"], seed=s, kind="smooth", bounds=bounds).manifest() for s in seeds]
    if len(set(seeds)) != 20 or len({f["sha256"] for f in fields}) != 20:
        raise ValueError("Weather pool identity collision")
    data = dict(scene=scene, seeds=seeds, fields=fields, count=20, algorithm=ALGORITHM,
                time="one fixed scene field across both timestamps; no inferred scan timing",
                interpretation="balanced finite weather-field pool, not twenty independent road scenes")
    return dict(data, sha256=identity(data))


def restore_field(manifest):
    field = WorldField(**{k: manifest[k] for k in ("mean_alpha", "length_m", "contrast", "seed", "kind", "bounds")},
                       modes=len(manifest["coefficients"]))
    if field.manifest() != manifest:
        raise ValueError("Frozen field coefficients/identity cannot be reproduced")
    return field

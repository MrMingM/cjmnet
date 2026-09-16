"""Fast dependency and label-semantics check for all TripleMixer adapters."""

import argparse
import json

import numpy as np

from gspr_supervision.triplemixer_adapter import TripleMixerWeatherAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triplemixer-root", required=True)
    parser.add_argument("--points", type=int, default=128)
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    yaw = rng.uniform(-np.pi, np.pi, args.points)
    radius = rng.uniform(5.0, 120.0, args.points)
    points = np.stack((radius * np.cos(yaw), radius * np.sin(yaw),
                       rng.uniform(-1.5, 0.5, args.points),
                       rng.uniform(0.05, 1.0, args.points)), axis=1).astype(np.float32)
    adapter = TripleMixerWeatherAdapter(args.triplemixer_root)
    report = {}
    for weather in ("fog", "rain", "snow"):
        result = adapter.simulate(points, weather, "moderate", seed=17)
        count = len(result["points"])
        assert result["reliability"].shape == (count,)
        assert result["noise_type"].shape == (count,)
        assert result["source_index"].shape == (count,)
        assert np.isfinite(result["points"]).all()
        assert np.isin(result["reliability"], [0.0, 1.0]).all()
        report[weather] = {
            "input": len(points), "output": count,
            "noise": int((result["reliability"] < 0.5).sum()),
            "lost": int(result["lost_count"]),
            "noise_types": np.unique(result["noise_type"]).tolist(),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

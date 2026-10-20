"""A bounded, continuous, deterministic world field; independent approximation."""
from __future__ import annotations

import math
import random

from shared_weather_benchmark.provenance import identity

ALGORITHM = "bounded-random-fourier-tanh-v1"


class WorldField:
    def __init__(self, mean_alpha=0.06, length_m=25., contrast=0.8, seed=0,
                 modes=32, kind="smooth", bounds=None, layers=None):
        if (not all(math.isfinite(float(x)) for x in (mean_alpha, length_m, contrast))
                or mean_alpha < 0 or length_m <= 0 or not 0 <= contrast < 1
                or not isinstance(modes, int) or modes < 1 or kind not in {"constant","smooth","layered"}):
            raise ValueError("Invalid field parameters")
        if bounds is not None and (len(bounds) != 6 or not all(map(math.isfinite,bounds))
                                   or any(bounds[i] >= bounds[i+3] for i in range(3))):
            raise ValueError("Invalid world bounds")
        self.mean, self.length, self.contrast = float(mean_alpha), float(length_m), float(contrast)
        self.seed, self.kind, self.bounds = int(seed), kind, bounds
        self.layers = layers or []
        for low, high, alpha in self.layers:
            if not all(map(math.isfinite,(low, high, alpha))) or low >= high or alpha < 0:
                raise ValueError("Invalid horizontal fog layer")
        ordered = sorted(self.layers)
        if any(a[1] > b[0] for a,b in zip(ordered,ordered[1:])):
            raise ValueError("Overlapping layers")
        rng = random.Random(self.seed)
        self.coefficients = [[rng.gauss(0, 1/self.length) for _ in range(3)]
                             + [rng.uniform(0,2*math.pi)] for _ in range(modes)]

    def _validate(self, point):
        if len(point) != 3 or not all(map(math.isfinite,point)):
            raise ValueError("Nonfinite world query")
        if self.bounds and any(not self.bounds[i] <= point[i] <= self.bounds[i+3] for i in range(3)):
            raise ValueError("World query outside audited field domain")

    def alpha(self, point, t=0):
        self._validate(point)
        if not math.isfinite(float(t)):
            raise ValueError("Nonfinite time")
        if self.kind == "constant":
            return self.mean
        if self.kind == "layered":
            return next((a for lo,hi,a in self.layers if lo <= point[2] < hi), self.mean)
        g = math.sqrt(2/len(self.coefficients))*sum(
            math.cos(sum(row[j]*point[j] for j in range(3))+row[3]) for row in self.coefficients)
        return self.mean*(1+self.contrast*math.tanh(g))

    def alpha_numpy(self, points):
        """Server batch path; compared to scalar implementation in P3 preflight."""
        import numpy as np
        p = np.asarray(points, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("Invalid field query array")
        if self.bounds and ((p < self.bounds[:3]).any() or (p > self.bounds[3:]).any()):
            raise ValueError("World batch query outside audited domain")
        if self.kind == "constant":
            return np.full(len(p), self.mean)
        if self.kind == "layered":
            out = np.full(len(p), self.mean)
            for low, high, alpha in self.layers:
                out[(p[:,2] >= low) & (p[:,2] < high)] = alpha
            return out
        g = np.zeros(len(p))
        # Fixed operation order; changing batching cannot change a point's field.
        for row in self.coefficients:
            g += np.cos(p[:,0]*row[0] + p[:,1]*row[1] + p[:,2]*row[2] + row[3])
        return self.mean*(1+self.contrast*np.tanh(g*math.sqrt(2/len(self.coefficients))))

    def manifest(self):
        data = dict(algorithm=ALGORITHM, mean_alpha=self.mean, length_m=self.length,
                    contrast=self.contrast, seed=self.seed, kind=self.kind, bounds=self.bounds,
                    outside="error", time="frozen within scene; t accepted, no inferred emission time",
                    layers=self.layers, coefficients=self.coefficients,
                    correlation="latent ensemble exp(-r^2/(2 length_m^2)); finite realization and tanh alter it")
        return dict(data, sha256=identity(data))

"""Midpoint integration of extinction along a straight world-space segment."""
import math


def integrate(field, start, end, step_m=0.5, t=0):
    if not math.isfinite(step_m) or step_m <= 0:
        raise ValueError("step_m must be positive")
    field.alpha(start,t)
    field.alpha(end,t)
    distance = math.dist(start,end)
    n = max(1, math.ceil(distance/step_m))
    integral = distance/n*math.fsum(field.alpha(
        [start[j]+(end[j]-start[j])*(i+.5)/n for j in range(3)],t) for i in range(n))
    if not math.isfinite(integral) or integral < 0:
        raise ValueError("Invalid optical depth")
    return integral


def transmittance(integral):
    if not math.isfinite(integral) or integral < 0:
        raise ValueError("Invalid optical depth")
    return math.exp(-2*integral)


def integrate_numpy(field, origin, endpoints, step_m=0.5, batch=256):
    import numpy as np
    if not math.isfinite(step_m) or step_m <= 0 or batch < 1:
        raise ValueError("Invalid integrator settings")
    end = np.asarray(endpoints, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    field.alpha_numpy(end)
    field.alpha(origin.tolist())
    result = np.empty(len(end), dtype=np.float64)
    # Group equal sample counts to avoid padding samples or a point-order RNG.
    lengths = np.linalg.norm(end-origin, axis=1)
    counts = np.maximum(1,np.ceil(lengths/step_m).astype(np.int64))
    for n in np.unique(counts):
        selected = np.flatnonzero(counts == n)
        u = (np.arange(n,dtype=np.float64)+.5)/n
        for offset in range(0,len(selected),batch):
            ix = selected[offset:offset+batch]
            samples = origin + (end[ix]-origin)[:,None,:]*u[None,:,None]
            values = field.alpha_numpy(samples.reshape(-1,3)).reshape(len(ix),n)
            result[ix] = values.mean(axis=1)*lengths[ix]
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Invalid batch optical depths")
    return result

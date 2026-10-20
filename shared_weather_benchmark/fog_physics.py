"""Layer 1 attenuation; Layer 2 uncalibrated scattering interface only."""
from __future__ import annotations

import math


def attenuate_intensity(intensity, integral):
    if not 0 <= intensity <= 1 or integral < 0 or not math.isfinite(integral):
        raise ValueError("Invalid intensity/optical depth")
    return intensity*math.exp(-2*integral)


def scattering_kernel_samples(distances, alphas, betas, overlaps):
    """Dimensionless shape beta(r)*overlap(r)*exp(-2*tau(r))/r^2.

    Trapezoidal cumulative optical depth on the supplied nodes. No sensor gain,
    pulse convolution, waveform peak selection, or calibrated return synthesis.
    Nodes must begin at zero with zero overlap; zero-node kernel is defined 0.
    """
    if not (len(distances) == len(alphas) == len(betas) == len(overlaps)) or len(distances) < 2:
        raise ValueError("Invalid scattering nodes")
    if distances[0] != 0 or overlaps[0] != 0:
        raise ValueError("Kernel needs zero origin and zero overlap")
    if any(not math.isfinite(v) or v < 0 for seq in (distances,alphas,betas,overlaps) for v in seq):
        raise ValueError("Invalid scattering values")
    if any(v > 1 for v in overlaps):
        raise ValueError("Overlap fraction must lie in [0,1]")
    tau, values = 0., [0.]
    for i in range(1,len(distances)):
        ds = distances[i]-distances[i-1]
        if ds <= 0:
            raise ValueError("Distances must increase")
        tau += ds*(alphas[i]+alphas[i-1])/2
        values.append(betas[i]*overlaps[i]*math.exp(-2*tau)/distances[i]**2)
    return values


def simulate_nonuniform_soft_returns(*args, **kwargs):
    raise NotImplementedError("Nonuniform waveform/scattering is not calibrated or implemented in v1")

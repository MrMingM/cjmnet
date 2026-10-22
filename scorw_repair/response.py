"""Single-scattering pulse response. No uniform-alpha lookup for heterogeneous fog."""
from __future__ import annotations

import math
import numpy as np

from shared_weather_benchmark.geometry import pose_matrix
from shared_weather_benchmark.ray_integrator import integrate_numpy
from scorw_repair.core import identity

C = 299792458.0
ORIGINS = {0:"original_hard_target_return", 1:"simulated_weather_return", 2:"unknown"}


def pulse(offset_m, width_m):
    x = np.asarray(offset_m)
    return np.where((x >= 0) & (x <= width_m), np.sin(np.pi*x/width_m)**2, 0.)


def single_response(field, origin, endpoint, clean_intensity, config, step=None, return_waveform=False):
    """Compute one path's waveform using beta(s), prefix tau(s) and a finite pulse.

    Pulse/overlap/gamma are inherited *model defaults*, not calibrated OPV2V hardware.
    Soft power is normalized by the hypothetical clear-target peak. No droplet RNG.
    """
    dr = float(step or config["response_step_m"])
    if dr <= 0 or not math.isfinite(dr) or not math.isfinite(clean_intensity) or clean_intensity < 0:
        raise ValueError("Invalid receiver input")
    origin, end = np.asarray(origin,dtype=float), np.asarray(endpoint,dtype=float)
    distance = float(np.linalg.norm(end-origin))
    if not math.isfinite(distance) or distance <= 1e-5:
        raise ValueError("A clear return must define a finite nonzero path")
    n = max(1,math.ceil(distance/dr)); ds = distance/n
    s = (np.arange(n)+.5)*ds
    alpha = field.alpha_numpy(origin+(end-origin)[None,:]*(s/distance)[:,None])
    tau = float(integrate_numpy(field,origin,end[None,:],config["attenuation_step_m"],1)[0])
    # New prefix integration is required for scattering; hard attenuation reuses the frozen integrator.
    tau_prefix = (np.cumsum(alpha)-.5*alpha)*ds
    prefix_total = float(np.sum(alpha)*ds)
    if not math.isclose(prefix_total,tau,abs_tol=config["prefix_absolute_tolerance"],rel_tol=config["prefix_relative_tolerance"]):
        raise ValueError("Prefix/target integration mismatch; reduce the frozen discretization in a new protocol")
    overlap = np.clip((s-config["overlap_start_m"])/(config["overlap_full_m"]-config["overlap_start_m"]),0,1)
    at_target = float(np.clip((distance-config["overlap_start_m"])/(config["overlap_full_m"]-config["overlap_start_m"]),0,1))
    if np.all(alpha == 0):
        return dict(tau=0.,unidentifiable_overlap=False,range_m=distance,intensity=clean_intensity,
                    origin=0,hard_signal=clean_intensity,soft_signal=0.,waveform_peak=clean_intensity,
                    zero_fog_identity=True)
    if at_target == 0:
        return dict(tau=tau,unidentifiable_overlap=True,range_m=distance,intensity=clean_intensity,
                    origin=2,hard_signal=clean_intensity,soft_signal=0.,waveform_peak=clean_intensity)
    beta = config["beta_over_alpha"]*alpha
    width = C*config["pulse_half_width_s"]
    beta0 = config["gamma"]/np.pi
    density = (clean_intensity*distance**2/(beta0*at_target) * beta * overlap/s**2
               * np.exp(-2*tau_prefix) * (2/C)*ds)
    # Convolution on the same physical range grid, with exact target-peak insertion below.
    offsets = np.arange(math.ceil(width/ds)+1)*ds
    kernel = pulse(offsets,width)
    soft = np.convolve(density,kernel)
    u = s[0]+np.arange(len(soft))*ds
    hard = clean_intensity*math.exp(-2*tau)*pulse(u-distance,width)
    total = hard+soft
    winner = int(np.argmax(total))
    # Ensure a hard target's exact peak is represented even for long rays/coarse grids.
    target_u = distance+width/2
    target_soft = float(np.sum(density*pulse(target_u-s,width)))
    target_hard = clean_intensity*math.exp(-2*tau)
    if target_hard+target_soft >= total[winner]:
        r,hs,ss = distance,target_hard,target_soft
    else:
        r,hs,ss = float(u[winner]-width/2),float(hard[winner]),float(soft[winner])
    power = hs+ss
    if not all(math.isfinite(v) and v >= 0 for v in (hs,ss,power)):
        raise ValueError("Nonfinite/negative received power")
    # Quantization is an output convention, NOT a calibrated detection threshold.
    quantum = config["intensity_quantum"]
    encoded = float(np.clip(np.rint(power/quantum)*quantum,0,1))
    valid = encoded > 0 and r > 0
    # A target-centred, hard-dominated peak retains the original return; this does not imply photon purity.
    # Shifted overlapping peaks remain unknown. Component powers are always recorded separately.
    if r == distance and hs > 0 and hs >= ss:
        origin_type = 0
    elif hs <= config["pure_component_fraction"]*ss:
        origin_type = 1
    else:
        origin_type = 2
    result = dict(tau=tau,unidentifiable_overlap=False,range_m=r,intensity=encoded,origin=origin_type,
                hard_signal=hs,soft_signal=ss,waveform_peak=power,emitted=valid,
                hard_dominant=hs >= ss,finite_pulse_width_m=width,
                high_optical_depth_warning=tau > 1.)
    if return_waveform:
        # Synthetic diagnostics only. These normalized waveform areas have units
        # of encoded intensity * metres; they are NOT calibrated joules/watts.
        result.update(waveform_grid_m=u,soft_waveform=soft,hard_waveform=hard,
            soft_area=float(soft.sum()*ds),
            soft_area_discrete_expected=float(density.sum()*kernel.sum()*ds),
            soft_area_continuum_pulse_expected=float(density.sum()*width/2))
    return result


def simulate(raw, pose, field, config, frame_identity, original_indices=None):
    raw = np.asarray(raw)
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all() or (raw[:,3] < 0).any() or (raw[:,3] > 1).any():
        raise ValueError("Invalid original XYZI")
    m = np.asarray(pose_matrix(pose)); origin = m[:3,3]
    world = raw[:,:3].astype(float)@m[:3,:3].T+origin
    n = len(raw)
    indices = np.arange(n,dtype=np.int64) if original_indices is None else np.asarray(original_indices,dtype=np.int64)
    if len(indices) != n or len(np.unique(indices)) != n or (indices < 0).any():
        raise ValueError("Invalid original point indices")
    weather = raw.astype(float).copy()
    kinds = np.full(n,2,dtype=np.uint8); emitted = np.zeros(n,dtype=bool)
    tau = np.zeros(n); hard = np.zeros(n); soft = np.zeros(n); overlap_unknown = np.zeros(n,dtype=bool)
    for i in range(n):
        row = single_response(field,origin,world[i],float(raw[i,3]),config)
        kinds[i] = row["origin"]
        tau[i],hard[i],soft[i] = row["tau"],row["hard_signal"],row["soft_signal"]
        overlap_unknown[i] = row["unidentifiable_overlap"]
        # Unknown near-field overlap keeps an observed point, explicitly unmodelled.
        emitted[i] = row.get("emitted",True)
        length = float(np.linalg.norm(raw[i,:3]))
        if kinds[i] == 0:
            weather[i,:3] = raw[i,:3]
        else:
            weather[i,:3] = raw[i,:3]*(row["range_m"]/length)
        weather[i,3] = row["intensity"]
    out = dict(original_xyzi=raw.copy(),weather_xyzi=weather.astype(np.float32),
        emitted=emitted,origin=kinds,retained=emitted & (kinds == 0),integral=tau,
        hard_signal=hard,soft_signal=soft,overlap_unknown=overlap_unknown,
        original_index=indices,
        point_identity=np.asarray([identity([*frame_identity,int(i)]) for i in indices]),
        frame_identity=np.asarray(frame_identity))
    validate(out)
    return out


def validate(data):
    n = len(data["original_xyzi"])
    if data["original_xyzi"].shape!=(n,4) or data["weather_xyzi"].shape!=(n,4):
        raise ValueError("Invalid response array shape")
    if any(np.asarray(data[k]).shape != (n,) for k in ("emitted","origin","retained","integral","original_index","point_identity")):
        raise ValueError("Return source count mismatch")
    if (not np.isin(data["origin"],[0,1,2]).all() or not np.isfinite(data["weather_xyzi"]).all()
            or not np.isfinite(data["original_xyzi"]).all()
            or (data["weather_xyzi"][:,3]<0).any() or (data["weather_xyzi"][:,3]>1).any()
            or data["original_index"].dtype.kind not in "iu" or data["emitted"].dtype.kind!="b"
            or data["retained"].dtype.kind!="b" or len(np.unique(data["point_identity"]))!=n
            or not np.isfinite(data["integral"]).all() or (data["integral"] < 0).any()
            or len(np.unique(data["original_index"])) != n or (data["original_index"] < 0).any()):
        raise ValueError("Invalid response provenance")
    for key in ("hard_signal","soft_signal"):
        if key in data and (np.asarray(data[key]).shape!=(n,) or not np.isfinite(data[key]).all() or (data[key]<0).any()):
            raise ValueError("Invalid waveform component provenance")
    emitted, kinds = data["emitted"],data["origin"]
    if not np.array_equal(data["retained"],emitted & (kinds == 0)):
        raise ValueError("Retained source attribution mismatch")
    if not np.array_equal(data["weather_xyzi"][data["retained"],:3],data["original_xyzi"][data["retained"],:3]):
        raise ValueError("Retained original target moved")
    counts = sum(int((emitted & (kinds == k)).sum()) for k in ORIGINS)+int((~emitted).sum())
    if counts != n:
        raise ValueError("Origin count conservation failed")


def summary(data, edges):
    validate(data)
    raw,out = data["original_xyzi"],data["weather_xyzi"]
    emitted,kinds = data["emitted"],data["origin"]
    ranges = np.linalg.norm(raw[:,:3],axis=1)
    weather_ranges = np.linalg.norm(out[:,:3],axis=1)
    n = len(raw)
    def quantiles(values):
        return np.quantile(values,[0,.1,.5,.9,1]).tolist() if len(values) else None
    return dict(original=n,original_hard=int((emitted & (kinds==0)).sum()),
        replaced_by_weather=int((emitted & (kinds==1)).sum()),unknown=int((emitted & (kinds==2)).sum()),
        no_return=int((~emitted).sum()),true_return_retention_lower_bound=float(data["retained"].mean()) if n else None,
        true_return_retention_upper_bound=float((emitted & (kinds!=1)).mean()) if n else None,
        hard_dominant_peak_proxy=int((emitted & (data.get("hard_signal",np.zeros(n))>=data.get("soft_signal",np.full(n,np.inf)))).sum()),
        unmodelled_near_field=int(data.get("overlap_unknown",np.zeros(n,dtype=bool)).sum()),
        retention_scope="model-selected target-centred hard-dominated peaks; shifted overlapping peaks unknown; bounds are conditional on this model, not empirically guaranteed photon provenance",
        original_distance_bins=np.histogram(ranges,edges)[0].tolist(),
        output_distance_bins=np.histogram(weather_ranges[emitted],edges)[0].tolist(),
        scatter_distance_quantiles=quantiles(weather_ranges[emitted & (kinds==1)]),
        distance_error_quantiles=quantiles((weather_ranges-ranges)[emitted]),
        no_return_fraction=float((~emitted).mean()) if n else None,
        replacement_fraction=float((emitted & (kinds==1)).mean()) if n else None,
        high_tau_fraction=float((data["integral"] > 1).mean()) if n else None,
        per_distance=[dict(low=a,high=b,count=int(((ranges>=a)&(ranges<b)).sum()),
            hard=int(((ranges>=a)&(ranges<b)&data["retained"]).sum()),
            replaced=int(((ranges>=a)&(ranges<b)&emitted&(kinds==1)).sum()),
            lost=int(((ranges>=a)&(ranges<b)&~emitted).sum())) for a,b in zip(edges,edges[1:])])

"""Independent uniform-waveform checks and exact replay of the frozen Hahner LUT kernel."""
import hashlib
import math
from pathlib import Path
import pickle

import numpy as np
from scipy.integrate import quad

from scorw_repair.core import REPO, Blocked, read_json, digest, atomic_json, atomic_text
from scorw_repair.response import C, single_response
from shared_weather_benchmark.world_field import WorldField


def table_resource(production=False):
    old=read_json(REPO/"shared_weather_benchmark/config/prototype.json")
    name=f"integral_0m_to_200m_stepsize_0.1m_tau_h_20ns_alpha_{old['e0']['alpha']}.pickle"
    candidates=[Path(old["assets_root"])/"fog/integral_lookup_tables/original"/name,
                REPO/"related_code/LiDAR_fog_sim/integral_lookup_tables/original"/name]
    if production:candidates=candidates[:1]
    path=next((p for p in candidates if p.is_file()),None)
    if path is None:
        raise Blocked("HAHNER_PINNED_LOOKUP_MISSING: "+name)
    payload=path.read_bytes()
    entry=next((r for r in read_json(REPO/"reference_weather/assets_lock.json")["assets"]
                if r["relative_path"].endswith(name)),None)
    blob=hashlib.sha1(f"blob {len(payload)}\0".encode()+payload).hexdigest()
    if entry is None or len(payload)!=entry["size"] or blob!=entry["git_blob_sha1"]:
        raise Blocked("HAHNER_PINNED_LOOKUP_CONTENT_MISMATCH")
    return path,dict(path=str(path),sha256=digest(path),content_blob_sha1=blob,bytes=len(payload))


def uniform_soft_quadrature(u, distance, intensity, alpha, receiver):
    """Hahner continuous equation in pulse-time coordinates, independently integrated.

    u is the unshifted time-of-flight range coordinate. Output is relative to the
    inferred clear-target peak, not calibrated received watts.
    """
    width=C*receiver["pulse_half_width_s"]
    target_overlap=float(np.clip((distance-receiver["overlap_start_m"])/(
        receiver["overlap_full_m"]-receiver["overlap_start_m"]),0,1))
    if target_overlap<=0:raise ValueError("Clear power normalization is unidentified")
    def kernel(v):
        s=u-width*v
        if s<=receiver["overlap_start_m"] or s>=distance:return 0.
        overlap=min(1.,(s-receiver["overlap_start_m"])/(
            receiver["overlap_full_m"]-receiver["overlap_start_m"]))
        return math.sin(math.pi*v)**2*math.exp(-2*alpha*s)*overlap/s**2
    points=sorted({(u-s)/width for s in (receiver["overlap_start_m"],receiver["overlap_full_m"],distance)
                   if 0<(u-s)/width<1})
    integral=quad(kernel,0,1,points=points,epsabs=1e-13,epsrel=1e-10)[0]
    scale=intensity*distance**2/(receiver["gamma"]/math.pi*target_overlap)
    return scale*(receiver["beta_over_alpha"]*alpha)*(2*receiver["pulse_half_width_s"])*integral


def check(config, output=None):
    from reference_weather.vendor.fog import fog_simulation as fog
    receiver=config["receiver"]; alpha=config["field"]["mean_alpha"]
    params=fog.ParameterSet(alpha=alpha,gamma=config["b1"]["gamma"])
    if (alpha!=.06 or receiver["pulse_half_width_s"]!=params.tau_h
            or not math.isclose(receiver["beta_over_alpha"]*alpha,params.beta,rel_tol=1e-14)
            or not math.isclose(receiver["gamma"]/math.pi,params.beta_0,rel_tol=1e-14)):
        raise Blocked("HAHNER_REGRESSION_PARAMETER_DOMAIN_CHANGED")
    path,table_identity=table_resource()
    # Unpickle only after the frozen public resource content has been verified.
    lookup=pickle.loads(path.read_bytes())
    grid=[(r,i) for r in config["regression"]["distances_m"] for i in config["regression"]["intensities_adc"]]
    raw=np.asarray([[r,0,0,i,k] for k,(r,i) in enumerate(grid)],dtype=float)
    ranges=raw[:,0]; analytic=raw[:,3]*np.exp(-2*alpha*ranges)
    expected=raw.copy(); expected[:,3]=np.round(analytic)
    hard=fog.P_R_fog_hard(params,raw.copy())
    if not np.array_equal(hard,expected):raise Blocked("HAHNER_HARD_ADC_REGRESSION_FAILED")
    replacements=[]; table_soft=[]
    for k,(r,i) in enumerate(grid):
        fog_distance,value=lookup[min(float(str(round(r,1))),200)]
        soft=min(value*i*r*r*params.beta/params.beta_0,255)
        table_soft.append(soft)
        replace=soft>expected[k,3]; replacements.append(replace)
        if replace:
            expected[k,:3]=raw[k,:3]*(fog_distance/r);expected[k,3]=soft
    previous_path,previous_rng=fog.INTEGRAL_PATH,fog.RNG
    try:
        fog.INTEGRAL_PATH=path.parent;fog.RNG=np.random.default_rng(config["seed"])
        observed,soft_points,info=fog.simulate_fog(params,raw.copy(),noise=0,gain=False)
    finally:
        fog.INTEGRAL_PATH=previous_path;fog.RNG=previous_rng
    tolerance=config["regression"]["lookup_replay_absolute_tolerance"]
    if not np.allclose(observed,expected,rtol=0,atol=tolerance):
        raise Blocked("HAHNER_SOFT_REPLACEMENT_REPLAY_FAILED")
    if info["num_fog_responses"]!=sum(replacements):raise Blocked("HAHNER_ORIGIN_REPLAY_FAILED")
    if not np.array_equal(observed[:,3]>0,expected[:,3]>0):raise Blocked("HAHNER_NO_RETURN_REPLAY_FAILED")
    strict=[]; descriptions=[]
    field=WorldField(kind="constant",mean_alpha=alpha)
    for k,(distance,adc) in enumerate(grid):
        row=single_response(field,[0]*3,[distance,0,0],adc/255,receiver,return_waveform=True)
        new_hard=adc/255*math.exp(-2*row["tau"])
        hard_error=abs(new_hard-analytic[k]/255)
        u=row["range_m"]+C*receiver["pulse_half_width_s"]/2
        independent=uniform_soft_quadrature(u,distance,adc/255,alpha,receiver)
        soft_error=abs(row["soft_signal"]-independent)/max(abs(independent),1e-12)
        energy_error=abs(row["soft_area"]-row["soft_area_discrete_expected"])
        pulse_area_error=abs(row["soft_area"]-row["soft_area_continuum_pulse_expected"])/max(row["soft_area"],1e-12)
        if (hard_error>config["tests"]["hard_absolute_tolerance"]
                or soft_error>config["tests"]["waveform_relative_tolerance"]
                or energy_error>1e-10*max(row["soft_area"],1)
                or pulse_area_error>config["tests"]["waveform_relative_tolerance"]
                or np.any(row["soft_waveform"]<0) or not np.isfinite(row["soft_waveform"]).all()):
            raise Blocked("NEW_ENGINE_UNIFORM_CONTINUOUS_WAVEFORM_REGRESSION_FAILED")
        strict.append(dict(distance_m=distance,adc=adc,hard_error=hard_error,
            soft_relative_error=soft_error,waveform_area_identity_absolute_error=energy_error,
            pulse_area_relative_error=pulse_area_error))
        descriptions.append(dict(distance_m=distance,clean_adc=adc,
            hahner=dict(output_distance_m=float(np.linalg.norm(observed[k,:3])),
                intensity=float(observed[k,3]/255),soft_peak_after_clip=float(table_soft[k]/255),
                replaced=bool(replacements[k]),no_return=bool(observed[k,3]<=0)),
            new_engine=dict(output_distance_m=row["range_m"],intensity=row["intensity"],
                selected_soft_unclipped=row["soft_signal"],replaced=bool(row["emitted"] and row["origin"]==1),
                selected_peak_origin=row["origin"],unknown=bool(row["emitted"] and row["origin"]==2),
                no_return=not row["emitted"]),
            strict_selected_return_equivalence_required=False))
    convergence=[]
    for kind in ("constant","smooth"):
        f=WorldField(kind=kind,mean_alpha=alpha,seed=12,
            length_m=config["field"]["length_m"],contrast=config["field"]["contrast"],modes=config["field"]["modes"])
        refined=[single_response(f,[0]*3,[45,0,0],.8,receiver,step=h) for h in (.1,.05,.025)]
        error=abs(refined[1]["waveform_peak"]-refined[2]["waveform_peak"])/max(refined[2]["waveform_peak"],1e-12)
        range_error=abs(refined[1]["range_m"]-refined[2]["range_m"])
        if error>config["tests"]["waveform_relative_tolerance"] or range_error>config["tests"]["range_convergence_m"]:
            raise Blocked("RECEIVER_SPATIAL_CONVERGENCE_FAILED")
        convergence.append(dict(kind=kind,steps_m=[.1,.05,.025],relative_peak_change=error,
            range_change_m=range_error,hard_integrator_step_m=receiver["attenuation_step_m"]))
    result=dict(status="passed",scope="synthetic uniform physics/regression; no real fog data",
        mathematical_tests=strict,table=table_identity,
        strict_checks=dict(hard_transmission="passed",upstream_hard_ADC="passed",
            upstream_soft_LUT_replay="passed",upstream_replacement_and_no_return_replay="passed",
            new_soft_vs_independent_time_quadrature="passed",nonnegative_waveform_and_area="passed",
            constant_and_smooth_spatial_convergence="passed"),convergence=convergence,
        descriptive_selected_return_comparison=descriptions,
        behavior_totals={engine:{key:sum(int(r[engine][key]) for r in descriptions)
            for key in ("replaced","no_return")} for engine in ("hahner","new_engine")},
        comparison_boundaries=["equal hard attenuation before quantization",
            "new uniform soft signal agrees with independent Hahner continuous single-scattering equation at the same time coordinate",
            "LUT peak grid/time convention, separate-peak competition, noise and rounding make selected returns descriptive comparisons only",
            "B1 production retains frozen noise=10; this regression uses noise=0 only to isolate equations",
            "no-return is zero output code, not calibrated hardware failure",
            "normalized waveform area conservation is numerical; no absolute emitted/received energy calibration"],
        physical_calibration_passed=False,real_validation_status="NOT_VALIDATED_NO_REAL_REFERENCE")
    if output is not None:
        atomic_json(output/"R1_HAHNER_REGRESSION.json",result)
        atomic_text(output/"R1_HAHNER_REGRESSION.md","# Hahner 均匀雾回归\n\npassed\n\n"
            f"固定 {len(grid)} 条合成光路。硬回波衰减、原版 ADC 舍入、原版查表软回波/替换/零输出码逐项核对。"
            "新引擎的均匀软回波在相同时间坐标与独立时间积分比较，另检验非负性及波形面积。\n\n"
            "原版分别比较硬/软峰并加距离噪声；新引擎选择相加波形的峰。最终距离、替换率和丢失率只作描述比较，"
            "没有要求两种接收规则逐点相同。真实雾天标定仍缺失。\n")
    return result

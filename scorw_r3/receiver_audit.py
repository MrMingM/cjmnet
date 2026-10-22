"""R3-A: algebra, receiver counterfactuals and descriptive immutable-cache audits."""
from copy import deepcopy
from pathlib import Path
import math
import pickle
import numpy as np

from scorw_r3.core import (Blocked,read_json,atomic_json,atomic_text,identity,digest,load_arrays,quantiles)
from scorw_repair.response import single_response,C,validate
from scorw_repair.regression import check as regression,table_resource
from shared_weather_benchmark.world_field import WorldField
from scorw_p7.field_pool import restore_field


def describe(row):
    emitted=row.get("emitted",True)
    return dict(range_m=row["range_m"],peak=row["waveform_peak"],intensity=row["intensity"],
        hard=row["hard_signal"],soft=row["soft_signal"],tau=row["tau"],emitted=emitted,
        retained=bool(emitted and row["origin"]==0),replaced=bool(emitted and row["origin"]==1),
        unknown=bool(emitted and row["origin"]==2),no_return=not emitted,
        origin=row["origin"],unmodelled=row["unidentifiable_overlap"])


def intensity_test(c):
    results=[]
    for distance in (5.,10.,30.,80.,180.):
        for alpha in (.01,.06,.12):
            field=WorldField(kind="constant",mean_alpha=alpha)
            rows=[single_response(field,[0]*3,[distance,0,0],i,c["receiver"],return_waveform=True) for i in (1/255,.1,.5,1.)]
            ref=rows[-1];errors=[]
            for intensity,row in zip((1/255,.1,.5,1.),rows):
                err=abs(row["waveform_peak"]/intensity-ref["waveform_peak"])
                errors.append(err)
                if not math.isclose(row["range_m"],ref["range_m"],abs_tol=1e-12,rel_tol=0) or row["origin"]!=ref["origin"]:
                    raise Blocked("R3A_INTENSITY_ARGMAX_INVARIANCE_FAILED")
                if not math.isclose(row["waveform_peak"]/intensity,ref["waveform_peak"],abs_tol=1e-12,rel_tol=1e-12):
                    raise Blocked("R3A_POSITIVE_INTENSITY_LINEARITY_FAILED")
            results.append(dict(distance_m=distance,alpha=alpha,maximum_absolute_linearity_error=max(errors),
                observations=[dict(clean_intensity=i,**describe(r)) for i,r in zip((1/255,.1,.5,1.),rows)],
                soft_peak_over_target_hard=float(max(ref["soft_waveform"])/(math.exp(-2*ref["tau"])))) )
    special=[]
    for alpha in (0.,.06):
        for distance in (.5,10.):
            special.append(dict(alpha=alpha,distance_m=distance,clean_intensity=0.,
                **describe(single_response(WorldField(kind="constant",mean_alpha=alpha),[0]*3,[distance,0,0],0.,c["receiver"]))))
    return dict(status="passed",cases=results,zero_input_cases=special,
        derivation="For fixed path/default beta0, H(I,u)=I*h(u), S(I,u)=I*s(u); S/H and argmax(H+S) cancel positive I. ADC validity, zero intensity and unmodelled overlap are exceptions.",
        upstream="Hahner P_R_fog_soft also multiplies original intensity*R^2*beta/beta0; inherited normalization plus fixed assumed gamma, not an additional R3 tuning",
        reflectance_limit="Actual intensity can reflect material, geometry and encoding. If emitted power is fixed and only true reflectance changes, real soft power need not scale with target intensity. Under that counterfactual, the inherited normalization suppresses low-reflectance vulnerability; no universally known empirical bias direction without radiometric calibration.",
        calibration_status="REAL_WORLD_VALIDATION_MISSING")


def differences(c,reg):
    from reference_weather.vendor.fog.fog_simulation import ParameterSet
    path,table_id=table_resource();lookup=pickle.loads(path.read_bytes())
    p=ParameterSet(alpha=.06,gamma=c["b1"]["gamma"]);out=[]
    for old in reg["descriptive_selected_return_comparison"]:
        r,i=old["distance_m"],old["clean_adc"]/255
        wave=single_response(WorldField(kind="constant",mean_alpha=.06),[0]*3,[r,0,0],i,c["receiver"],return_waveform=True)
        hard=i*math.exp(-2*wave["tau"]);hard_code=float(np.rint(hard*255))/255
        soft=float(max(wave["soft_waveform"]));ix=int(np.argmax(wave["soft_waveform"]))
        soft_range=float(wave["waveform_grid_m"][ix]-C*c["receiver"]["pulse_half_width_s"]/2)
        def separate(soft,hard,quantize):
            weather=soft>hard;value=min(soft if weather else hard,1.)
            encoded=float(np.clip(np.rint(value*255)/255,0,1)) if quantize else value
            valid=encoded>0
            return dict(intensity=encoded,replaced=bool(weather and valid),no_return=not valid)
        variants=dict(legacy_LUT_separate=old["hahner"],
            continuum_separate_legacy_codes=separate(soft,hard_code,False),
            continuum_separate_quantized_codes=separate(soft,hard_code,True),
            continuum_separate_unrounded_competition=separate(soft,hard,True),
            continuum_total_quantized=old["new_engine"])
        steps=list(variants.items())
        transitions=[dict(from_model=a,to_model=b,
            replacement_changed=x["replaced"]!=y["replaced"],no_return_changed=x["no_return"]!=y["no_return"],
            intensity_difference=y["intensity"]-x["intensity"]) for (a,x),(b,y) in zip(steps,steps[1:])]
        out.append(dict(distance_m=r,clean_adc=old["clean_adc"],components=dict(hard_analytic=hard,
            hard_after_original_ADC=hard_code,continuous_soft_peak=soft,continuous_soft_peak_range=soft_range,
            original_soft_peak=old["hahner"]["soft_peak_after_clip"],total_selected_peak=wave["waveform_peak"]),
            counterfactuals=variants,changed_replacement=old["hahner"]["replaced"]!=old["new_engine"]["replaced"],
            controlled_transition_differences=transitions,
            changed_no_return=old["hahner"]["no_return"]!=old["new_engine"]["no_return"],
            normalized_intensity_scale_identical=True,noise_in_these_40_cases=0))
    return dict(status="passed",table=table_id,rows=out,totals=reg["behavior_totals"],
        interpretation="Controlled counterfactual chain, not a unique additive causal decomposition. Continuous full pulse versus LUT range support/grid/time convention, separate versus total-peak competition, hard rounding and soft-output quantization interact. Noise is zero in these 40 cases; production B1 keeps noise10, B2-B4 no extra RNG. Equal default normalization is shared, not an explanation for differences by itself.")


DIST=[0,3,10,20,40,60,80,120,200,1e9]
ADC=[-.5,.5,1.5,4.5,20.5,64.5,128.5,255.5]
TAU=[0,.25,.5,1,2,4,8,1e9]


def cache_diagnosis(path,quantum):
    d=load_arrays(path);validate(d)
    raw=d["original_xyzi"];n=len(raw)
    if "hard_signal" not in d or "soft_signal" not in d:
        return dict(path=str(path),n=n,prequantization_available=False,no_return=int((~d["emitted"]).sum()))
    peak=d["hard_signal"]+d["soft_signal"]
    bypass=d.get("overlap_unknown",np.zeros(n,dtype=bool))|((d["integral"]==0)&(d["soft_signal"]==0))
    ranges=np.linalg.norm(raw[:,:3].astype(float),axis=1)
    signed_range=np.sum(d["weather_xyzi"][:,:3].astype(float)*raw[:,:3],axis=1)/np.maximum(ranges,1e-12)
    zero=(np.clip(np.rint(peak/quantum)*quantum,0,1)==0)&~bypass
    expected=~zero & (signed_range>0)
    if not np.array_equal(expected[~bypass],d["emitted"][~bypass]):raise Blocked("R3A_CACHE_ZERO_ENCODING_REPLAY_FAILED")
    grid=np.column_stack((ranges,np.rint(raw[:,3]*255),d["integral"]))
    counts=np.histogramdd(grid,bins=(DIST,ADC,TAU))[0].astype(int)
    zeros=np.histogramdd(grid[zero],bins=(DIST,ADC,TAU))[0].astype(int)
    return dict(path=str(path),sha256=digest(path),n=n,prequantization_available=True,
        peak_quantiles=quantiles(peak),minimum_peak=float(min(peak)) if n else None,
        below_half_quantum=int(zero.sum()),no_return=int((~d["emitted"]).sum()),zero_clean=int((raw[:,3]==0).sum()),
        zero_encoding_fraction=float(zero.mean()) if n else None,
        no_return_fraction=float((~d["emitted"]).mean()) if n else None,
        nonpositive_range_with_positive_code=int(((signed_range<=0)&~zero&~bypass).sum()),
        clean_intensity_quantiles=quantiles(raw[:,3]),unmodelled_overlap=int(bypass.sum()),
        stratified_counts=counts.tolist(),stratified_zero_codes=zeros.tolist(),
        measured_hardware_threshold_rejections=None,assumed_extra_receiver_rejections=0)


def sensitivity_variants(c):
    out=[dict(name="baseline",factor="none",value=1)]
    for factor,key in (("intensity","intensity_factors"),("soft","soft_factors"),("quantum","quantum_alternatives"),
                       ("response_step_m","waveform_steps_m"),("attenuation_step_m","attenuation_steps_m")):
        out.extend(dict(name=factor+"_"+str(v),factor=factor,value=v) for v in c["sensitivity"][key])
    return out


def trial(field,origin,end,intensity,c,variant):
    cfg=deepcopy(c["receiver"]);factor=variant["factor"];v=variant["value"]
    if factor=="soft":cfg["beta_over_alpha"]*=v
    if factor in {"response_step_m","attenuation_step_m"}:cfg[factor]=v
    if factor=="quantum" and v>0:cfg["intensity_quantum"]=v
    if factor=="intensity":intensity*=v
    r=single_response(field,origin,end,intensity,cfg)
    if factor=="quantum" and v==0 and not r.get("zero_fog_identity") and not r["unidentifiable_overlap"]:
        r["intensity"]=min(r["waveform_peak"],1.)
        r["emitted"]=r["intensity"]>0 and r["range_m"]>0
    return describe(r)


def run(c,old,output,smoke=False):
    reg=regression(dict(old.config,receiver=c["receiver"]),output)
    intensity=intensity_test(c);diff=differences(c,reg)
    atomic_json(output/"R3A_INTENSITY_DEPENDENCE.json",intensity)
    atomic_json(output/"R3A_HAHNER_DIFFERENCES.json",diff)
    paths=sorted((old.r2/"cloud_cache").rglob("response.npz"))
    if not paths:raise Blocked("R3A_REQUIRED_R2_CACHE_MISSING")
    inspected=[cache_diagnosis(p,c["receiver"]["intensity_quantum"]) for p in (paths[:2] if smoke else paths)]
    r1_paths=sorted((old.r1/"responses").rglob("*.npz"))
    if not r1_paths:raise Blocked("R3A_REQUIRED_R1_CACHE_MISSING")
    r1=[cache_diagnosis(p,c["receiver"]["intensity_quantum"]) for p in (r1_paths[:2] if smoke else r1_paths)]
    modes={}
    for mode in ("B0","B1","B2","B3","B4"):
        rows=[r["stats"] for r in old.report["echo_statistics"] if r["mode"]==mode]
        modes[mode]={k:sum(s[k] for s in rows) for k in ("original","original_hard","replaced_by_weather","no_return","unknown")}
    no_return=dict(status="passed",scope="full cache" if not smoke else "SMOKE partial caches; full totals are verified old report values",
        verified_R2_report_totals=modes,R2_cache_audit=inspected,R1_cache_audit=r1,
        distance_edges=DIST,clean_ADC_edges=ADC,optical_depth_edges=TAU,
        numerical_zero_code_boundary=c["receiver"]["intensity_quantum"]/2,
        explanation="Compare actual minimum selected prequantized power with half quantum. Both R1 samples and full R2 report zero losses; this is a data/model result, not a proven realistic failure rate. Zero fog and unidentified overlap bypass the ADC rule.",
        hardware_no_return_validation="NOT_VALIDATED_NO_REAL_REFERENCE",calibrated_receiver_threshold=None)
    atomic_json(output/"R3A_NO_RETURN_DIAGNOSIS.json",no_return)
    # Freeze deterministic Clean point identities before running any sensitivity.
    candidates=[]
    for path in sorted((old.r1/"rays").rglob("rays.npz")):
        a=load_arrays(path);meta=read_json(path.parent/"SAMPLING.json")
        for k,pid in enumerate(a["point_id"]):
            candidates.append((identity([c["seed"],str(pid)]),str(pid),str(path),k,meta["scene"]))
    chosen=sorted(candidates)[:c["smoke"]["real_sensitivity_rays"] if smoke else c["sensitivity"]["real_rays"]]
    if not chosen:raise Blocked("R3A_NO_FIXED_CLEAN_SAMPLE_RAYS")
    variants=sensitivity_variants(c);ids=c["smoke"]["sensitivity_fields"] if smoke else c["sensitivity"]["field_ids"]
    freeze=dict(points=[dict(point_identity=p,path=path,slot=k,scene=s) for _,p,path,k,s in chosen],
        field_ids=ids,variants=variants,label="UNCALIBRATED_SENSITIVITY_ONLY",default_receiver_unchanged=c["receiver"])
    atomic_json(output/"R3A_SENSITIVITY_FREEZE.json",freeze)
    pool=read_json(old.root/"R1_FIELD_POOL_FREEZE.json")["pools"];observations=[];cache={}
    for _,pid,path,k,scene in chosen:
        if path not in cache:cache[path]=load_arrays(path)
        a=cache[path]
        for j in ids:
            f=restore_field(pool[scene]["fields"][j])
            for variant in variants:
                try:
                    r=trial(f,a["origin"],a["world_endpoints"][k],float(a["original_xyzi"][k,3]),c,variant)
                    observations.append(dict(point_identity=pid,scene=scene,field=j,variant=variant["name"],result=r))
                except ValueError as error:
                    if variant["name"]=="baseline":raise Blocked("R3A_BASELINE_REAL_RAY_ERROR") from error
                    observations.append(dict(point_identity=pid,scene=scene,field=j,variant=variant["name"],
                        invalid_sensitivity_case=type(error).__name__,note="Frozen numerical tolerance not relaxed; this variant is not a new default"))
    # Synthetic one-factor checks include the original40 distance/intensity grid.
    synth=[]
    for distance in old.config["regression"]["distances_m"]:
        for adc in old.config["regression"]["intensities_adc"]:
            f=WorldField(kind="constant",mean_alpha=.06)
            for v in variants:synth.append(dict(distance_m=distance,clean_adc=adc,variant=v["name"],result=trial(f,[0]*3,[distance,0,0],adc/255,c,v)))
    atomic_json(output/"R3A_SENSITIVITY.json",dict(label="UNCALIBRATED_SENSITIVITY_ONLY",freeze=freeze,
        real_sampled_observations=observations,synthetic_observations=synth,formal_configuration_changed=False,
        fixed_transmit_power_counterfactual="If only target reflectance varies, soft power can be independent of target intensity; current fixed-gamma normalization instead infers relative transmitted scale from the observed target. This ambiguity is not calibrated by these sensitivity curves."))
    gate=dict(status="PASS_WITH_DISCLOSED_APPROXIMATIONS",numerical_and_Hahner_checks="passed",real_cache_identity="passed",
        receiver_model_version=c["receiver"]["algorithm"],physical_error_found=False,formal_receiver_changed=False,
        allowed_next_stage="UNCALIBRATED_EXPLORATORY_ONLY",REAL_WORLD_VALIDATION_MISSING=True)
    atomic_json(output/"R3A_GATE.json",gate)
    text=("# R3-A 接收器审计\n\nPASS_WITH_DISCLOSED_APPROXIMATIONS\n\n"
        "固定光路上，硬波形和软波形均为原强度I乘固定函数；正I在峰值比和选峰中约掉。量化有效性、零I和未建模近场是例外。"
        "这沿用Hahner的原强度×R²/beta0归一化，并依赖固定gamma及线性强度功率假设。不能声称替换率真实反映材质反射率差异。\n\n"
        "若发射功率固定而强度差异只来自反射率，真实软回波可独立于目标强度；当前假设可能低估低反射率目标相对软回波的脆弱性。真实偏差方向仍依赖未知编码和硬件，不能据此自动换参数。\n\n"
        "R2全部B0-B4报告无返回数均为0。逐缓存给出量化前峰值、强度、距离/光学厚度分层和半量化步长边界。"
        "当前只有零编码及非正报告距离规则，没有经真实接收阈值标定的无返回。\n\n"
        "40条回归的原版19替换/8无返回与新引擎12替换/15无返回逐条给出组件和反事实。"
        "硬衰减严格核验；软表网格/时间支持、分峰与叠加峰及编码规则相互影响，未假装唯一分摊原因。40例噪声均为0，生产B1仍保留noise10。\n\n"
        "敏感性预先冻结，不选最强丢点或最大AP变化参数。默认R2模型没有改变；只允许继续未标定探索。\n")
    atomic_text(output/"R3A_PHYSICS_AUDIT.md",text)
    atomic_text(output/"R3A_RESULTS.md",text+f"\n检查R2缓存 {len(inspected)} 个、R1缓存 {len(r1)} 个；固定真实抽样光路 {len(chosen)} 条。\n")
    return gate

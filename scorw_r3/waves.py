"""One immutable response and RGB serialization per original CAV/path/field."""
from pathlib import Path
import math
import numpy as np

from scorw_r3.core import (Blocked,read_json,digest,identity,array_hash,load_arrays,transaction,save_arrays,atomic_json,check_inputs)
from scorw_repair.generate import cached_response,original_b1
from scorw_repair.reuse import cached as reuse_response
from scorw_repair.response import simulate,summary,validate
from shared_weather_benchmark.dataset_io import read_points
from shared_weather_benchmark.world_field import WorldField
from scorw_p7.field_pool import restore_field


def pool_balance(records,plan):
    """Numerical marginals plus the shared cyclic-shift negative control."""
    rows={}
    for r in records:
        if r["name"].isdigit():rows[(r["cav"],int(r["name"]))]=r
    cavs=plan["cavs"];s=plan["schedules"];per_cav=[]
    for cav in cavs:
        if any(sorted(s[mode][cav])!=list(range(20)) for mode in ("E2","E3","E2_shift")):
            raise Blocked("R3_INCOMPLETE_POOL_MARGINAL")
        diffs={}
        for key in ("mean_tau","emitted_signal_sum","hard_signal_sum"):
            a=[rows[cav,j]["moments"][key] for j in s["E2"][cav]]
            b=[rows[cav,j]["moments"][key] for j in s["E3"][cav]]
            if sorted(a)!=sorted(b):raise Blocked("R3_RECEIVER_MARGINAL_MISMATCH")
            diffs[key]=math.fsum(a)/20-math.fsum(b)/20
        per_cav.append(dict(cav=cav,mean_differences=diffs,status="passed"))
    for j in range(20):
        if len({s["E3"][v][j] for v in cavs})!=len(cavs):raise Blocked("R3_INDEPENDENT_OFFSET_COLLISION")
    def joint(mode):return sorted(tuple(rows[v,s[mode][v][j]]["response_sha256"] for v in cavs) for j in range(20))
    if joint("E2")!=joint("E2_shift"):raise Blocked("R3_SHARED_SHIFT_NEGATIVE_CONTROL_FAILED")
    actual=[]
    for j in range(20):
        modes={}
        for label,mode in (("B3","E2"),("B4","E3")):
            a=[rows[v,s[mode][v][j]] for v in cavs];n=sum(r["stats"]["original"] for r in a)
            modes[label]=dict(original=n,hard_retention=sum(r["stats"]["original_hard"] for r in a)/n,
                no_return=sum(r["stats"]["no_return"] for r in a)/n,
                mean_emitted_signal=sum(r["moments"]["emitted_signal_sum"] for r in a)/n,
                mean_tau=sum(r["moments"]["mean_tau"]*r["stats"]["original"] for r in a)/n)
        actual.append(dict(repeat=j,**modes))
    return dict(status="passed",per_cav=per_cav,shared_joint_shift="passed",per_repeat=actual,
        same_repeat_severity_equality_required=False,
        scope="full receiver-output marginals exactly balanced per CAV over20; per-repeat severity is allowed to differ; same cached arrays and no new scattering RNG")


def build(c,old,selection,coord,out,parent,smoke=False):
    from reference_weather.common import write_pcd
    rows=read_json(coord/"INPUT_MANIFEST.json");check_inputs(rows)
    index={(r["scene"],r["timestamp"],r["cav"]):r for r in rows}
    domains=read_json(coord/"FIELD_DOMAINS.json");ledger=[];balance=[];scene_balance=[]
    physics=dict(old.config,receiver=c["receiver"],field=c["field"],b1=c["b1"])
    for f in selection["frames"]:
        scene,stamp=f["scene"],f["timestamp"]
        for cav in f["selected_cavs"]:
            row=index[(scene,stamp,cav)];raw=read_points(row["pcd"])
            if digest(row["pcd"])!=row["pcd_sha256"] or len(raw)!=row["points"]:raise Blocked("R3_WAVE_CLEAN_INPUT_CHANGED")
            key=[scene,stamp,cav,row["pcd_sha256"]];indices=np.arange(len(raw),dtype=np.int64)
            fingerprints=[]
            for name in ["B0","B1","B2",*range(20)]:
                label="field_%02d"%name if isinstance(name,int) else name
                folder=out/"responses"/scene/stamp/cav/label
                if name in ("B0","B1"):
                    def work(code):
                        d=(simulate(raw,row["pose"],WorldField(kind="constant",mean_alpha=0,bounds=domains[scene]),c["receiver"],key)
                            if name=="B0" else original_b1(raw,physics,key,indices))
                        save_arrays(folder/"response.npz",**d);atomic_json(folder/"STATS.json",summary(d,physics["distance_bins_m"]))
                    transaction(folder,dict(parent=parent,source=row,mode=name,physics=physics),work)
                    data=load_arrays(folder/"response.npz")
                else:
                    field=(WorldField(kind="constant",mean_alpha=c["field"]["mean_alpha"],bounds=domains[scene])
                        if name=="B2" else restore_field(selection["pools"][scene]["fields"][name]))
                    relative=Path(scene)/stamp/cav/("constant" if name=="B2" else "field_%02d"%name)
                    if smoke:
                        data=reuse_response(raw,row["pose"],field,physics,key,indices,folder,parent,old.cache_info(),relative)
                    else:data=cached_response(raw,row["pose"],field,physics,key,indices,folder,parent)
                validate(data)
                expected=np.asarray([identity([*key,int(i)]) for i in indices])
                if not np.array_equal(data["point_identity"],expected):raise Blocked("R3_WAVE_POINT_IDENTITY_CHANGED")
                if isinstance(name,int):fingerprints.append(identity({k:array_hash(data[k]) for k in ("weather_xyzi","emitted","origin","integral")}))
                encoded=out/"encoded"/scene/stamp/cav/label
                def encode(code):
                    write_pcd(encoded/"input.pcd",data["weather_xyzi"][data["emitted"]])
                    actual=read_points(encoded/"input.pcd")
                    if len(actual)!=int(data["emitted"].sum()):raise Blocked("R3_SERIALIZED_SOURCE_COUNT_CHANGED")
                    atomic_json(encoded/"ENCODING.json",dict(response_sha256=digest(folder/"response.npz"),
                        pcd_sha256=digest(encoded/"input.pcd"),point_order="same original emitted row order",XYZI_hash=array_hash(actual)))
                transaction(encoded,dict(parent=parent,response_sha256=digest(folder/"response.npz")),encode)
                ledger.append(dict(scene=scene,timestamp=stamp,cav=cav,name=str(name),
                    response=str(folder/"response.npz"),response_sha256=digest(folder/"response.npz"),
                    pcd=str(encoded/"input.pcd"),pcd_sha256=digest(encoded/"input.pcd"),
                    moments=dict(mean_tau=float(np.mean(data["integral"])),
                        emitted_signal_sum=float(data["weather_xyzi"][data["emitted"],3].astype(float).sum()),
                        hard_signal_sum=float(data["hard_signal"].sum()) if "hard_signal" in data else None),
                    stats=summary(data,physics["distance_bins_m"])))
                print("R3 response ready "+scene+"/"+stamp+"/"+cav+"/"+label,flush=True)
            schedules=selection["assignments"][scene]["schedules"]
            if (sorted(schedules["E2"][cav])!=list(range(20)) or sorted(schedules["E3"][cav])!=list(range(20))
                    or sorted(fingerprints)!=sorted(fingerprints[j] for j in schedules["E3"][cav])):
                raise Blocked("R3_FULL_RESPONSE_MARGINAL_BALANCE_FAILED")
            balance.append(dict(scene=scene,timestamp=stamp,cav=cav,status="passed",fields=20,
                full_response_multiset_sha256=identity(sorted(fingerprints)),extra_scattering_randomness=False))
        current=[r for r in ledger if r["scene"]==scene and r["timestamp"]==stamp]
        scene_plan=selection["assignments"][scene]
        scene_balance.append(dict(scene=scene,timestamp=stamp,**pool_balance(current,scene_plan),
            views=[dict(view=v["name"],**pool_balance(current,dict(scene_plan,cavs=v["cavs"]))) for v in f["views"]]))
    result=dict(status="passed",responses=ledger,balance=balance,
        scope="same response engine for B2-B4; complete per-CAV twenty-field pool; all views reuse identical cached responses")
    atomic_json(out/"R3B_WAVE_INDEX.json",result)
    atomic_json(out/"R3B_BALANCE_CHECK.json",dict(status="passed",scenes=scene_balance,full_pool_size=20))
    check_inputs(rows)
    return result

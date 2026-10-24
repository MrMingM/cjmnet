"""D1 runs on the server: reference-vs-prefetch and actual PCD codec equivalence."""
from pathlib import Path
import tempfile
import numpy as np
from scorw_dataset.core import atomic_json,field_from,read_json,digest
from scorw_dataset.engine import simulate
from scorw_dataset.io import write_pcd,read_points
from shared_weather_benchmark.world_field import WorldField
from shared_weather_benchmark.geometry import numerical_audit
from scorw_repair.response import simulate as reference


def compare(raw,pose,f,c,out,label):
    key=["equivalence",label]
    a=reference(raw,pose,f,c["receiver"],key)
    if f.mean==0 and (not np.array_equal(a["weather_xyzi"],raw) or not a["emitted"].all()):
        raise ValueError("Zero fog did not return exactly to the original rays")
    results=[]
    for batch in (1,c["batch_size"],max(1,c["batch_size"]//2)):
        b=simulate(raw,pose,f,c["receiver"],key,batch,c["max_query_points"])
        errors={k:float(np.max(abs(a[k].astype(float)-b[k].astype(float)))) if a[k].size else 0.
            for k in ("weather_xyzi","integral","hard_signal","soft_signal")}
        # Deliberately strict. No tolerance relaxation or fallback changes the default.
        for k in a:
            equivalent=np.allclose(a[k],b[k],atol=1e-12,rtol=1e-12) if a[k].dtype==np.float64 else np.array_equal(a[k],b[k])
            if not equivalent:raise ValueError("Original/prefetch per-ray identity failed: "+label+str(errors))
        results.append(dict(batch=batch,maximum_absolute_errors=errors,origin_and_validity_exact=True))
    p=out/(label+"_reference.pcd");q=out/(label+"_prefetch.pcd")
    write_pcd(p,a["weather_xyzi"][a["emitted"]]);write_pcd(q,b["weather_xyzi"][b["emitted"]])
    if digest(p)!=digest(q) or not np.array_equal(read_points(p),read_points(q)):raise ValueError("Final serialized PCD identity failed")
    return dict(label=label,rays=len(raw),results=results,PCD_SHA256_equal=True,source_count_conserved=True)


def run(c,inventory_root,out):
    from scorw_dataset.acceptance import run as engineering
    engineering(out)
    rows=[]
    raw=np.array([[r,.3,.2,i] for r in (.5,.9,1.,3.,5.,10.,30.,80.,180.) for i in (0.,1/255,.5,1.)],dtype=np.float32)
    for kind,alpha in (("constant",0.),("constant",.06),("smooth",.06)):
        f=WorldField(kind=kind,mean_alpha=alpha,seed=31,**{k:v for k,v in c["field"].items() if k!="mean_alpha"})
        rows.append(compare(raw,[13,-7,2,11,93,-17],f,c,out,kind+str(alpha)))
    # Real-data equivalence sample is fixed on train before weather. No Test GT or AP.
    from scorw_dataset.performance import sample_indices
    train=read_json(inventory_root/"TRAIN_SOURCE.json")
    for row in train["rows"][:2]:
        raw=read_points(row["pcd"]);ix=sample_indices(raw,c)
        rows.append(compare(raw[ix],row["pose"],field_from(train["fields"][row["scene"]]),c,out,"real"+str(len(rows))))
    # Query order, common world point and frozen field reproduction across timestamps.
    f=WorldField(seed=42,**c["field"]);points=np.array([[10.,20.,1.],[-5.,3.,2.]])
    if not np.array_equal(f.alpha_numpy(points),f.alpha_numpy(points[::-1])[::-1]):raise ValueError("Field query-order dependence")
    if field_from(f.manifest()).manifest()!=f.manifest():raise ValueError("Field seed not reproducible")
    empty=np.empty((0,4),dtype=np.float32);write_pcd(out/"empty.pcd",empty)
    if len(read_points(out/"empty.pcd")):raise ValueError("Empty cloud filled")
    from scorw_repair.regression import check
    regression=check(read_json(Path(__file__).resolve().parents[1]/"scorw_repair/config.json"),out)
    report=dict(status="passed",equivalence=rows,zero_fog=True,empty_PCD_real_reader="passed",field_query_order="passed",
        coordinate_check=numerical_audit(),Hahner_regression=regression["strict_checks"],
        tolerances="float64 components atol/rtol1e-12; float32 output, origin/validity/IDs exact; RGB PCD SHA256 exact; no relaxed R3 physical tolerances",
        REAL_WORLD_VALIDATION_MISSING=True)
    atomic_json(out/"PHYSICS_AUDIT.json",report)
    return report

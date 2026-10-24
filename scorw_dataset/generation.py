"""Streaming per-cloud atomic records, bounded CPU workers and full split QA."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import resource
import numpy as np
from scorw_dataset.core import (atomic_json,digest,identity,transaction,read_json,save_arrays,verify_input,
    field_from,ensure_link,immutable_json)
from scorw_dataset.io import read_points,write_pcd,inventory_layout
from scorw_dataset.engine import simulate


def statistics(d,edges,decoded=None):
    a=d["original_xyzi"];b=d["weather_xyzi"].copy();em=d["emitted"];k=d["origin"]
    if decoded is not None:b[em]=decoded
    r=np.linalg.norm(a[:,:3].astype(float),axis=1);rout=np.linalg.norm(b[:,:3].astype(float),axis=1);err=(rout-r)[em]
    return dict(original=len(a),output=int(em.sum()),hard=int((em&(k==0)).sum()),weather=int((em&(k==1)).sum()),
        unknown=int((em&(k==2)).sum()),lost=int((~em).sum()),high_tau=int((d["integral"]>1).sum()),
        unmodelled_near_field=int(d["overlap_unknown"].sum()),empty_output=not bool(em.any()),
        distance_error=dict(count=len(err),absolute_sum=float(abs(err).sum()),squared_sum=float((err*err).sum()),
            min=float(err.min()) if len(err) else None,max=float(err.max()) if len(err) else None,
            changed=int((abs(err)>2e-5).sum())),
        input_distance_hist=np.histogram(r,edges)[0].tolist(),output_distance_hist=np.histogram(rout[em],edges)[0].tolist(),
        per_distance_scope="cohorts defined by original Clean ray distance; output_distance_hist instead uses final emitted distance",
        per_distance=[dict(low=x,high=y,input=int(((r>=x)&(r<y)).sum()),
            output=int(((r>=x)&(r<y)&em).sum()),hard=int(((r>=x)&(r<y)&d["retained"]).sum()),
            replaced=int(((r>=x)&(r<y)&em&(k==1)).sum()),lost=int(((r>=x)&(r<y)&~em).sum()),
            input_signal=float(a[(r>=x)&(r<y),3].astype(float).sum()),
            output_signal=float(b[(r>=x)&(r<y)&em,3].astype(float).sum())) for x,y in zip(edges,edges[1:])])


def generate_one(c,root,parent,row,manifest):
    root=Path(root);target=root/"fog"/row["relative"]
    record=root/"records"/row["split"]/row["scene"]/"clouds"/row["cav"]/row["timestamp"]
    start=time.monotonic();cached=(record/"COMMIT.json").is_file()
    verify_input(row)
    def work(code):
        raw=read_points(row["pcd"])
        if len(raw)!=row["points"]:raise ValueError("Original point count changed")
        field=field_from(manifest);key=[row["split"],row["scene"],row["timestamp"],row["cav"],row["pcd_sha256"]]
        result=simulate(raw,row["pose"],field,c["receiver"],key,c["batch_size"],c["max_query_points"])
        decoded=write_pcd(target,result["weather_xyzi"][result["emitted"]])
        if len(decoded)!=int(result["emitted"].sum()):raise ValueError("Source/output count mismatch")
        save_arrays(record/"provenance.npz",**{k:result[k] for k in ("original_index","emitted","origin","integral","hard_signal","soft_signal","overlap_unknown")})
        verify_input(row)
        atomic_json(record/"RESULT.json",dict(relative=row["relative"],input_sha256=row["pcd_sha256"],yaml_sha256=row["yaml_sha256"],
            output_sha256=digest(target),seed=manifest["seed"],field_sha256=manifest["sha256"],
            point_identity_rule="SHA256 canonical JSON [split,scene,timestamp,cav,input_pcd_sha256,original_index]; original index maps every sidecar row",
            point_frame_identity=key,output_mapping="PCD rows follow original_index[emitted] in unchanged order",
            origin_codes={0:"original_hard_target_return",1:"simulated_weather_return",2:"unknown"},
            provenance_sha256=digest(record/"provenance.npz"),stats=statistics(result,c["distance_edges_m"],decoded),
            serialization_error=dict(max_XYZ=float(np.max(abs(decoded[:,:3]-result["weather_xyzi"][result["emitted"],:3]))) if len(decoded) else 0.,
                max_intensity=float(np.max(abs(decoded[:,3]-result["weather_xyzi"][result["emitted"],3]))) if len(decoded) else 0.),
            elapsed_seconds=time.monotonic()-start,peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
    transaction(record,dict(parent=parent,source=row,field=manifest,generator="R3-prefetched-v1"),work)
    r=read_json(record/"RESULT.json")
    if digest(target)!=r["output_sha256"]:raise ValueError("Committed Fog PCD changed: "+str(target))
    verify_input(row)
    return dict(r,record=str(record),cache_reused=cached)


def materialize_links(c,root,data,smoke):
    source=Path(c["source_root"]);root=Path(root)
    if not smoke:ensure_link(source/data["split"],root/"clean"/data["split"])
    else:
        for row in data["rows"]:ensure_link(row["pcd"],root/"clean"/row["relative"])
        for m in data["metadata"]:ensure_link(m["path"],root/"clean"/m["relative"])
    for m in data["metadata"]:
        if digest(m["path"])!=m["sha256"]:raise ValueError("Metadata changed before linking")
        ensure_link(m["path"],root/"fog"/m["relative"])


def generate_split(c,root,data,parent,workers,smoke):
    root=Path(root);materialize_links(c,root,data,smoke)
    rows=iter(data["rows"]);pending={};results=[];frames={};start=time.monotonic();processed_points=0;cached_clouds=0
    expected={(s["scene"],t):len(s["cavs"]) for s in data["layout"] for t in s["timestamps"]}
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"))
    def submit():
        try:row=next(rows)
        except StopIteration:return False
        future=pool.submit(generate_one,c,str(root),parent,row,data["fields"][row["scene"]]);pending[future]=row
        return True
    try:
        for _ in range(2*workers):
            if not submit():break
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                row=pending.pop(f)
                try:result=f.result()
                except Exception:
                    atomic_json(root/"audits/LAST_FAILURE.json",dict(split=data["split"],relative=row["relative"],completed=len(results)))
                    raise
                results.append(result);key=(row["scene"],row["timestamp"]);frames[key]=frames.get(key,0)+1
                processed_points+=result["stats"]["original"];cached_clouds+=int(result["cache_reused"])
                if len(results)%10==0 or len(results)==data["pcds"]:
                    count=sum(frames[k]==n for k,n in expected.items() if k in frames)
                    rate=processed_points/max(time.monotonic()-start,1e-6)
                    print(f"{data['split']}: frames {count}/{data['frames']}; PCD {len(results)}/{data['pcds']}; {rate:.1f} input rays/s (including reused clouds); free {shutil.disk_usage(root).free/2**30:.1f}GiB; cached {cached_clouds}; latest {row['relative']}",flush=True)
                if shutil.disk_usage(root).free<c["performance"]["reserve_disk_bytes"]:raise OSError("Disk reserve exhausted; generation stopped without dropping clouds")
                submit()
    finally:
        for f in pending:f.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
    if len(results)!=data["pcds"] or any(frames.get(k)!=n for k,n in expected.items()):raise ValueError("Incomplete split generation")
    return audit_split(c,root,data,parent)


def audit_split(c,root,data,parent):
    root=Path(root);split=data["split"];results=[];frame_records={};cav_records={}
    for mode in ("clean","fog"):
        if inventory_layout(root/mode/split)!=data["layout"]:raise ValueError("Output scene/CAV/frame coverage mismatch: "+mode+"/"+split)
    for row in data["rows"]:
        folder=root/"records"/split/row["scene"]/"clouds"/row["cav"]/row["timestamp"]
        # Re-run only inventory verification for committed records; callback would
        # deliberately fail if a completion record vanished.
        def missing(code):raise ValueError("Missing cloud commit: "+row["relative"])
        transaction(folder,dict(parent=parent,source=row,field=data["fields"][row["scene"]],generator="R3-prefetched-v1"),missing)
        r=read_json(folder/"RESULT.json");p=root/"fog"/row["relative"]
        clean=root/"clean"/row["relative"]
        if clean.resolve(strict=True)!=Path(row["pcd"]).resolve(strict=True):raise ValueError("Clean view reads a different original PCD")
        if p.is_symlink() or digest(p)!=r["output_sha256"]:raise ValueError("Fog must be a real PCD with its recorded hash")
        raw=read_points(p)
        if len(raw)!=r["stats"]["output"] or r["field_sha256"]!=data["fields"][row["scene"]]["sha256"]:raise ValueError("Output/scene field identity mismatch")
        if sum(r["stats"][k] for k in ("hard","weather","unknown","lost"))!=row["points"]:raise ValueError("Origin count conservation failed")
        verify_input(row);results.append(r)
        frame_records.setdefault((row["scene"],row["timestamp"]),[]).append(dict(cav=row["cav"],sha256=r["output_sha256"]))
        cav_records.setdefault((row["scene"],row["cav"]),[]).append(dict(timestamp=row["timestamp"],sha256=r["output_sha256"]))
    expected_meta={m["relative"] for m in data["metadata"]}
    actual_meta={p.relative_to(root/"fog").as_posix() for p in (root/"fog"/split).rglob("*") if p.is_file() and p.suffix.lower()!=".pcd"}
    if actual_meta!=expected_meta:raise ValueError("Fog metadata coverage mismatch")
    for m in data["metadata"]:
        for mode in ("clean","fog"):
            p=root/mode/m["relative"]
            if p.resolve(strict=True)!=Path(m["path"]).resolve(strict=True) or digest(p)!=m["sha256"]:raise ValueError("Metadata/GT/pose identity changed")
    for (scene,stamp),r in frame_records.items():immutable_json(root/"records"/split/scene/"frames"/(stamp+".json"),dict(status="passed",clouds=r,identity=identity(r)))
    for (scene,cav),r in cav_records.items():immutable_json(root/"records"/split/scene/"cavs"/(cav+".json"),dict(status="passed",frames=r,identity=identity(r)))
    for scene in data["layout"]:
        r=[v for v in results if Path(v["relative"]).parts[1]==scene["scene"]]
        immutable_json(root/"records"/split/scene["scene"]/"SCENE_COMPLETE.json",dict(status="passed",pcds=len(r),hashes_identity=identity(r),field_sha256=data["fields"][scene["scene"]]["sha256"]))
    report=dict(status="passed",split=split,scenes=len(data["layout"]),frames=data["frames"],pcds=len(results),rows=results,
        metadata_count=len(data["metadata"]),metadata_identity=identity(data["metadata"]),field_identity=identity(data["fields"]))
    atomic_json(root/"manifests"/(split.upper()+"_MANIFEST.json"),report)
    atomic_json(root/(split.upper()+"_MANIFEST.json"),report)
    return report

"""Frozen train-only timing sample; no weather/AP-based sample selection or tuning."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import resource
import shutil
import time
import numpy as np
from scorw_dataset.core import read_json,atomic_json,field_from,verify_input,immutable_json
from scorw_dataset.io import read_points,write_pcd
from scorw_dataset.engine import simulate
from scorw_repair.response import simulate as reference


def sample_indices(raw,c,bins=None):
    r=np.linalg.norm(raw[:,:3].astype(float),axis=1);out=[]
    per=max(1,c["performance"]["rays_per_cloud"]//(len(c["distance_edges_m"])-1))
    wanted=range(len(c["distance_edges_m"])-1) if bins is None else bins
    for i in wanted:
        a,b=c["distance_edges_m"][i:i+2]
        ix=np.flatnonzero((r>=a)&(r<b));out.extend(ix[np.linspace(0,len(ix)-1,min(per,len(ix)),dtype=int)].tolist() if len(ix) else [])
    return np.asarray(sorted(set(out)),dtype=np.int64)


def coverage_plan(train,picked,global_hist):
    """Choose finite supplemental train sources from Clean histograms, before timing.

    Keep the three structural samples. For each nonempty full-data distance bin
    they miss, use the lexical first train PCD with that bin. One PCD may cover
    several bins; no weather, runtime or detector result participates in selection.
    """
    if train["split"]!="train" or any(r["split"]!="train" for r in train["rows"]):
        raise ValueError("Timing coverage selection is restricted to original train")
    n=len(global_hist);covered=[any(r["distance_hist"][i]>0 for r in picked) for i in range(n)]
    entries=[dict(row=r,kind="structural",bins=list(range(n))) for r in picked]
    additional={};missing=[];required=[i for i,v in enumerate(global_hist) if v>0 and not covered[i]]
    ordered=sorted(train["rows"],key=lambda r:r["relative"])
    for i in required:
        row=next((r for r in ordered if r["distance_hist"][i]>0),None)
        if row is None:missing.append(i);continue
        key=row["relative"]
        if key not in additional:additional[key]=dict(row=row,kind="distance_coverage",bins=[])
        additional[key]["bins"].append(i)
    entries.extend(additional[key] for key in sorted(additional))
    return dict(entries=entries,missing_before_supplement=required,no_train_candidate_bins=missing,
        rule="lexical first original train PCD per missed nonempty full-data bin; fixed point-index spacing within each bin; no resampling")


def timed(job):
    raw,pose,manifest,c=job;start=time.monotonic()
    d=simulate(raw,pose,field_from(manifest),c["receiver"],["performance","fixed"],c["batch_size"],c["max_query_points"])
    return dict(rays=len(raw),seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        output=int(d["emitted"].sum()))


def choose_clouds(train,count):
    descriptors=[]
    for s in train["layout"]:
        positions=np.asarray([r["pose"][:3] for r in train["rows"]
            if r["scene"]==s["scene"] and r["timestamp"]==s["timestamps"][0]])
        descriptors.append(dict(scene=s["scene"],cavs=len(s["cavs"]),initial_origin_span_m=float(np.linalg.norm(np.ptp(positions,axis=0)))))
    descriptors.sort(key=lambda d:(d["cavs"],d["initial_origin_span_m"],d["scene"]))
    ix=sorted(set(np.linspace(0,len(descriptors)-1,min(count,len(descriptors)),dtype=int).tolist()))
    picked=[];structure=[]
    for rank,i in enumerate(ix):
        desc=descriptors[i];rows=sorted((r for r in train["rows"] if r["scene"]==desc["scene"]),key=lambda r:(r["points"],r["relative"]))
        j=round((len(rows)-1)*rank/max(1,len(ix)-1));picked.append(rows[j]);structure.append(desc)
    return picked,structure


def timed_cloud(job):
    row,manifest,c,out=job;verify_input(row);start=time.monotonic();raw=read_points(row["pcd"])
    t=time.monotonic();d=simulate(raw,row["pose"],field_from(manifest),c["receiver"],["performance-full-CAV",row["pcd_sha256"]],c["batch_size"],c["max_query_points"])
    response_seconds=time.monotonic()-t
    write_pcd(Path(out)/"PERFORMANCE_FULL_CAV.pcd",d["weather_xyzi"][d["emitted"]]);verify_input(row)
    return dict(relative=row["relative"],rays=len(raw),response_seconds=response_seconds,wall_seconds=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,output=int(d["emitted"].sum()),
        scope="one complete original CAV cloud, all rays; actual RGB output/readback; timing only, never a dataset completion")


def run(c,inventory_root,out,workers):
    inventories=[read_json(inventory_root/(s+"_SOURCE.json")) for s in ("TRAIN","VALIDATE","TEST")]
    train=inventories[0];picked,structure=choose_clouds(train,c["performance"]["clouds"])
    global_hist=np.sum([r["distance_hist"] for d in inventories for r in d["rows"]],axis=0)
    plan=coverage_plan(train,picked,global_hist)
    frozen=[dict(relative=r["relative"],input_sha256=r["pcd_sha256"],structure=s,
        rule="scene quantiles ordered by CAV count/initial origin span; within-scene low/middle/high point-count quantiles, lexical ties; train only") for r,s in zip(picked,structure)]
    full_row=picked[len(picked)//2]
    jobs=[];io=[];scalar=[];samples=[];sampling_reads=[]
    for k,entry in enumerate(plan["entries"]):
        row=entry["row"]
        verify_input(row);t=time.monotonic();raw=read_points(row["pcd"]);read_seconds=time.monotonic()-t
        ix=sample_indices(raw,c,entry["bins"]);small=raw[ix];manifest=train["fields"][row["scene"]]
        jobs.append((small,row["pose"],manifest,c))
        samples.append(dict(relative=row["relative"],input_sha256=row["pcd_sha256"],kind=entry["kind"],
            requested_bins=entry["bins"],original_indices=ix.tolist(),field_sha256=manifest["sha256"],
            sampled_distance_hist=np.histogram(np.linalg.norm(small[:,:3].astype(float),axis=1),c["distance_edges_m"])[0].tolist()))
        sampling_reads.append(dict(relative=row["relative"],kind=entry["kind"],read_seconds=read_seconds))
        if entry["kind"]=="structural":
            t=time.monotonic();write_pcd(out/("IO_"+str(k)+".pcd"),raw);write_seconds=time.monotonic()-t
            io.append(dict(relative=row["relative"],full_cloud_points=len(raw),sampled_timing_rays=len(ix),
                indices=ix.tolist(),read_seconds=read_seconds,write_readback_seconds=write_seconds,bytes=row["bytes"]))
        else:print("D0 fixed distance timing supplement: "+row["relative"]+"; bins="+str(entry["bins"]),flush=True)
        verify_input(row)
    sampled_hist=np.sum([s["sampled_distance_hist"] for s in samples],axis=0).tolist()
    coverage=dict(rule=plan["rule"],missing_before_supplement=plan["missing_before_supplement"],
        no_train_candidate_bins=plan["no_train_candidate_bins"],global_distance_hist=global_hist.tolist(),
        sampled_distance_hist=sampled_hist,supplemental_clouds=sum(s["kind"]=="distance_coverage" for s in samples))
    # Freeze every source and original index before any reference/optimized weather call.
    immutable_json(out/"PERFORMANCE_SAMPLE_FREEZE.json",dict(clouds=frozen,samples=samples,coverage=coverage,
        full_CAV_benchmark=full_row["relative"],settings=c["performance"]))
    for small,pose,manifest,cfg in jobs:
        t=time.monotonic();reference(small,pose,field_from(manifest),cfg["receiver"],["performance","fixed"])
        scalar.append(dict(rays=len(small),seconds=time.monotonic()-t))
    serial=[timed(job) for job in jobs]
    workload=[jobs[i%len(jobs)] for i in range(max(workers*2,len(jobs)))]
    t=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:parallel=list(pool.map(timed,workload))
    elapsed=time.monotonic()-t;rate=sum(r["rays"] for r in parallel)/elapsed
    print("D0 timing one complete CAV cloud: "+full_row["relative"],flush=True)
    with ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context("spawn")) as pool:
        full=pool.submit(timed_cloud,(full_row,train["fields"][full_row["scene"]],c,str(out))).result()
    total=sum(d["points"] for d in inventories)
    all_bytes=sum(r["bytes"] for d in inventories for r in d["rows"])
    maximum_cloud_points=max(r["points"] for d in inventories for r in d["rows"])
    bin_timings=[];serial_CPU_seconds=0.;uncovered=[]
    for i,(low,high) in enumerate(zip(c["distance_edges_m"],c["distance_edges_m"][1:])):
        measured=[]
        for raw,pose,manifest,cfg in jobs:
            distances=np.linalg.norm(raw[:,:3].astype(float),axis=1);subset=raw[(distances>=low)&(distances<high)]
            if len(subset):measured.append(timed((subset,pose,manifest,cfg)))
        n=sum(r["rays"] for r in measured);dt=sum(r["seconds"] for r in measured)
        cost=dt/n if n else None
        if cost is None and global_hist[i]>0:uncovered.append(i)
        if cost is not None:serial_CPU_seconds+=int(global_hist[i])*cost
        bin_timings.append(dict(low=low,high=high,full_data_rays=int(global_hist[i]),timing_rays=n,seconds_per_ray=cost))
    serial_rate=sum(r["rays"] for r in serial)/max(sum(r["seconds"] for r in serial),1e-9)
    speedup=max(.01,min(workers,rate/max(serial_rate,1e-9)))
    factor=c["performance"]["conservative_cost_factor"]
    full_rate=full["rays"]/max(full["response_seconds"],1e-9)
    cpu_seconds=max(serial_CPU_seconds,total/max(full_rate,1e-9))
    seconds=factor*max(total/max(rate,1e-9),cpu_seconds/speedup)
    disk=total*c["performance"]["disk_bytes_per_input_point"]
    available=int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:")))*1024
    peak=max(full["peak_rss_bytes"],max(r["peak_rss_bytes"] for r in parallel))
    memory=workers*(1024**3+peak*2+c["max_query_points"]*64+maximum_cloud_points*512)
    reasons=[]
    if uncovered:reasons.append("untimed_nonempty_distance_bins:"+str(uncovered))
    if shutil.disk_usage(out).free<disk+c["performance"]["reserve_disk_bytes"]:reasons.append("insufficient_disk")
    if memory+c["performance"]["reserve_ram_bytes"]>available:reasons.append("estimated_parallel_RAM_exceeds_available")
    if seconds/86400>c["performance"]["max_wall_days"]:reasons.append("estimated_wall_time_exceeds_frozen_budget")
    if factor*cpu_seconds/3600>c["performance"]["max_cpu_hours"]:reasons.append("estimated_CPU_hours_exceed_frozen_budget")
    uncovered_ranges=[dict(bin=i,low_m=c["distance_edges_m"][i],high_m=c["distance_edges_m"][i+1],
        full_data_rays=int(global_hist[i]),train_candidate_available=i not in plan["no_train_candidate_bins"]) for i in uncovered]
    report=dict(status="passed" if not uncovered else "TIMING_COVERAGE_INCOMPLETE",frozen_sample=frozen,
        sampling_reads=sampling_reads,coverage=coverage,scalar=scalar,prefetched_serial=serial,parallel=parallel,full_CAV=full,
        workers=workers,parallel_wall_seconds=elapsed,parallel_rays_per_second=rate,IO=io,
        peak_worker_rss_bytes=peak,scope="small stratified train-only timing workload; includes process startup; not a guaranteed full-run rate",
        distance_stratified_timing=bin_timings,observed_parallel_speedup=speedup,
        optimization="batch only identical field queries, each ray calls original single_response; no reduced resolution")
    atomic_json(out/"PERFORMANCE_BENCHMARK.json",report)
    budget=dict(status="passed" if not reasons else "RESOURCE_BLOCKED",total_original_points=total,
        estimated_wall_days=seconds/86400,estimated_CPU_hours=factor*cpu_seconds/3600,estimated_disk_upper_bytes=disk,
        estimated_parallel_host_bytes=memory,available_host_bytes=available,free_disk_bytes=shutil.disk_usage(out).free,
        workers=workers,cost_safety_factor=factor,limits=c["performance"],reasons=reasons,
        distance_coverage_complete=not bool(uncovered),uncovered_distance_ranges=uncovered_ranges,
        estimates_provisional_until_distance_coverage_complete=bool(uncovered),
        estimated_IO_seconds_per_full_read_write_pass=sum(v["read_seconds"]+v["write_readback_seconds"] for v in io)*all_bytes/max(sum(v["bytes"] for v in io),1),
        time_scope="CPU generation wall estimate; repeated hash/reader/voxel QA and GPU validation are additional measured-cost stages; no guaranteed completion ETA",
        recommendations="If blocked, provision disk/CPU/RAM or review a new performance-only version; retain all original clouds, step sizes and scenes")
    atomic_json(out/"GENERATION_BUDGET.json",budget)
    return budget

"""Dataset QA summaries; no severity/AP gate and no claim of real-fog calibration."""
from pathlib import Path
import gzip
import io
import tarfile
import numpy as np
from scorw_dataset.core import SPLITS,read_json,atomic_json,atomic_text,digest,identity,field_from,atomic_bytes
from scorw_dataset.configure import write_views


def aggregate(rows):
    totals={k:sum(r["stats"][k] for r in rows) for k in ("original","output","hard","weather","unknown","lost","high_tau","unmodelled_near_field")}
    totals["empty_clouds"]=sum(r["stats"]["empty_output"] for r in rows)
    errors=[r["stats"]["distance_error"] for r in rows];count=sum(r["count"] for r in errors)
    low=[r["min"] for r in errors if r["count"]];high=[r["max"] for r in errors if r["count"]]
    bins=[]
    for i,b in enumerate(rows[0]["stats"]["per_distance"]):
        bins.append(dict(low=b["low"],high=b["high"],**{k:sum(r["stats"]["per_distance"][i][k] for r in rows)
            for k in ("input","output","hard","replaced","lost","input_signal","output_signal")}))
    return dict(totals=totals,distance_error=dict(count=count,min=min(low) if low else None,max=max(high) if high else None,
        changed=sum(r["changed"] for r in errors),mean_absolute=sum(r["absolute_sum"] for r in errors)/count if count else None,
        RMSE=float(np.sqrt(sum(r["squared_sum"] for r in errors)/count)) if count else None),
        input_distance_hist=np.sum([r["stats"]["input_distance_hist"] for r in rows],axis=0).tolist(),
        output_distance_hist=np.sum([r["stats"]["output_distance_hist"] for r in rows],axis=0).tolist(),per_distance=bins,
        per_distance_scope="cohorts defined by original Clean ray distance; output_distance_hist uses final emitted distance")


def run(c,root,smoke,out=None):
    root=Path(root);out=Path(out) if out is not None else root;summary={};scenes=[]
    for split in SPLITS:
        m=read_json(root/(split.upper()+"_MANIFEST.json"));selection=read_json(root/"manifests"/(split.upper()+"_SELECTION.json"))
        if m["status"]!="passed":raise ValueError("Incomplete split QA")
        summary[split]=dict(scenes=m["scenes"],frames=m["frames"],pcds=m["pcds"],**aggregate(m["rows"]))
        for scene in selection["layout"]:
            rows=[r for r in m["rows"] if Path(r["relative"]).parts[1]==scene["scene"]]
            manifest=selection["fields"][scene["scene"]];field=field_from(manifest);b=manifest["bounds"]
            grid=np.asarray([[x,y,z] for x in np.linspace(b[0],b[3],5) for y in np.linspace(b[1],b[4],5) for z in np.linspace(b[2],b[5],5)])
            alpha=field.alpha_numpy(grid)
            scenes.append(dict(split=split,scene=scene["scene"],frames=len(scene["timestamps"]),cavs=scene["cavs"],
                **aggregate(rows),field_sha256=manifest["sha256"],seed=manifest["seed"],
                alpha_grid=dict(samples=125,min=float(alpha.min()),max=float(alpha.max()),mean=float(alpha.mean()),scope="fixed5x5x5 world grid; not an exact volume mean")))
    compatibility=read_json(root/"COMPATIBILITY_AUDIT.json");physics=read_json(root/"PHYSICS_AUDIT.json")
    if compatibility["status"]!="passed" or physics["status"]!="passed":raise ValueError("Dataset compatibility/physics gate not passed")
    if not smoke and any({k:summary[s][k] for k in ("scenes","frames")}!=c["expected"][s] for s in SPLITS):raise ValueError("Final full-size count mismatch")
    status="SMOKE_COMPLETE" if smoke else "DATASET_COMPLETE"
    qa=dict(status=status,splits=summary,per_scene=scenes,all_CAVs_all_timestamps=True,all_output_PCD_hashes_and_reader_checks="passed",
        metadata_GT_pose_bytes="same source links and SHA256",single_shared_scene_field_all_times=True,split_isolation=True,
        REAL_WORLD_VALIDATION_MISSING=True,zero_no_return_is_allowed=True)
    atomic_json(out/"DATASET_QA.json",qa);write_views(c,root)
    atomic_text(out/"FINAL_RESULTS.md","# SCOR-W-Fog v1\n\n"+status+"\n\n"+
        "\n".join(f"- {s}: {v['scenes']} scenes, {v['frames']} cooperative frames, {v['pcds']} PCDs; points {v['totals']['original']} → {v['totals']['output']}; replaced {v['totals']['weather']}; lost {v['totals']['lost']}; empty {v['totals']['empty_clouds']}" for s,v in summary.items())+
        "\n\nD6只评测Validate；AP下降不构成生成成功门槛。完整逐PCD来源、距离与强度统计见各split manifest及records。\n\nREAL_WORLD_VALIDATION_MISSING\n")
    readme=(Path(__file__).resolve().parent/"README.md").read_text("utf-8")
    atomic_text(out/"README.md",readme+"\n\n实际生成规模：\n\n"+"\n".join(f"- {s}: {v}" for s,v in summary.items())+"\n\n状态："+status+"\n")
    return qa


def archive(root):
    root=Path(root)
    reports=[p for p in root.iterdir() if p.is_file() and p.suffix in {".json",".md"} and p.name not in {"REPORT_MANIFEST.json","PUBLICATION_IN_PROGRESS.json"}]
    rows={p.name:dict(sha256=digest(p),bytes=p.stat().st_size,in_small_archive=p.stat().st_size<8*1024**2 and "_MANIFEST" not in p.name) for p in reports}
    atomic_json(root/"REPORT_MANIFEST.json",rows)
    stream=io.BytesIO()
    with gzip.GzipFile(fileobj=stream,mode="wb",mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode="w") as tar:
            for p in [root/"REPORT_MANIFEST.json",*[p for p in reports if rows[p.name]["in_small_archive"]]]:
                payload=p.read_bytes();info=tarfile.TarInfo(p.name);info.size=len(payload);info.mtime=0
                tar.addfile(info,io.BytesIO(payload))
    atomic_bytes(root/"SMALL_REPORTS.tar.gz",stream.getvalue())

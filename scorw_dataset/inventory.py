"""Complete physical inventory and whole-scene bounds using XYZ/poses only."""
from pathlib import Path
import numpy as np
from scorw_dataset.core import (SPLITS,source_path,digest,identity,seed_for,atomic_json,read_json)
from scorw_dataset.io import inventory_layout,read_points,pose_only
from shared_weather_benchmark.geometry import pose_matrix
from shared_weather_benchmark.world_field import WorldField


def run(c,out):
    root=Path(c["source_root"]).resolve(strict=True);layouts={};errors=[]
    for split in SPLITS:
        folder=source_path(root,split)
        try:layout=inventory_layout(folder)
        except Exception as error:
            atomic_json(out/"SOURCE_LAYOUT.json",dict(layouts=layouts,failed_split=split,error=str(error)))
            raise
        layouts[split]=layout
        counts=dict(scenes=len(layout),frames=sum(len(s["timestamps"]) for s in layout))
        if counts!=c["expected"][split]:errors.append(dict(split=split,expected=c["expected"][split],actual=counts))
    seen={}
    for split,layout in layouts.items():
        for s in layout:
            if s["scene"] in seen:errors.append(dict(scene=s["scene"],splits=[seen[s["scene"]],split],reason="scene split overlap"))
            seen[s["scene"]]=split
    atomic_json(out/"SOURCE_LAYOUT.json",dict(layouts=layouts,count_errors=errors))
    if errors:raise ValueError("SOURCE_COUNTS_OR_SPLIT_IDENTITY_MISMATCH: see SOURCE_LAYOUT.json; expected counts unchanged")
    result={};seeds=[]
    for split,layout in layouts.items():
        rows=[];metadata=[];fields={}
        for scene in layout:
            name=scene["scene"];low=np.full(3,np.inf);high=-low
            for cav in scene["cavs"]:
                directory=root/split/name/cav
                for stamp in scene["timestamps"]:
                    rel=Path(split)/name/cav/(stamp+".pcd");p=source_path(root,rel);y=source_path(root,rel.with_suffix(".yaml"))
                    hashes=(digest(p),digest(y));pose=pose_only(y);raw=read_points(p);m=np.asarray(pose_matrix(pose))
                    if max(map(abs,pose[3:]))>360:raise ValueError("Pose angle outside frozen degree convention")
                    from opencood.utils.transformation_utils import x_to_world
                    if not np.allclose(m,x_to_world(pose),rtol=0,atol=1e-12):raise ValueError("Native/independent world transform differs")
                    if len(raw) and (np.linalg.norm(raw[:,:3],axis=1)<=1e-5).any():raise ValueError("Zero-range Clean return: "+str(p))
                    if not np.allclose(raw[:,3]*255,np.rint(raw[:,3]*255),atol=2e-5):raise ValueError("Unexpected original intensity encoding")
                    xyz=raw[:,:3].astype(float)@m[:3,:3].T+m[:3,3]
                    if len(raw) and not np.allclose((xyz[:128]-m[:3,3])@m[:3,:3],raw[:128,:3],atol=1e-8,rtol=0):raise ValueError("Actual point local/world roundtrip failed")
                    low=np.minimum(low,m[:3,3]);high=np.maximum(high,m[:3,3])
                    if len(xyz):low=np.minimum(low,xyz.min(axis=0));high=np.maximum(high,xyz.max(axis=0))
                    if hashes!=(digest(p),digest(y)):raise ValueError("Source changed during inventory")
                    rows.append(dict(split=split,scene=name,cav=cav,timestamp=stamp,relative=rel.as_posix(),pcd=str(p),yaml=str(y),
                        pcd_sha256=hashes[0],yaml_sha256=hashes[1],points=len(raw),bytes=p.stat().st_size,pose=pose,
                        distance_hist=np.histogram(np.linalg.norm(raw[:,:3].astype(float),axis=1),c["distance_edges_m"])[0].tolist()))
            margin=c["field_margin_m"];bounds=(low-margin).tolist()+(high+margin).tolist()
            seed=seed_for(c["master_seed"],"scorw-fog-scene-v1",split,name);seeds.append(seed)
            fields[name]=WorldField(kind="smooth",seed=seed,bounds=bounds,**c["field"]).manifest()
            print(f"D0 inventory {split}/{name}: {len(scene['timestamps'])} frames, {len(scene['cavs'])} CAVs",flush=True)
        all_pcd={p.relative_to(root).as_posix() for p in (root/split).rglob("*") if p.is_file() and p.suffix.lower()==".pcd"}
        if all_pcd!={r["relative"] for r in rows}:raise ValueError("Unaccounted original PCD outside canonical CAV layout")
        for p in sorted((root/split).rglob("*")):
            if not p.is_file() or p.suffix.lower()==".pcd":continue
            resolved=source_path(root,p.relative_to(root))
            metadata.append(dict(relative=p.relative_to(root).as_posix(),path=str(resolved),sha256=digest(resolved),bytes=resolved.stat().st_size))
        data=dict(split=split,layout=layout,rows=rows,metadata=metadata,fields=fields,
            points=sum(r["points"] for r in rows),frames=sum(len(s["timestamps"]) for s in layout),pcds=len(rows))
        atomic_json(out/(split.upper()+"_SOURCE.json"),data)
        result[split]=dict(scenes=len(layout),frames=data["frames"],pcds=len(rows),points=data["points"],
            inventory_sha256=digest(out/(split.upper()+"_SOURCE.json")),inventory=str(out/(split.upper()+"_SOURCE.json")))
    if len(seeds)!=len(set(seeds)):raise ValueError("Scene seed collision")
    report=dict(status="passed",splits=result,identity=identity(result),scene_split_overlap=False,
        GT_used_for_generation=False,test_GT_loaded=False,field_bounds="entire original scene XYZ plus sensor origins and2m margin; convex box covers every segment")
    atomic_json(out/"SOURCE_INVENTORY.json",report)
    return report


def selected(data,c,smoke):
    if not smoke:return data
    layout=[]
    for s in data["layout"][:c["smoke"]["scenes_per_split"]]:
        layout.append(dict(s,timestamps=s["timestamps"][:c["smoke"]["frames_per_scene"]]))
    keys={(s["scene"],t) for s in layout for t in s["timestamps"]}
    rows=[r for r in data["rows"] if (r["scene"],r["timestamp"]) in keys]
    relatives={str(Path(r["relative"]).with_suffix(".yaml")) for r in rows}
    # All auxiliary metadata from selected scene/CAV directories remains available;
    # unselected primary frame YAMLs are excluded so loaders cannot discover absent PCDs.
    def keep(m):
        p=Path(m["relative"])
        if len(p.parts)>2 and p.parts[1] not in {s["scene"] for s in layout}:return False
        primary=p.suffix==".yaml" and len(p.parts)==4 and p.stem.isdigit()
        return not primary or p.as_posix() in relatives
    return dict(data,layout=layout,rows=rows,metadata=[m for m in data["metadata"] if keep(m)],
        frames=sum(len(s["timestamps"]) for s in layout),pcds=len(rows),points=sum(r["points"] for r in rows),
        fields={s["scene"]:data["fields"][s["scene"]] for s in layout},smoke=True)


def verify_source_tree(c,inventory_root):
    root=Path(c["source_root"])
    for split in SPLITS:
        d=read_json(inventory_root/(split.upper()+"_SOURCE.json"))
        if inventory_layout(root/split)!=d["layout"]:raise ValueError("Original split layout changed: "+split)
        actual_pcd={p.relative_to(root).as_posix() for p in (root/split).rglob("*")
            if p.is_file() and p.suffix.lower()==".pcd"}
        if actual_pcd!={r["relative"] for r in d["rows"]}:raise ValueError("Original PCD inventory changed: "+split)
        for r in d["rows"]:
            from scorw_dataset.core import verify_input
            verify_input(r)
        actual={p.relative_to(root).as_posix() for p in (root/split).rglob("*")
            if p.is_file() and p.suffix.lower()!=".pcd"}
        if actual!={m["relative"] for m in d["metadata"]}:raise ValueError("Original metadata inventory changed")
        for m in d["metadata"]:
            if digest(m["path"])!=m["sha256"]:raise ValueError("Original metadata changed: "+m["path"])

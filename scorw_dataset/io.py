"""Independent full train/validate/test PCD and pose IO; never inspect Test GT."""
from pathlib import Path
import numpy as np
from scorw_dataset.core import digest


def pose_only(path):
    import yaml
    from shared_weather_benchmark.geometry import pose_matrix
    # Compose nodes without constructing arbitrary Python tags or reading vehicle objects.
    node=yaml.compose(Path(path).read_text("utf-8"))
    if not isinstance(node,yaml.MappingNode):raise ValueError("Frame YAML must be a mapping")
    found=[v for k,v in node.value if k.value=="lidar_pose"]
    if len(found)!=1 or not isinstance(found[0],yaml.SequenceNode) or len(found[0].value)!=6:
        raise ValueError("Unsupported lidar_pose encoding: "+str(path))
    pose=[float(v.value) for v in found[0].value];pose_matrix(pose)
    return pose


def header(path):
    fields=None;count=None
    with Path(path).open("rb") as f:
        for _ in range(100):
            s=f.readline().decode("ascii").strip()
            if s.startswith("FIELDS "):fields=s.split()[1:]
            if s.startswith("POINTS "):count=int(s.split()[1])
            if s.startswith("DATA "):break
        else:raise ValueError("PCD header has no DATA")
    if fields!=["x","y","z","rgb"] or count is None or count<0:raise ValueError("Expected original x y z rgb PCD: "+str(path))
    return count


def read_points(path):
    from opencood.utils.pcd_utils import pcd_to_np
    n=header(path)
    # Exercise the real OpenCOOD reader for empty clouds too. Never insert fake points.
    a=pcd_to_np(str(path))
    if a.shape!=(n,4) or not np.isfinite(a).all() or (a[:,3]<0).any() or (a[:,3]>1).any():
        raise ValueError("OpenCOOD PCD decoding/schema failure: "+str(path))
    return a


def write_pcd(path,points):
    # Serialization has no split guard and is exactly the pinned historical RGB codec.
    from reference_weather.common import write_pcd as writer
    writer(path,points)
    a=read_points(path)
    if len(a)!=len(points):raise ValueError("PCD roundtrip source count changed")
    return a


def inventory_layout(root):
    """No scene/CAV/range/point subsampling. All per-CAV frame sets must agree."""
    scenes=[]
    for scene in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        if scene.is_symlink():raise ValueError("Aliased source scene directory: "+str(scene))
        cavs=sorted(p.name for p in scene.iterdir() if p.is_dir())
        if not cavs or any(not c.lstrip("-").isdigit() for c in cavs):raise ValueError("Invalid CAV inventory: "+str(scene))
        if int(cavs[0])<0:cavs=cavs[1:]+cavs[:1]
        stamps=None
        for cav in cavs:
            folder=scene/cav
            if folder.is_symlink():raise ValueError("Aliased source CAV directory: "+str(folder))
            own=sorted(p.stem for p in folder.glob("*.yaml") if "additional" not in p.name)
            pcds=sorted(p.stem for p in folder.glob("*.pcd"))
            if not own or any(not t.isdigit() for t in own) or len(own)!=len(set(own)) or pcds!=own:
                raise ValueError("PCD/YAML identity mismatch: "+str(folder))
            if stamps is not None and stamps!=own:raise ValueError("CAV timestamp inventory differs: "+str(folder))
            stamps=own
        scenes.append(dict(scene=scene.name,cavs=cavs,timestamps=stamps))
    return scenes

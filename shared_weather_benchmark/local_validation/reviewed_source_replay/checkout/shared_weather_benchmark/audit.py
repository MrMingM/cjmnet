"""Mandatory real-coordinate/GT gate before any weather generation."""
from __future__ import annotations

import math

from shared_weather_benchmark.geometry import pose_matrix, inverse, box_corners, object_matrix
from shared_weather_benchmark.dataset_io import read_metadata, read_points, source_paths
from shared_weather_benchmark.provenance import atomic_json, atomic_text, digest, identity


def world_xyz(points, pose):
    import numpy as np
    m = np.asarray(pose_matrix(pose))
    return points[:,:3].astype(np.float64) @ m[:3,:3].T + m[:3,3]


def canonical_gt(metadata, tolerance):
    import numpy as np
    merged = {}
    seen = {}
    for cav, data in metadata.items():
        ids = [str(k) for k in data["vehicles"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Ambiguous numeric/string object IDs in one CAV")
        for key, obj in data["vehicles"].items():
            key = str(key)
            corners = np.asarray(box_corners(obj))
            if key in merged:
                # Nearest-corner set distance permits 180-degree symmetric boxes.
                delta = np.linalg.norm(corners[:,None,:]-np.asarray(box_corners(merged[key]))[None,:,:],axis=2)
                if max(delta.min(axis=0).max(),delta.min(axis=1).max()) > tolerance:
                    raise ValueError("Cross-CAV GT geometry mismatch for ID " + key)
            else:
                merged[key] = {k: [float(x) for x in obj[k]] for k in ("location","center","angle","extent")}
            seen.setdefault(key,[]).append(cav)
    if not merged:
        raise ValueError("No clean GT objects")
    return merged, seen


def real_coordinate_audit(config, selection, root, output):
    import numpy as np
    from opencood.utils.transformation_utils import x_to_world, x1_to_x2
    inputs, gt_frames, bounds, checks = [], [], {}, []
    previous = {}
    previous_gt = {}
    maxerr = 0.
    matched_ids = 0
    for frame in selection["frames"]:
        metadata = {}
        for cav in frame["cavs"]:
            pcd,yaml = source_paths(root,frame,cav)
            before = (digest(pcd),digest(yaml))
            data = read_metadata(yaml,root)
            pose = [float(v) for v in data["lidar_pose"]]
            if max(map(abs,pose[3:])) > config["audit"]["max_angular_abs_deg"]:
                raise ValueError("Pose angles inconsistent with degree convention")
            m = np.asarray(pose_matrix(pose))
            if not np.allclose(m,x_to_world(pose),rtol=0,atol=1e-12):
                raise ValueError("Independent/source transform mismatch")
            if not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-12) or abs(np.linalg.det(m[:3,:3])-1) > 1e-12:
                raise ValueError("Nonrigid pose")
            raw = read_points(pcd)
            if not len(raw) or (np.linalg.norm(raw[:,:3],axis=1) <= 1e-5).any():
                raise ValueError("Empty/zero-range clear cloud; stop rather than silently discard source IDs")
            if not np.allclose(raw[:,3]*255,np.rint(raw[:,3]*255),atol=2e-5):
                raise ValueError("RGB intensity is not 8-bit quantized as assumed")
            xyz = world_xyz(raw,pose)
            back = (xyz-m[:3,3])@m[:3,:3]
            error = float(np.max(np.abs(back-raw[:,:3])))
            maxerr = max(maxerr,error)
            if error > config["audit"]["coordinate_tolerance_m"]:
                raise ValueError("Local/world roundtrip failed")
            key = (frame["scene"],cav)
            if key in previous:
                prev = previous[key]
                yaw_step = abs((pose[4]-prev[4]+180)%360-180)
                if math.dist(pose[:3],prev[:3]) > config["audit"]["max_pose_step_m"] or yaw_step > config["audit"]["max_yaw_step_deg"]:
                    raise ValueError("CAV trajectory identity/continuity gate failed")
            previous[key] = pose
            b = bounds.setdefault(frame["scene"], [list(m[:3,3]),list(m[:3,3])])
            b[0] = np.minimum(b[0],np.minimum(xyz.min(axis=0),m[:3,3])).tolist()
            b[1] = np.maximum(b[1],np.maximum(xyz.max(axis=0),m[:3,3])).tolist()
            metadata[cav] = data
            if before != (digest(pcd),digest(yaml)):
                raise ValueError("Input changed during audit")
            inputs.append(dict(scene=frame["scene"],timestamp=frame["timestamp"],cav=cav,
                               pcd=str(pcd),yaml=str(yaml),pcd_sha256=before[0],yaml_sha256=before[1],
                               points=len(raw),pose=pose,partition=frame["partition"]))
        gt, seen = canonical_gt(metadata,config["audit"]["gt_pair_tolerance_m"])
        for oid,obj in gt.items():
            center = np.asarray(object_matrix(obj))[:3,3]
            key = (frame["scene"],oid)
            if key in previous_gt and math.dist(center,previous_gt[key]) > config["audit"]["max_object_step_m"]:
                raise ValueError("Object ID temporal position jump: " + oid)
        previous_gt = {**{k:v for k,v in previous_gt.items() if k[0] != frame["scene"]},
                       **{(frame["scene"],oid):np.asarray(object_matrix(obj))[:3,3] for oid,obj in gt.items()}}
        matched_ids += sum(len(v)>1 for v in seen.values())
        origins = {c:np.asarray(metadata[c]["lidar_pose"][:3],dtype=float) for c in frame["cavs"]}
        active = [c for c in frame["cavs"] if np.linalg.norm((origins[c]-origins[frame["ego"]])[:2]) <= config["audit"]["communication_range_m"]]
        if len(active) < 2:
            raise ValueError("Selected frame has fewer than two CAVs inside fixed communication radius")
        # Freeze a clear world-GT universe, independent of weather or point visibility.
        scoped = {k:v for k,v in gt.items() if k != frame["ego"] and min(
            np.linalg.norm(np.asarray(object_matrix(v))[:3,3]-origins[c]) for c in active
        ) <= config["audit"]["max_object_range_m"]}
        if not scoped:
            raise ValueError("Empty clean GT scope")
        for cav in frame["cavs"]:
            m = np.asarray(pose_matrix(metadata[cav]["lidar_pose"]))
            inv = np.asarray(inverse(m.tolist()))
            for obj in gt.values():
                corners = np.asarray(box_corners(obj))
                local = corners@inv[:3,:3].T+inv[:3,3]
                if not np.allclose(local@m[:3,:3].T+m[:3,3],corners,atol=1e-8,rtol=0):
                    raise ValueError("GT local/world recovery failed")
            ego_pose = metadata[frame["ego"]]["lidar_pose"]
            relative = np.asarray(inverse(pose_matrix(ego_pose)))@m
            if not np.allclose(relative,x1_to_x2(metadata[cav]["lidar_pose"],ego_pose),atol=1e-10,rtol=0):
                raise ValueError("Cross-CAV source transform mismatch")
        gt_frames.append(dict(**frame,active_cavs=active,objects=scoped,gt_sources=seen,
                              clean_gt_sha256=identity(scoped)))
        checks.append(dict(scene=frame["scene"],timestamp=frame["timestamp"],cavs=len(frame["cavs"]),objects=len(scoped)))
    if matched_ids == 0:
        raise ValueError("No cross-CAV common object IDs to validate pairing")
    margin = config["audit"]["field_margin_m"]
    domains = {s: [x-margin for x in b[0]]+[x+margin for x in b[1]] for s,b in bounds.items()}
    report = dict(status="passed",input_identity=identity(inputs),gt_identity=identity(gt_frames),
                  roundtrip_max_m=maxerr,shared_object_instances=matched_ids,frames=checks,
                  unrecoverable=["physical beam/channel identity","per-beam emission times","missing clear returns","waveforms","calibrated detection threshold"],
                  object_center_convention="OpenCOOD location+center before pose rotation; inherited labeling convention",
                  timing="mocked integer timestamp continuity only; seconds not inferred",
                  vehicle_identity="scene/CAV directory identity plus pose/GT consistency; independent sensor serial unavailable",
                  world="CARLA numeric world; yaw +90 maps local +X to world +Y; visual handedness not reinterpreted")
    atomic_json(output/"INPUT_MANIFEST.json",inputs)
    atomic_json(output/"CLEAN_GT.json",gt_frames)
    atomic_json(output/"REAL_COORDINATE_AUDIT.json",report)
    atomic_json(output/"FIELD_DOMAINS.json",domains)
    atomic_text(output/"COORDINATE_AUDIT.md", "# 坐标审计\n\n真实数据检查通过。\n\n"+
                f"检查 {len(checks)} 个协作帧；正反变换最大误差 {maxerr:.6g} m；共享目标实例 {matched_ids}。\n\n"+
                "逐CAV读取原始PCD，不做shuffle。原始强度为RGB红通道/255。位姿顺序为[x,y,z,roll,yaw,pitch]，角度单位度；Rz(yaw) Ry(-pitch) Rx(-roll)。\n\n"+
                "检查同帧各CAV时间戳集合、恒定索引步长、位姿连续性、跨CAV同ID框角点、每个CAV的GT正反投影及原始文件哈希。\n\n"+
                "GT沿用OpenCOOD的location+center约定。框内Clean回波仅是几何关联代理；没有逐点语义真值，无法证明每个点属于车辆表面。\n\n"+
                "实际扫描束编号、逐束发射时间、物理时间间隔和接收阈值不可恢复。结果不能验证CARLA标签约定与真实传感器标定相同。\n")
    return report

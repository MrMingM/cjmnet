"""Freeze official STF split samples and compare unpaired marginal summaries only."""
from pathlib import Path
import json
import numpy as np

from scorw_repair.core import Blocked, atomic_json, atomic_text, identity, digest


def sample_freeze(config, smoke):
    root = Path(config["stf_code"])/"splits"
    result, sources = [], {}
    count = config["real_samples"]["smoke_per_category" if smoke else "full_per_category"]
    for category, names in config["real_samples"]["splits"].items():
        candidates = set()
        for name in names:
            path = root/name
            sources[name] = digest(path)
            for line in path.read_text("utf-8").splitlines():
                parts = line.strip().split(",")
                if len(parts) != 2 or not parts[1].isdigit():
                    raise Blocked("INVALID_OFFICIAL_SPLIT: " + name)
                candidates.add((parts[0], parts[1]))
        # One sample per recording first, then fill deterministically; never weather-result selection.
        order = sorted(candidates,key=lambda p:(identity([config["seed"],category,*p]),p))
        seen, first, remainder = set(), [], []
        for recording, index in order:
            target = remainder if recording in seen else first
            target.append((recording,index)); seen.add(recording)
        for recording,index in (first+remainder)[:count]:
            result.append(dict(category=category,recording=recording,index=index,sample=recording+"_"+index,
                               label_basis="official pre-defined weather split; not a paired clean/fog capture"))
    if len({r["sample"] for r in result}) != len(result):
        raise Blocked("OFFICIAL_SPLIT_CATEGORY_OVERLAP")
    return dict(samples=result,split_hashes=sources,smoke=smoke,rule="fixed hash order, one recording first",
                identity=identity(dict(samples=result,split_hashes=sources)))


def point_stats(points, edges):
    if points.ndim != 2 or points.shape[1] < 4 or not np.isfinite(points).all():
        raise Blocked("STF_POINT_FORMAT_NONFINITE")
    r = np.linalg.norm(points[:,:3],axis=1)
    return dict(points=len(points),range_bins=np.histogram(r,edges)[0].tolist(),
        range_quantiles=np.quantile(r,[0,.1,.5,.9,1]).tolist() if len(r) else None,
        intensity_quantiles=np.quantile(points[:,3],[0,.1,.5,.9,1]).tolist() if len(r) else None,
        near_0_3m_count=int(((r>0)&(r<3)).sum()),zero_range_count=int((r==0).sum()),
        note="near-range population is not labelled as fog/clutter without semantic evidence",
        bins=[dict(low=a,high=b,points=int(((r>=a)&(r<b)).sum()),
            mean_intensity=float(points[(r>=a)&(r<b),3].mean()) if ((r>=a)&(r<b)).any() else None)
            for a,b in zip(edges,edges[1:])])


def sensor_transform_audit(tree):
    if not isinstance(tree,list):
        raise Blocked("STF_TF_TREE_SCHEMA")
    matches=[r for r in tree if r.get("child_frame_id")=="lidar_hdl64_s3_roof"]
    if len(matches)!=1 or matches[0].get("header",{}).get("frame_id")!="body":
        raise Blocked("STF_LIDAR_FRAME_PAIRING")
    transform=matches[0]["transform"]
    q=np.asarray([transform["rotation"][k] for k in ("x","y","z","w")],dtype=float)
    t=np.asarray([transform["translation"][k] for k in ("x","y","z")],dtype=float)
    if not np.isfinite(q).all() or not np.isfinite(t).all() or abs(np.linalg.norm(q)-1)>1e-6:
        raise Blocked("STF_TF_QUATERNION_OR_TRANSLATION_INVALID")
    x,y,z,w=q
    rotation=np.asarray([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    probe=np.asarray([[0,0,0],[10,2,3],[-5,1,4]],dtype=float)
    back=((probe@rotation.T+t)-t)@rotation
    error=float(np.max(np.abs(back-probe)))
    if error>1e-5 or abs(np.linalg.det(rotation)-1)>1e-5:
        raise Blocked("STF_TF_NUMERICAL_ROUNDTRIP_FAILED")
    return dict(parent="body",child="lidar_hdl64_s3_roof",quaternion_order="xyzw",roundtrip_max_m=error,
                determinant=float(np.linalg.det(rotation)),role="metadata coordinate consistency; not receiver calibration")


def audit(config, frozen, output):
    root = Path(config["reference_root"])/"selected"
    records = []
    for row in frozen["samples"]:
        path = root/"lidar_hdl64_last"/(row["sample"]+".bin")
        if not path.is_file():
            raise Blocked("BLOCKED_STF_SAMPLE_MISSING: " + path.name)
        if path.stat().st_size % 20:
            raise Blocked("STF_BIN_NOT_FLOAT32_X5: " + path.name)
        points = np.fromfile(path,dtype="<f4").reshape(-1,5)
        if not len(points):
            raise Blocked("STF_EMPTY_CLOUD: " + path.name)
        extra = {}
        for folder in ("weather_station","labeltool_labels"):
            meta = root/folder/(row["sample"]+".json")
            if meta.is_file():
                value = json.loads(meta.read_text("utf-8"))
                if folder == "labeltool_labels":
                    key = "clear" if row["category"] == "clear" else row["category"]
                    if value.get("weather",{}).get(key) is not True or value.get("bad_sensor") is True:
                        raise Blocked("STF_WEATHER_LABEL_CONFLICT: " + path.name)
                # Keep dataset payload private. Public reports retain schema/identity and validation only.
                extra[folder] = dict(sha256=digest(meta),status="available",
                    schema_keys=[k for k in value if isinstance(k,str) and k.replace("_","").isalnum()][:40],
                    unit_interpretation="no visibility/extinction conversion without a verified field/unit definition")
            else:
                extra[folder] = dict(status="unavailable",basis="weather classification relies on official split only")
        records.append(dict(row,sha256=digest(path),stats=point_stats(points,config["distance_bins_m"]),metadata=extra))
    calib = root/"calib_tf_tree_full.json"
    if not calib.is_file():
        raise Blocked("BLOCKED_STF_EXTRINSICS_MISSING")
    tree = json.loads(calib.read_text("utf-8"))
    transform_check=sensor_transform_audit(tree)
    minimum = config["real_samples"]["minimum_recordings_per_category"]
    counts = {category:len({r["recording"] for r in records if r["category"]==category})
              for category in config["real_samples"]["splits"]}
    comparable = not frozen["smoke"] and all(v>=minimum for v in counts.values())
    report = dict(status="statistically_compared" if comparable else "smoke_or_insufficient_recordings",
        records=records,independent_recording_counts=counts,range_bins_m=config["distance_bins_m"],
        format="little-endian float32 x5; XYZ/intensity/fifth feature preserved, fifth not interpreted as OPV2V beam identity",
        sensor="Velodyne HDL64-S3D, last return",coordinate="ROS extrinsic tree, lidar_hdl64_s3_roof; do not infer OPV2V equivalence",
        calib_sha256=digest(calib),sensor_transform_check=transform_check,unpaired=True,per_point_loss_rate=None,
        physical_calibration_passed=False,opv2v_threshold_transfer_allowed=False,
        unavailable_sensor_parameters=["verified intensity-to-power mapping", "receiver threshold/noise floor", "per-beam waveform", "OPV2V sensor equivalence"],
        statistical_comparison_scope="per-recording point-count/range/intensity/near-range distributions; scene content and sensor confounds remain",
        authenticity="use DOWNLOAD_MANIFEST checksum status; no per-sensor official checksum is invented")
    atomic_json(output/"R0_SENSOR_AUDIT.json",report)
    lines=["# R0 真实天气检查", "",report["status"],"",
        "固定官方 split 样本；晴天与雾天不配对，不能计算逐点丢失率。传感器与 OPV2V 不同。",
        "已进行统计比较不等于完成物理标定。未得到 OPV2V 接收阈值或强度到功率映射。", "",
        "| 类别 | 样本 | 独立 recording | 平均点数 |", "|---|---:|---:|---:|"]
    for category,count in counts.items():
        rr=[r for r in records if r["category"]==category]
        lines.append(f"| {category} | {len(rr)} | {count} | {np.mean([r['stats']['points'] for r in rr]) if rr else None} |")
    lines += ["", "完整逐样本、分距离统计和类别依据见 R0_SENSOR_AUDIT.json；近距离点不自动视为天气杂波。", ""]
    atomic_text(output/"R0_RESULTS.md","\n".join(lines))
    return report


def compare_simulation(real_report, simulated_rows, output):
    # Store actual numerical distributions, never fit simulator parameters to them.
    categories = {}
    for category in {r["category"] for r in real_report["records"]}:
        rr = [r for r in real_report["records"] if r["category"] == category]
        per_recording={}
        for recording in sorted({r["recording"] for r in rr}):
            hist=np.asarray([r["stats"]["range_bins"] for r in rr if r["recording"]==recording],dtype=float)
            per_recording[recording]=(hist/np.maximum(hist.sum(axis=1,keepdims=True),1)).mean(axis=0).tolist()
        categories[category] = dict(samples=len(rr),recordings=len(per_recording),per_recording_range_bin_fractions=per_recording,
            mean_range_bin_fractions=np.asarray(list(per_recording.values())).mean(axis=0).tolist())
    simulation = {}
    for mode in ("B0","B1","B2","B3","B4"):
        rr = [r["stats"] for r in simulated_rows if r["mode"]==mode]
        if not rr:
            continue
        per_scene={}
        for scene in sorted({r["scene"] for r in simulated_rows if r["mode"]==mode}):
            hist=np.asarray([r["stats"]["output_distance_bins"] for r in simulated_rows if r["mode"]==mode and r["scene"]==scene],dtype=float)
            per_scene[scene]=(hist/np.maximum(hist.sum(axis=1,keepdims=True),1)).mean(axis=0).tolist()
        mean=np.asarray(list(per_scene.values())).mean(axis=0)
        simulation[mode] = dict(mean_sampled_range_bin_fractions=mean.tolist(),per_scene_range_bin_fractions=per_scene,
            l1_to_real={k:float(np.abs(mean-np.asarray(v["mean_range_bin_fractions"])).sum()) for k,v in categories.items()})
    data=dict(real=categories,simulation=simulation,parameters_fitted=False,calibration_passed=False,
        limitation="OPV2V uses fixed sampled target-conditioned rays; STF uses complete unpaired last-return scans. These descriptors have different scene and sampling measures; numerical L1 is descriptive only, never a validation gate or calibration score.")
    atomic_json(output/"REAL_SIM_COMPARISON.json",data)
    return data

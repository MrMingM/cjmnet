"""R0: audited source and actual Clean schema, without any real-weather IO."""
from pathlib import Path

from scorw_repair.core import REPO, Blocked, atomic_json, atomic_text, read_json, digest

REAL_STATUS = "NOT_VALIDATED_NO_REAL_REFERENCE"


def audit(config, coordinate_root, output):
    coordinates=read_json(coordinate_root/"REAL_COORDINATE_AUDIT.json")
    inputs=read_json(coordinate_root/"INPUT_MANIFEST.json")
    if coordinates.get("status")!="passed" or not inputs:
        raise Blocked("R0_CLEAN_COORDINATE_IDENTITY_GATE_FAILED")
    headers=[]
    for row in inputs:
        fields=None; encoding=None
        with Path(row["pcd"]).open("rb") as stream:
            for _ in range(100):
                line=stream.readline().decode("ascii").strip()
                if line.startswith("FIELDS "):fields=line.split()[1:]
                if line.startswith("DATA "):
                    encoding=line.split()[1];break
        if fields!=["x","y","z","rgb"] or not encoding:
            raise Blocked("R0_UNSUPPORTED_ORIGINAL_PCD_FIELDS")
        if digest(row["pcd"])!=row["pcd_sha256"] or digest(row["yaml"])!=row["yaml_sha256"]:
            raise Blocked("R0_CLEAN_INPUT_CHANGED")
        headers.append(dict(scene=row["scene"],timestamp=row["timestamp"],cav=row["cav"],
            fields=fields,encoding=encoding,points=row["points"],pose=row["pose"],
            pcd_sha256=row["pcd_sha256"],yaml_sha256=row["yaml_sha256"]))
    required=["reference_weather/UPSTREAM.json","reference_weather/assets_lock.json",
              "reference_weather/vendor/fog/fog_simulation.py"]
    optional=["related_code/LiDAR_fog_sim/theory.py",
              "related_code/LiDAR_fog_sim/generate_integral_lookup_table.py",
              "related_code/weather_papers/Multi-Sensor_Alignment_2026.pdf",
              "related_code/weather_papers/Impact_MultiScattered_LiDAR_2024.pdf"]
    result=dict(status="SOURCE_AND_CLEAN_SENSOR_AUDIT_PASS",real_validation_status=REAL_STATUS,
        physical_calibration_passed=False,real_data_compared=False,
        downloads_enabled=False,extraction_enabled=False,private_url_files_read=False,
        sources={n:digest(REPO/n) for n in required},
        optional_materials={n:dict(present=(REPO/n).is_file(),blocking=False,
            sha256=digest(REPO/n) if (REPO/n).is_file() else None) for n in optional},
        clean_inputs=headers,coordinate_audit_sha256=digest(coordinate_root/"REAL_COORDINATE_AUDIT.json"),
        intensity="original OPV2V RGB red channel /255; 8-bit values checked by frozen Clean audit; no power/reflectance calibration",
        pose_convention="[x,y,z,roll,yaw,pitch], degrees; frozen world transforms reused",
        recoverable=["observed Clean XYZ", "encoded intensity", "CAV poses", "Clean object IDs/boxes"],
        unavailable=["transmitted/missing rays", "physical laser channel IDs", "per-beam timing",
            "received waveform", "detector noise floor/threshold", "intensity-to-power curve",
            "OPV2V-calibrated pulse/overlap/phase-function parameters", "real fog reference captures"],
        receiver_parameters="Hahner model defaults, not measured OPV2V hardware",
        approximation="clear-return-conditioned finite-pulse single scattering; beta(s)/alpha(s) fixed",
        real_reference_substitutes_used=False,future_real_reference_interface="dormant modules only; a future protocol must explicitly enable and audit data")
    atomic_json(output/"R0_SENSOR_AUDIT.json",result)
    atomic_text(output/"R0_RESULTS.md","# R0 源码与传感器审计\n\n"+REAL_STATUS+
        f"\n\n核验 {len(headers)} 份原始 OPV2V train 点云和对应位姿/GT 身份。"
        "本阶段只验证晴天数据字段与源码依据，没有真实雾天统计比较。\n\n"
        "原始字段为 x/y/z/rgb，强度取红通道/255。波形、逐束时刻、真实接收阈值和强度功率曲线不可恢复。"
        "脉宽、光学重叠和散射参数沿用 Hahner 模型默认值，未获 OPV2V 物理标定。\n\n"
        "自动下载、解压和私有链接读取均关闭；缺少可选论文或 STF 材料不阻断数值实验。"
        "OPV2V-W 与自生成天气均未充当真实标定数据。\n")
    return result

# UECP + GSPR：V2X-R E0–E3 一键流水线

代码已经接入数据、训练、选模、原生评测、恢复和报告。真实模型等价性、梯度、数据、HIP和训练结果需要在服务器确认。本地没有执行科研训练、推理或读取真实PCD；没有实验AP结果。具体审计见 [INTERFACE_AUDIT.md](INTERFACE_AUDIT.md)，本地记录见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

## 启动时 attfuse_gspr/README.md 哈希报错的修复

初版把未执行的GSPR/UECP文档、图片及其他模型文件也纳入了直接依赖。服务器文档不同会在训练开始前误拦截。修复后，GSPR固定实际显式加载的reliability.py；UECP固定从模型/损失入口递归导入的12个Python文件及实际使用的YAML。原生V2X-R历史身份链和实际运行代码SHA256继续严格核对；所有不匹配代码一次性列出expected/actual，不在服务器重新计算期望值绕过校验。

对于报错位于`pipeline.py`构造`request`时的情况，实验尚未创建request.json，可保留原OUTPUT_ROOT。上传`uecp_gspr_v2xr_source_pins_fix.zip`到`/home/cjm/`后执行：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
/home/cjm/miniconda3/envs/opencood/bin/python /home/cjm/uecp_gspr_v2xr_source_pins_fix.zip --project "$PWD" --output "$OUTPUT_ROOT"
/home/cjm/miniconda3/envs/opencood/bin/python -B uecp_gspr_v2xr/check_static.py
RESUME=1 sh uecp_gspr_v2xr/run_all.sh
tail -f "$OUTPUT_ROOT/launcher.log"
```

安装器先检查新的运行依赖、旧版/新版已知文件哈希和该输出没有实验契约，再备份旧包中将改变的文件并只更新uecp_gspr_v2xr目录，source_snapshot.json最后提交。历史GSPR/UECP文件、启动日志不修改；已创建request.json的实验拒绝原地换代码。重复执行已完成的修复不会再修改文件。最后的run_all.sh仍默认后台运行，需要沿用原先设置的GPU等环境变量。

## 实验做什么

|阶段|实际实现|
|---|---|
|E0|原版UECP LiDAR-only前向、多尺度、不确定性头、融合/单车检测及方向损失、密度辅助损失；适配V2X-R|
|E1|冻结已完成OPV2V天气监督的GSPR，统计区域质量并比较密度A、质量B、联合C、打乱质量D；可信标签不足时INCONCLUSIVE|
|E2|113参数、末层零初始化的质量适配器补充原UECP不确定性图；GSPR冻结；单独训练原detach对照|
|E3|覆盖C管理区域内聚合/跨尺度先验，质量Q管理车辆间precision；真实观测掩码warp，平滑后再次清除缺失区域|

共9个独立检测训练：E0、E2、E2_original_detach、E3_joint、E3_coverage、E3_quality、E3_shuffled、E3_no_separation、E3_constant_quality。默认共180个检测器训练epoch，另有开发集评测、一次锁定模型后的三天气829帧评测、E1与smoke；可能运行很久。20轮为固定预算，不自动延长。独立输出预留至少50–150GiB，实际以服务器模型大小为准；启动前低于30GiB直接停止。

## 数据和公平性

- 固定接受8083协作帧、每天气24088单车帧；32个场景固定划分26 core/6 dev，完整UID列表写入输出，不虚构8084帧。
- 每epoch每协作帧一次等概率Mix3天气选择，同帧全部CAV使用相同天气；所有组的顺序、天气与实际CAV记录逐项比较。
- seed20261017，20epochs，FP32，单HIP卡，microbatch1×accumulation8。所有组相同microbatch/有效batch8。可在**新输出开始前**统一设置MICROBATCH=2/4/8；不自动OOM降级、不分组改batch。最后不足8的训练尾批统一丢弃并记录。
- AdamW lr.002/eps1e-10/wd1e-4；完成第10/15轮后lr×.1。共享骨干、融合及检测头初始化哈希相同；额外模块初始化不改变主模型随机流。
- 逐epoch只用train_dev Clean **O AP70的未舍入值**选模，精确并列取最早。20轮全部完成后锁定selected.json，再用829 validation最终评测。所有正式输出同时保留O、O*、固定GT的F，使用已有原生预测/AP函数。
- 最终推理只接收当前天气LiDAR，不传GT不确定性图、不读配对Clean。最终F协议使用已冻结的Clean GT作评价目标，这是评测器的GT处理。
- 历史829 validation已多次使用，本轮不能宣称独立盲测；未访问正式test。旧V2X-R Clean-only GSPR负结果作为风险约束；跨域GSPR有效性必须由本轮数据判断。

## 同步代码

最新完整交付包为 `uecp_gspr_v2xr_sync_v2.zip`，只含新目录及被SHA256固定的只读源码依赖，没有数据、checkpoint或.git。服务器已存在的历史源码必须与归档对应SHA相同；解包采用**存在且相同则保留、缺失才新增、内容不同则停止**的方式，不覆盖历史代码/结果。已经安装初版的本次启动失败请使用上方专用修复包。

Windows PowerShell（把地址换成实际服务器）：

```powershell
$ResearchServer = 'cjm@你的服务器地址'
scp 'D:\Collaborative perception\mymodule\cjmnet\uecp_gspr_v2xr_sync_v2.zip' "${ResearchServer}:/home/cjm/uecp_gspr_v2xr_sync_v2.zip"
```

在服务器运行以下短命令，检查全部冲突后才新增缺失文件：

```sh
/home/cjm/miniconda3/envs/opencood/bin/python - <<'PY'
from pathlib import Path
import hashlib, zipfile
root = Path('/home/cjm/OpenCOOD-main/cjmnet').resolve()
with zipfile.ZipFile('/home/cjm/uecp_gspr_v2xr_sync_v2.zip') as z:
    files = [(r, root / r.filename) for r in z.infolist() if not r.is_dir()]
    for row, path in files:
        if not path.resolve().is_relative_to(root):
            raise SystemExit('Unsafe archive path')
        if path.exists() and hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(z.read(row)).digest():
            raise SystemExit('Source differs; preserve and reconcile before running: ' + str(path))
    for row, path in files:
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as f:
                f.write(z.read(row))
print('Missing code added; existing matching files retained')
PY
cd /home/cjm/OpenCOOD-main/cjmnet
/home/cjm/miniconda3/envs/opencood/bin/python -B uecp_gspr_v2xr/check_static.py
sh -n uecp_gspr_v2xr/run_all.sh
```

如果旧服务器缺少上轮 `v2xr_raygen_e0e3` 包，归档会补齐本轮实际复用的底层文件；本轮只复用其中I/O、原生适配、恢复和身份检查，不执行旧AttFuse扩散实验，也不要求旧实验控制器或文档保持同一版本。

## 启动：一个总脚本自动走完

先运行 `rocm-smi` 检查GPU占用，选空闲物理卡号。代码还会在服务器自检前检查至少90%显存空闲；这不能代替查看GPU计算利用率。以下卡号由你填入，不固定使用某张卡。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
rocm-smi
export GPU=你选择的物理卡号
export OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1
sh uecp_gspr_v2xr/run_all.sh
```

默认BACKGROUND=1，脚本内部nohup后台运行并立即打印PID和日志；没有日志管道，子进程返回码直接保留。已存在的输出拒绝普通启动，请显式RESUME=1；不覆盖旧实验。也可不指定OUTPUT_ROOT，自动生成带时间戳的新目录；恢复时必须明确提供打印出的旧路径。

执行顺序：preflight → 服务器tensor self-test → 九组极小真实smoke及恢复重放 → 全部训练天气内容校验 → E0完整训练/选模/最终评测 → E1诊断 → E2及detach对照 → E3及全部独立重训消融 → 报告。负AP差值会继续；数据/权重不兼容、NaN/Inf、无预期梯度、GT不一致或缺结果会停止。

只通过ROCR_VISIBLE_DEVICES选择卡；脚本清除HIP_VISIBLE_DEVICES和CUDA_VISIBLE_DEVICES。训练/评测使用逻辑cuda:0（PyTorch的ROCm接口命名）。物理GPU型号和卡号写入契约，恢复不自动换卡。默认PYTHON是 `/home/cjm/miniconda3/envs/opencood/bin/python`。

## 分步确认 self-test 和 smoke

这些模式也默认后台运行；每一步等日志显示正常退出后，再启动下一步，避免同一输出的kernel lock冲突。最后all模式会验证已完成结果并复用。

```sh
GPU=物理卡号 OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1 MODE=preflight sh uecp_gspr_v2xr/run_all.sh
GPU=物理卡号 OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1 RESUME=1 MODE=self-test sh uecp_gspr_v2xr/run_all.sh
GPU=物理卡号 OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1 RESUME=1 MODE=smoke sh uecp_gspr_v2xr/run_all.sh
GPU=物理卡号 OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1 RESUME=1 MODE=all sh uecp_gspr_v2xr/run_all.sh
```

self-test检查原版E0同张量输出/辅助/检测/方向损失等价、真实坐标warp、原版detach与受控梯度、E3空区域掩码、全空输出、适配器恒等、冻结GSPR和全部方法的梯度。smoke每组2轮×2次有效更新、极少真实帧；训练进程释放GPU后才运行评测；E0第1轮在首个有效更新后主动中断，恢复后与独立不间断重训比较；另执行Clean缓存中断恢复及E0/E2/joint第2轮的完整optimizer/scheduler/RNG恢复重放。数值重放容差预设atol1e-5/rtol1e-4，RNG与整数精确匹配，并记录是否逐位相同。smoke结果保存在独立smoke目录，不进入科研表格。

## 查看进度、停止、恢复

```sh
tail -f /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/launcher.log
cat /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/current_stage.json
cat /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/pipeline_status.json
# 阶段详细日志路径在current_stage.json；例如：
tail -f /data/cjm/datasets/logs/uecp_gspr_v2xr_v1/logs/E0_train_01.log
```

```sh
/home/cjm/miniconda3/envs/opencood/bin/python -B uecp_gspr_v2xr/stop.py /data/cjm/datasets/logs/uecp_gspr_v2xr_v1
GPU=原物理卡号 OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1 RESUME=1 sh uecp_gspr_v2xr/run_all.sh
```

stop.py检查PID启动时间和命令身份，只通知本任务controller，由它转发到自己启动的子进程组。恢复检查源码/config/data/checkpoint身份，完成的epoch和阶段核对输出哈希后跳过。每100个有效更新保存一次交替槽位的原子恢复点（smoke每次更新保存），中断最多重做尚未提交的更新；包含AdamW、scheduler、CPU/HIP/Python/NumPy RNG和采样位置，恢复后不重复累计已提交更新。epoch完成后另外提交完整resume.pth和detector.pth。成功训练但评测中断时复用已有模型；预测缓存逐帧校验与续写。

目录存在、文件名存在或进程返回0均不足以认定完成，必须有完整且哈希一致的commit记录。状态包含PENDING/RUNNING/COMPLETE/FAILED/SKIPPED（SKIPPED用于已预期的smoke故障注入）；失败记录command/epoch/UID线索/traceback/log，另自动输出PARTIAL报告。

只重生成已有结果的报告：

```sh
GPU=原物理卡号 OUTPUT_ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1 RESUME=1 MODE=report sh uecp_gspr_v2xr/run_all.sh
```

## 服务器必须已有的输入

预检会逐项验证，不下载数据、不重新提取归档、不执行正式test：

- TRAIN_ROOT `/data/cjm/datasets/v2xr-official-train-v2`（内部train及manifest/data_acceptance）。也接受以train结尾的实际数据路径。
- VALIDATE_ROOT `/data/cjm/datasets/v2xr-official-validate-v1/validate`。
- PAIRED_VALIDATE_ROOT `/data/cjm/datasets/v2xr-lidar-weather-validate-v1`。
- RUNTIME_ROOT `/data/cjm/datasets/v2xr-official-runtime-v1`，以及已验收的D1A_ROOT、GT_AUDIT_ROOT、D1C_ROOT和MODEL_DIR；默认值沿用现有已验证流水线。
- GSPR checkpoint `/data/cjm/datasets/logs/gspr_joint_full_v1_seed20260907_20260907_161155/net_best_validation.pth` 及同目录config.yaml，固定SHA256见settings.json。缺失或不兼容立即停止，不用随机质量替代。

所有路径、配置与源码SHA均写入独立契约，不需要Git。SEED/EPOCHS/MICROBATCH可在新输出前统一覆盖，恢复时须完全一致；论文比较默认保持20261017/20/1。修改代码后必须用新输出并重新固定源码身份，不能把已开始实验的代码静默换掉。

## 可选可信质量标签

默认没有可确认来源的V2X-R逐点天气标签，E1会报告INCONCLUSIVE；E2/E3继续作为探索性对照。PCD绿色通道不自动当噪声真值，也不把未匹配Clean的点标为噪声。

如果已经有仿真器记录的真实逐点来源，可在第一次启动时设QUALITY_LABEL_MANIFEST为JSON路径。格式：

```json
{
  "kind": "simulator_recorded_point_origin",
  "split": "train",
  "provenance": {"generator": "具体仿真器/版本和生成记录位置"},
  "files": {
    "场景/CAV/时间戳_fog_0.060.pcd": {
      "pcd_sha256": "已验收训练manifest中的PCD哈希",
      "path": "/data/cjm/datasets/quality_origins/对应标签.npz",
      "sha256": "标签文件哈希"
    }
  }
}
```

每个npz包含labels[N]，按**当前天气原始PCD逐行**对齐，1可靠、0仿真器记录噪声、-1未知。程序对近邻冲突剔除、槽位掩码、标签覆盖和scene隔离作检查；没有来源的文件保留未知。ABCD采用固定ridge回归预测区域可靠比例，core拟合、dev诊断；先验判定规则与场景bootstrap写入protocol.json，不作不可信AUROC包装。

## 输出位置

```text
OUTPUT_ROOT/
  protocol.json / contract.json / source_snapshot.json / data_manifest.json
  pipeline_status.json / current_stage.json / logs/
  self_test/ / smoke/ / smoke_replay/
  E0/baseline/epochs/01..20/{sampling,statistics,train_complete,dev_complete}.json
  E0/baseline/selected.json / final_evaluation/
  E1/quality_diagnosis/{feature_statistics,ABCD,shard_manifest,complete}.json
  E2/gspr_assisted_uecp/
  E3/dual_quality_fusion/joint/
  ablations/{E2_original_detach,E3_coverage,E3_quality,E3_shuffled,E3_no_separation,E3_constant_quality}/
  FINAL_RESULTS.md / FINAL_RESULTS.json / FINAL_metrics.csv / ABLATION_RESULTS.md
  PARTIAL_RESULTS.* / failure_history/   # 发生失败时
```

报告自动列出三天气O/O*/F AP30/50/70、E2/E3相对E0的pp差值、Clean下降情况、E1结论、独立重训消融、参数/历史预训练成本、训练时间、显存和同步前向耗时。缓存复用导致只测到部分前向时明确timing_complete=false，不把它当全程计时。缺结果填N/A及原因，已完整提交的报告保持原字节。

## 本地复查（不运行科研实验）

```sh
python -B uecp_gspr_v2xr/check_static.py
python -B -m uecp_gspr_v2xr.tests_local
sh -n uecp_gspr_v2xr/run_all.sh
```

服务器self-test/smoke和正式训练尚未在本地执行，不能据本地合成检查声称模型已验证有效。

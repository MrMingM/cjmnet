# 原版天气模拟 + AttFuse 开发评测

本目录接入 V2X-DGW 论文引用的 Fog、LISA Rain、Snow 源码，将天气点云存成
项目能读取的 PCD，再使用已有的纯 AttFuse checkpoint 评测。所有科研运行都在
Linux 服务器进行。本地只编写代码和做静态检查。

默认只处理原始 **validate 全部帧**，不会读取或修改正式 test，也不训练模型。
这一步回答“原版模拟器能否建立足够困难的开发集”。天气数据、日志和结果都在
`/data/cjm/datasets`；不会覆盖旧 `opv2v-physics-fixed-v1`。

## 已固定的第一轮协议

| 项目 | 配置 |
|---|---|
| Clean 输入 | `/data/scd/datasets/opv2v_official_data_dumping/validate` |
| 输出 | `/data/cjm/datasets/opv2v-reference-weather-v1` |
| 参数与随机种子 | `config.yaml`，seed=20261013 |
| Fog | 原版默认 alpha=.06，gamma=1e-6，官方批处理示例 noise=10 |
| Rain | LISA 原论文 `python_old`，33 mm/h（上游 README 示例），strongest，rmax=200m |
| Snow | 1.5 mm/h、下落速度1m/s、Gunn 分布、原版80m预采样雪粒场 |
| 检测器 | `attfuse_code.PointPillarIntermediate`，纯 AttFuse，不含 GSPR |
| 权重 | 原始 run 的 `net_epoch13.pth`，严格加载全部参数 |
| AP 主指标 | 项目现有 BEV 多边形 IoU，global_sort=False，AP30/50/70 |
| 另存指标 | global_sort=True，单独记录，不与主指标混用 |

以上数值是本轮开发协议，**OPV2V-W 作者的完整生成配方仍未知**。即使 AP 差距相似，
也不能宣称完全复现了官方天气分布。Fog/Rain 的 AP70 差距均≥10 pp 只是最低筛选门槛。

## 同步与安装

将整个 `reference_weather/` 同步到服务器
`/home/cjm/OpenCOOD-main/cjmnet/reference_weather/`，保留当前项目的 OpenCOOD 与
`attfuse_code/`。服务器不需要 `.git`，不需要 clone 任何上游仓库。
`source_snapshot.json` 与源码必须一起同步；不要为通过检查手动改哈希。
同步时保留文件字节与换行格式。默认还会核对原始 validate 为1980帧。

在服务器现有 opencood 环境安装额外依赖（不替换现有 Torch/spconv/NumPy）：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
/home/cjm/miniconda3/envs/opencood/bin/python -m pip install -r reference_weather/requirements-server.txt
mkdir -p /data/cjm/datasets/logs
```

第一次运行会从固定上游版本下载约1MB的 Fog 表及 LISA Mie 系数，以及上游
约2.3GB雪粒压缩包；只解出本轮需要的64个雪粒文件。压缩包保留，便于中断后复用。
Fog/Mie 文件使用上游 blob 内容哈希校验。雪粒包上游未提供预期 SHA256；首次下载
记录实际 SHA256，后续拒绝变化，不能把这称为上游签名验证。

## 先做服务器 smoke，再完整评测

`GPU=0` 仅为写法示例，请换成当前空闲的物理 HIP 卡号。脚本仅设置
`ROCR_VISIBLE_DEVICES`，清除另外两个可见设备变量。

8帧 smoke 也可能耗时，放后台运行。它会完成资源准备、依赖和权重检查、服务器单元检查、
三天气生成，以及四条件 AttFuse 评测；小样本结果不会通过完整开发集门槛。

```sh
nohup env GPU=0 LIMIT_FRAMES=8 \
  OUTPUT_ROOT=/data/cjm/datasets/opv2v-reference-weather-smoke-v1 \
  sh reference_weather/run_server.sh \
  > /data/cjm/datasets/logs/reference_weather_smoke_v1.log 2>&1 &
echo $!
```

检查 smoke 日志与 `opv2v-reference-weather-smoke-v1/evaluation/RESULTS.md`。
确认三天气及 Clean 都成功后，启动完整 validate：

```sh
nohup env GPU=0 sh reference_weather/run_server.sh \
  > /data/cjm/datasets/logs/reference_weather_full_v1.log 2>&1 &
echo $!
```

原版 Rain 为逐点采样，Snow 为逐束遮挡/波形计算，完整生成可能很久。
`WORKERS=4` 是默认生成进程数；每个 Snow 进程默认2个通道线程。
中断后使用**相同参数、相同输出目录**加 `RESUME=1` 继续：

```sh
nohup env GPU=0 RESUME=1 sh reference_weather/run_server.sh \
  >> /data/cjm/datasets/logs/reference_weather_full_v1.log 2>&1 &
echo $!
```

恢复会检查原始 PCD、YAML、输出 PCD、参数和源码哈希；完整匹配才复用。
已经完成的天气评测也会复用。PCD 写出后中断而缺少记录时会重新确定性生成并核对，
已有不同内容会报错，不会覆盖。并发启动同一数据目录会被锁拒绝。

可分别执行 `STAGE=assets`、`STAGE=generate`、`STAGE=evaluate`、`STAGE=report`。
其中 assets/generate 不需要 GPU；evaluate 不会重新模拟天气。
可用 `ATT_CONFIG`、`ATT_CHECKPOINT` 指定准确的纯 AttFuse 配置和权重文件。

改变物理参数、天气或 split 时，先把 config.yaml **复制到数据盘**，编辑副本，
通过 `CONFIG=/data/.../protocol.yaml` 传入，并指定新的 OUTPUT_ROOT。
不要修改目录内已记录哈希的默认 config.yaml。
以后生成训练天气可在外部配置中设置 `splits: [train, validate]`，使用新输出目录；
默认脚本不训练检测器。天气种类或 Snow 配置变化还需要新 ASSETS_ROOT。

## 数据与结果

```text
opv2v-reference-weather-v1/
  manifest.json                  # 参数、源码/资源哈希、选择的帧、点云汇总
  validate_frames.json
  clean/validate/<scene>/<cav>/   # 原始 clean PCD 和标注链接
  fog/validate/<scene>/<cav>/
  rain/validate/<scene>/<cav>/
  snow/validate/<scene>/<cav>/
  records/<weather>/validate/... # 每个PCD的输入/输出哈希、seed、几何统计
  evaluation/
    protocol.json                # 精确checkpoint哈希、AP协议和模型源码哈希
    clean/result.json
    fog/result.json
    rain/result.json
    snow/result.json
    summary.json
    RESULTS.md
```

所有标注及图像链接原始 Clean，源目录需要保留。天气添加发生在每辆车自己的
雷达坐标系中，随后由现有数据集投影到 ego。PCD 按现有 RGB 约定编码，强度量化为8bit，
每个输出都会读回核对。不会补回 clean 点来掩盖空点云；极端天气导致空体素等错误时停止，
保留现场以便服务器诊断。

## 上游源码与适配范围

源码、上游固定版本、变更记录见 `UPSTREAM.json`；许可证随各目录保留。
Fog/Snow 为 CC BY-NC 4.0，LISA 为 GPL-3.0，分发源码时保留对应许可。

- Fog：保留原版查表、衰减和回波替换；适配0–1与0–255强度、每PCD随机数和零回波过滤。
  原版 `ParameterSet(alpha=...)` 在覆盖 alpha 前计算 beta；本轮明确保留此构造行为，
  实际 alpha/beta/gamma 逐PCD记录。改 alpha 时不能默默重算 beta。
- Rain：保留 `python_old` 的粒子枚举和回波竞争；使用原版 Mie 系数，不重新计算。
  固定表对应 m=1.328、波长905nm，配置中更改这两项会被拒绝。
  处理第五列丢点/雨滴/目标标签。为新 SciPy 补 `integrate.trapz=np.trapz` 兼容别名，
  不改变积分规则。为兼容原版相对文件路径，仅在构造时临时切换资源目录。
- Snow：用户已同意采用 `opv2v_elevation_no_focal_v1`。按每个点云的仰角聚类重建
  可观察到的扫描束，最多64个；它们是重建标签。歧义、超限或不匹配过多会停止。
  关闭 HDL-64 专属焦距修正、最低强度和特殊通道上限；摄像头 FOV 裁剪关闭。
  原版地面窗口固定于 STF 安装高度，改为 OPV2V 路面下包络 RANSAC 平面，保留后续
  原版地面强度拟合/噪声阈值。该适配也逐点云记录平面和仰角中心。
  保留原版雪粒遮挡、波形叠加和回波替换计算；空扫描束直接返回；波形数组支持超过120m
  的 OPV2V 目标。仅给 ThreadPool 设置线程上限，不改变粒子计算。
  发布雪粒场半径80m，远处目标只受到该粒子场内的遮挡；超出80m数量单独统计。
  当前没有调用额外湿地面模拟，避免加入未经定义的天气组合。

本地验证仅为语法/静态检查。服务器单元检查主要验证强度转换、丢点标签处理、种子、
通道重建和帧选择；实际物理模拟和 AttFuse 数值结果必须由服务器 smoke/full 运行确认。

# V2X-R 配对天气 LiDAR 子集

当前服务器的 V2X-R 是按场景/车辆保存的 `.7z` 压缩包。这个工具只提取用于纯 LiDAR
天气研究的部分，默认覆盖 validate 的所有场景和时间戳，保留每帧全部协作车辆。
提取 Clean、Fog、Snow 点云、逐帧 YAML 标注、场景 data_protocol.yaml。
相机、雷达、其他天气强度不落盘；train 需要显式指定；test 不读取、不提取。
既有 OPV2V 实验目录和 reference_weather 源码保持原样。

## 第一轮范围

| 项目 | 默认值 |
|---|---|
| 原始目录 | `/data/cjm/datasets/V2X-R` |
| 划分 | validate |
| 输出目录 | `/data/cjm/datasets/v2xr-lidar-weather-validate-v1` |
| 帧范围 | 每个场景全部时间戳，保持所有车辆及三条件对齐 |
| 天气强度 | 根据压缩包实际文件名识别；每种天气只有一档时自动选择 |
| 解压并行 | 一次一个车辆压缩包，7z最多2线程 |
| GPU | 不需要 |

官方 README 给出的 validate 为829个协作帧，这只是参考值，实际数量由 plan.json 列出。
“协作帧”表示一个场景的一个时间戳，包括当时各辆车的点云，不能与PCD文件数混用。
官方说明列出了 normal/fog/snow；本工具不生成 Rain。

来源：[官方数据说明](https://github.com/ylwhxht/V2X-R#-v2x-r-dataset-manual)、
[官方纯LiDAR AttFuse配置](https://github.com/ylwhxht/V2X-R/blob/master/opencood/hypes_yaml/V2X-R/LiDAR/V2XR_attfuse.yaml)。

## 同步和服务器运行

将整个 `v2xr_subset/` 同步到 `/home/cjm/OpenCOOD-main/cjmnet/v2xr_subset/`，
包括 source_snapshot.json，保留文件字节。服务器不依赖 `.git`。
使用现有 opencood Python，额外需要命令 `7zz`、`7z` 或 `7za` 三者之一。
脚本会自动寻找；找不到时请管理员提供7-Zip命令，再继续。这里不更改现有科研环境。

先后台生成清单。它会执行服务器合成数据检查、列出 validate 压缩包、校验源码、
计算原压缩包 SHA256，最后报告选中内容的大小和数据盘剩余空间。这个阶段不解出真实点云。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs
nohup env STAGE=plan sh v2xr_subset/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_subset_validate_plan_v1.log 2>&1 &
echo $!
```

```sh
tail -n 60 /data/cjm/datasets/logs/v2xr_subset_validate_plan_v1.log
```

成功时会写出 `plan.json`，并打印 `selected_GiB`、`all_archive_contents_GiB`、
`free_GiB`、协作帧数、所选天气后缀。检查计划符合预期后后台提取：

```sh
nohup env STAGE=extract RESUME=1 sh v2xr_subset/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_subset_validate_extract_v1.log 2>&1 &
echo $!
```

提取使用 plan.json 中固定的文件清单。压缩包若使用整块压缩，7z可能仍需计算其他数据流，
因此只提取少量文件也可能耗时；未选中的相机/雷达文件不会写到磁盘。
临时目录默认在 `/data/cjm/datasets/tmp/v2xr_subset`，避免占用 home 临时空间。
中断后用同一提取命令恢复；已完成压缩包会核对文件哈希后复用。
一个压缩包在写出记录前中断时，会重新提取该包所选文件并核对已写出的内容。
已有不同内容会报错；原始压缩包不覆盖。相同输出目录由文件锁保护。

若发现多档 Fog/Snow，plan会保存 inventory.json 并停止，打印所有共同可用的后缀。
根据实验方案明确选择后再规划，例如后缀确实存在时：

```sh
nohup env STAGE=plan FOG_VARIANT=fog_0.060 SNOW_VARIANT=snow_2.5_2.0 \
  sh v2xr_subset/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_subset_validate_plan_v1.log 2>&1 &
echo $!
```

上面两种强度名称来自官方配置示例，仅用于说明语法，不表示它们就是本机压缩包的内容。
缺少Clean/标注/天气配对或车辆间时间戳不同会报错，不会悄悄只取交集。

## 输出

```text
v2xr-lidar-weather-validate-v1/
  inventory.json                   # 实际压缩包成员和文件名
  plan.json                        # 固定选择、原包哈希、空间统计、源码哈希
  manifest.json                    # 提取进度/完整状态
  payload/validate/<scene>/<cav>/   # 所选原始文件；数据只存一份
  clean/validate/<scene>/<cav>/     # timestamp.pcd / timestamp.yaml 链接
  fog/validate/<scene>/<cav>/
  snow/validate/<scene>/<cav>/
  protocols/validate/<scene>.yaml
  records/validate/<scene>/<cav>.json
  samples.json                     # 三条件PCD头和第一份YAML前120行
  .partial/                        # 提取暂存；完成后保留7z日志与清单
```

提取成功只表示文件按清单配对并通过完整性校验。下一步先核对 samples.json 的点云
字段、强度编码、坐标/标注结构，再接入 AttFuse；本工具尚未训练或评测模型。
使用现有 OPV2V checkpoint 在 V2X-R 上评测属于跨数据集迁移，需要单独记录，
不能当作已经复现 OPV2V-W 天气分布。

## 后续训练子集

如果确定使用训练数据，可以显式设置 SPLIT=train、使用另一个 OUTPUT_ROOT。
`FRAMES_PER_SCENE=N` 可在每个场景内均匀选最多N个时间戳，覆盖场景开始与结束，
同时保留对应全部车辆和三天气。训练规模应在验证读取方式后再确定。
训练子集与验证子集始终从原有各自划分提取，不能互相混用。改变选择时使用新输出目录。

本地只做语法、快照和打包检查。真实压缩包读取、7z合成检查、提取均在服务器运行。

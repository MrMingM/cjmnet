# 纯 LiDAR AttFuse：V2X-R 跨数据集天气基线

输入已完成的 `v2xr-lidar-weather-validate-v1`，分别读取作者生成的 Clean、
`fog_0.060`、`snow_2.5_2.0` 点云。使用已有 OPV2V 训练的纯 AttFuse epoch13，
严格加载全部参数。本阶段只做评测，尚未用 V2X-R 训练。

已核对实际数据：三条件PCD为 `FIELDS x y z rgb`，原始Clean为ASCII、天气为binary；
现有Open3D读取器均支持，强度取RGB红色通道并保持0–1。YAML含 `lidar_pose`、
`ego_speed`、`vehicles`，车辆字段为 `location/angle/center/extent`，对应现有
世界坐标投影接口；extent是半尺寸，处理时由原函数转换成完整框尺寸。
保留原始点云和标注，不进行重编码或额外坐标翻转。

## 协议范围

- 输入：已提取validate的829个协作帧、20个车辆包；实际数量由plan.json校验。
- 模型：`attfuse_code.point_pillar_intermediate.PointPillarIntermediate`。
- 权重：`/data/cjm/datasets/logs/attfuse_opv2v_20260903_200923/net_epoch13.pth`。
- 沿用该checkpoint配置和现有OpenCOOD处理顺序，最多5辆车，通信范围70m，
  GT范围 `[-140,-40,-3,140,40,1]`，输入范围由checkpoint配置固定。
- 点云投影到ego，使用现有过滤/体素化/NMS；训练增强、在线天气增强和定位/时延噪声关闭。
- 不使用相机FOV裁剪或雷达，也不根据天气点云的可见性改变GT。
- 三条件逐帧核对GT数组内容/顺序相同；模型输出、框、多边形和每个IoU均检查有限性/有效性。
- 主AP为BEV多边形IoU、global_sort=False，AP30/50/70；global_sort=True另存JSON。

本轮是“OPV2V模型直接迁移到V2X-R”的基线。Clean自身的性能反映了场景、传感器等
变化；Clean与对应天气的差距用于观察该模型在V2X-R配对天气上的退化。
不能据此宣称复现OPV2V-W天气分布，也不能将数值当作V2X-R作者官方协议分数。
作者代码的FOV和基于点云筛选GT等步骤与本轮协议不同；查阅的固定版本见UPSTREAM.json。
V2X-R本轮只有Fog/Snow，Rain开发门槛仍需单独处理。

## 同步与先跑跨场景检查

将整个 `v2xr_attfuse/` 同步到 `/home/cjm/OpenCOOD-main/cjmnet/v2xr_attfuse/`。
保留已同步的 reference_weather/ 和 v2xr_subset/，两者的源码快照不修改。
不用重新解压数据，也不需要7-Zip参与评测。服务器可没有 `.git`，版本由源码哈希记录。

默认输出 `/data/cjm/datasets/logs/v2xr-attfuse-transfer-v1`。
先从每个场景均匀取两帧（当前6个场景共12帧），同时评测三条件，放后台运行。
GPU换成空闲物理HIP卡号：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env GPU=0 FRAMES_PER_SCENE=2 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-attfuse-transfer-smoke-v1 \
  sh v2xr_attfuse/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_attfuse_transfer_smoke_v1.log 2>&1 &
echo $!
```

脚本会先在服务器执行合成点云/框检查，再逐个校验12,164个已提取文件的哈希、PCD头、
标注字段及三天气链接；这段验证无需GPU计算，但读取约12.66GiB文件可能需要几分钟。
随后严格加载权重并评测。TMPDIR固定默认放数据盘，避免home临时空间问题。
在GPU选择上只设置ROCR_VISIBLE_DEVICES，清除另外两个可见设备变量。

```sh
tail -n 60 /data/cjm/datasets/logs/v2xr_attfuse_transfer_smoke_v1.log
```

三条件检查完成后启动全部829帧：

```sh
nohup env GPU=0 RESUME=1 sh v2xr_attfuse/run_server.sh \
  >> /data/cjm/datasets/logs/v2xr_attfuse_transfer_full_v1.log 2>&1 &
echo $!
```

评测中断后，在相同配置、相同输出目录下加RESUME=1恢复。已完成的条件会校验后复用；
未完成条件重新评测。天气参数/抽样/权重/源码改变时使用新输出目录。
小规模检查恢复时仍需保留FRAMES_PER_SCENE=2和smoke的OUTPUT_ROOT。

也可用 `STAGE=verify` 单独做数据验证，此阶段不需要设置GPU。
需要更换配置时，将config.yaml复制到数据盘编辑，通过CONFIG=/data/.../config.yaml传入。

## 输出和检查项

```text
v2xr-attfuse-transfer-v1/
  verification.json     # 数据哈希/链接/标注检查、三条件总点数和空点云数
  protocol.json         # 精确权重、源代码、实际配置、软件版本、帧顺序
  clean/result.json
  fog/result.json
  snow/result.json
  summary.json
  RESULTS.md
  last_failure.json     # 仅失败时写出；保留原失败记录供追溯
```

每条件结果记录逐帧GT指纹、预测/GT数量、警告摘要和几何检查数量。
RuntimeWarning在每帧按默认规则捕获汇总，避免刷屏；遇到实际NaN/无效框/IoU会停止并
记录场景时间戳。同一位置重复警告可能合并，warning_counts不是未经去重的总发生次数。
总输出点数包含天气替换回波，点数比不能直接解释成目标点保留率。

本地仅AST语法/源码快照/打包检查。合成检查与实际模型评测全部在服务器执行。

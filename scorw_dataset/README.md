# SCOR-W-Fog v1

本模块把原始OPV2V的三个完整划分生成一份共享非均匀雾数据，供内部训练和评测使用。每scene只有一个固定世界雾场，全部车辆、全部时间戳共享；每个原始点调用完整R3接收器，允许回波替换、距离改变和数字零编码无返回。没有训练、没有天气参数调优，也没有B3/B4显著性筛选。

当前修订为 `scorw-dataset-v1.1-timing-coverage`，只修复D0计时抽样漏掉距离段的问题。数据与日志使用新的coveragefix目录，保留旧版预检查报告。新源码不能恢复旧合同，新目录首次运行用RESUME=0。天气场、主seed、接收器、生成规模、计算步长及资源门槛都不变。

## 数据和规模

原始根：`/data/scd/datasets/opv2v_official_data_dumping`。

| 划分 | 预期scene | 预期协作帧 |
|---|---:|---:|
| train | 43 | 6374 |
| validate | 9 | 1980 |
| test | 16 | 2170 |

协作帧是scene+timestamp；一帧包含多个CAV PCD，实际PCD和原始点数由D0清点。每个CAV的全部时间戳都要处理，生成阶段没有通信范围、最大协作车辆数或点采样限制。任一计数、PCD/YAML配对或跨车时间戳不符，输出清单并停止，不修改预期数。

FULL数据根：`/data/cjm/datasets/scorw-fog-v1-coveragefix`，包含：

```text
clean/{train,validate,test}/<scene>/<cav>/<timestamp>.pcd
fog/{train,validate,test}/<scene>/<cav>/<timestamp>.pcd
manifests/     # 各划分冻结输入、输出完整清单
records/       # 每PCD的合同、原子提交、来源sidecar；每frame/CAV/scene的完成记录
audits/        # 最近失败文件
configs/       # 直接使用的AttFuse配置、数据路径映射
stages/        # D0-D7提交记录、原始输入审计、性能和检测结果
```

Clean划分是原始数据目录的符号链接；Fog PCD均为实际文件。全部YAML、图像和其他元数据逐文件链接原来源，保持原路径结构、GT、object ID和位姿字节不变。**本数据集依赖原始数据仍保存在原路径**。移动服务器时同时保留原始数据路径，或在核验清单后用跟随符号链接的复制方式物化链接；不要只复制Fog PCD后宣称元数据已齐全。冻结权重的评测视图也是链接，不复制或改变权重。

## 天气和物理边界

主seed=20261023。每scene seed通过冻结的SHA256规则由主seed、`scorw-fog-scene-v1`、split、scene身份派生。跨时间不重置；跨车不另抽场。世界边界由整个原始scene全部点云世界XYZ和传感器原点的包围盒加2m边距确定，覆盖每条原始射线路径，场外查询停止。边界不依赖目标标注或检测结果。

天气算法 `bounded-random-fourier-tanh-v1`；mean_alpha=0.06、length_m=25、contrast=0.8、modes=32。接收器 `single-scattering-finite-pulse-strongest-total-peak-v1` 的全部参数与冻结R3配置逐项相同，硬衰减步长0.5m、波形步长0.1m。场定义和接收器均只读复用原模块，不修改历史train/test保护规则。

这是以晴天已有返回为条件的单次散射、有限脉冲、最强叠加峰近似。原强度同时缩放软/硬信号，不能模拟经标定的材质反射率影响。8bit RGB强度量化造成零编码，未设置真实硬件接收阈值。0.9–1m重叠、20ns脉宽及gamma等为模型默认值；近场不可辨识部分保留unknown，高光学厚度不证明单次散射足够准确。未恢复真实扫描束、逐束时间、被遮挡物体或真实波形。某scene无返回数为0是允许的模型结果。

`REAL_WORLD_VALIDATION_MISSING` 始终保留。没有真实雾天分布校准，也不声称复现OPV2V-W或V2X-DGW的作者天气配方。工程、数值、读取和检测通过均不代表真实天气真实性已验证。

## 优化与等价门槛

新引擎仅把多条光路的同位置天气场查询拼接成NumPy批量查询。每个点的衰减积分、散射、卷积、峰竞争和编码仍由原 `single_response()` 执行。逐次检查接收器查询的位置和顺序相同；批次有限大小，默认16光路，最多200000个场查询点。CPU进程按PCD分工，不保存大型波形中间数组。

D1在服务器比较原版逐点与预取批量：合成距离/强度网格、零雾、均匀/非均匀场、旋转位姿和固定Train真实光路。float64分量预注册atol/rtol均1e-12；来源、有效性、原始身份和float32输出必须完全相同，最终RGB PCD SHA256必须相同。批次1/8/16均比较。还执行原Hahner独立积分/查表回归、空PCD真实读取及9项合同/清点/中断恢复/计时覆盖验收。失败停止，不自动放宽容差、退回仅强度衰减或改变步长。

本地只做语法、JSON、LF和源码哈希静态检查；本轮数值和性能测试结果由服务器D0/D1实际产生，尚未提供虚构的加速倍数。

## D0资源预检查

完整清点源文件哈希和场景边界后，按Train场景的CAV数量、首帧传感器空间跨度选低/中/高分位场景，再在各scene按点数低/中/高分位固定3个云，在距离分层的固定点上测原版/批量/多进程速度；另测全云读取、写盘和读回。中间样本还完整计算一次全部光路，实测单CAV吞吐和进程内存峰值，结果留在D0计时目录，不计入数据集完成量。计时采样只用于估算，不用于正式生成。不同距离的实测耗时按全数据距离直方图加权，并以完整CAV实测作保守检查，结合多进程吞吐和2倍成本系数给出CPU小时、生成墙钟时间、IO和内存估算。包含进程启动开销，速度预估不保证完整任务ETA。

三个结构样本未覆盖的非空距离段，按照Clean距离直方图选择路径字典序最前且包含该距离段的Train PCD，按原始索引等间距取固定数量的计时光路。同一补充PCD覆盖多个缺失段时合并读取。每段最多取原预算的rays_per_cloud除以距离段数，补充规模受固定段数限制。全部PCD、原始索引、输入哈希与场身份在任何天气计时前写入PERFORMANCE_SAMPLE_FREEZE.json；不按计时或天气结果重新抽样。若某段仅在Validate/Test出现且Train无可用点，仍停止并明确列出距离范围，不静默借用测试集或填入估计耗时。完整距离覆盖失败时，预算标记为暂估，不能当作最终耗时。

预注册资源上限：生成预估60天、20000 CPU小时，最多32进程；磁盘按每原始点128字节保守估算，额外保留100GiB；每worker使用实测RSS两倍、1GiB库开销和批查询/单云存储估算，主机另保留2GiB。资源不足时保留 `GENERATION_BUDGET.json` 并停止，建议增加资源或审查新优化版本；不会减少scene、帧、点或物理过程。

D0扫描和SHA256核验也可能较久。SMOKE也完整核对源规模与scene边界；只在写出数据时截取每split第一scene前2帧、全部CAV和原始点，独立根为 `scorw-fog-v1-coveragefix-smoke`。FULL不继承SMOKE截断。preflight根为 `scorw-fog-v1-coveragefix-preflight`。

## 一键运行

将完整 `scorw_dataset/` 同步至 `/home/cjm/OpenCOOD-main/cjmnet`。不要改写冻结的旧目录或服务器源码快照。Python固定使用opencood环境路径，不安装/升级Torch、ROCm或其他依赖。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
mkdir -p /data/cjm/datasets/logs/scorw-fog-v1-coveragefix
```

可单独后台预检查并获取真实预算：

```sh
nohup env MODE=preflight RESUME=0 WORKERS=4 GPU=0 sh scorw_dataset/run_all.sh > /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/preflight.log 2>&1 < /dev/null &
```

一键SMOKE：

```sh
nohup env MODE=smoke RESUME=0 WORKERS=4 GPU=0 sh scorw_dataset/run_all.sh > /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/smoke.log 2>&1 < /dev/null &
```

一键FULL已经包含全部D0-D7，无须先手动完成子实验：

```sh
nohup env MODE=full RESUME=0 WORKERS=4 GPU=0 sh scorw_dataset/run_all.sh > /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/full.log 2>&1 < /dev/null &
tail -f /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/full.log
```

停止旧进程后恢复，保持相同MODE、WORKERS、GPU、源码、配置、输入及依赖版本：

```sh
nohup env MODE=full RESUME=1 WORKERS=4 GPU=0 sh scorw_dataset/run_all.sh >> /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/full.log 2>&1 < /dev/null &
```

CPU阶段屏蔽GPU；D6新进程才设置 `ROCR_VISIBLE_DEVICES=GPU`，清除HIP/CUDA变量。原数据只读。所有阶段和每个PCD都有锁、源码/配置/输入/雾场身份和原子提交；中断仅重做未提交的云，已提交PCD核验SHA256后复用。任意损坏或身份变化停止，要求新输出版本，不能手改CONTRACT或snapshot。

## 完成门槛、来源和统计

D2/D3/D4分别生成并审计三个split；D5用实际OpenCOOD catalogue核验所有scene/CAV/timestamp路径，对每份Clean/Fog PCD执行原读取器、预处理和collation。在Train/Validate每scene固定首/末帧额外运行native `__getitem__` 和frame collate；Test只做完整索引及点云预处理，不解释目标标注。D6在Validate全帧做固定纯AttFuse Clean/Fog推理，D7再校验全部源文件和全部输出。

每PCD的 `provenance.npz` 保留原始索引、来源、是否输出、光学厚度和选峰硬/软分量；原始XYZ/I从已固定SHA256的Clean PCD恢复，天气XYZ/I从Fog PCD读取，输出行按emitted原始索引顺序对应。来源0=模型原硬回波、1=纯模拟天气回波、2=unknown；不会根据模拟点落入GT框判断来源。GT框中的点也不被当作逐点语义真值。

根层报告包括任务要求的 `DATASET_PROTOCOL.json`、`SOURCE_INVENTORY.json`、`GENERATION_BUDGET.json`、`PERFORMANCE_BENCHMARK.json`、三个 `*_MANIFEST.json`、`PHYSICS_AUDIT.json`、`COMPATIBILITY_AUDIT.json`、`DATASET_QA.json`、`VALIDATION_ATTFUSE_RESULTS.json`、`FINAL_RESULTS.md`、`README.md`。完整源PCD/元数据哈希在SOURCE_INVENTORY指向的D0分片内。清单包含逐PCD距离/强度/来源统计和scene场身份；QA另给全部划分/scene总量及固定5×5×5世界网格alpha统计。网格平均不是精确场体积均值。

三个split、全部CAV帧、PCD读取和预处理、元数据、场身份、数值等价及全量哈希均通过才标记 `DATASET_COMPLETE`；SMOKE只能 `SMOKE_COMPLETE`。任意失败保留有效已完成数据，状态为 `NOT_COMPLETE`。资源预检查通过仅为 `PREFLIGHT_READY`。没有AP必须下降多少的要求。

小报告为 `SMALL_REPORTS.tar.gz` 和 `REPORT_MANIFEST.json`。大型清单仍在数据盘，归档清单明确标记是否装入小包。PCD、逐点sidecar、大清单和权重不提交Git；新源码、配置及小报告可另存新目录，历史报告不覆盖。

## 训练、Clean/Fog验证与Fog Test

完成后自动生成完整AttFuse训练配置 `configs/attfuse_fog.yaml`、`configs/attfuse_clean.yaml`；各自root_dir指向对应train，validate_dir指向对应validate。

```sh
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
nohup env ROCR_VISIBLE_DEVICES=0 /home/cjm/miniconda3/envs/opencood/bin/python -u opencood/tools/train.py --hypes_yaml /data/cjm/datasets/scorw-fog-v1-coveragefix/configs/attfuse_fog.yaml > /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/future_train.log 2>&1 < /dev/null &
```

以上为后续自行训练命令，生成脚本不会启动训练。针对现有GSPR模型配置生成新路径配置，保留其model及训练参数：

```sh
/home/cjm/miniconda3/envs/opencood/bin/python -B -m scorw_dataset.configure --template gspr_attfuse_config.yaml --dataset-root /data/cjm/datasets/scorw-fog-v1-coveragefix --train-condition fog --eval-condition fog --eval-split validate --output /data/cjm/datasets/scorw-fog-v1-coveragefix/configs/gspr_fog.yaml
```

也可将 `configs/dataset_paths.yaml` 中路径合入自己的配置。预生成数据已包含雾，关闭额外的 `weather_augmentation`，保留常规几何训练增强。Fog Test路径为 `fog/test`；原OpenCOOD推理入口通常读取 `validate_dir`，因此要用派生配置把该字段切换到Fog Test，不能只设置未被入口读取的test_dir。

现有冻结AttFuse Clean/Fog验证视图目录带原权重符号链接，可用于native入口：

```sh
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
nohup env ROCR_VISIBLE_DEVICES=0 /home/cjm/miniconda3/envs/opencood/bin/python -u opencood/tools/inference.py --model_dir /data/cjm/datasets/scorw-fog-v1-coveragefix/configs/attfuse_eval_fog --fusion_method intermediate --global_sort_detections > /data/cjm/datasets/logs/scorw-fog-v1-coveragefix/future_native_fog_eval.log 2>&1 < /dev/null &
```

换为 `attfuse_eval_clean` 即Clean验证。原始权重目录不需要修改。对新训练的AttFuse/GSPR，使用该模型自己的权重和配置，派生数据路径视图即可；不要把纯AttFuse权重加载到GSPR结构。

D6采用冻结R3同步project-first适配器、70m及原模型CAV上限、同一Clean GT范围，报告全局分数排序BEV AP30/50/70和TP/FP/FN，None预测按空结果。native dataset可能有基于点数的GT可见性过滤，因此native入口AP和D6固定Clean GT AP需分开标注。公平比较应固定同一权重、GT、帧顺序、随机种子、预/后处理和CAV设置，不能根据Fog Test AP改天气参数。

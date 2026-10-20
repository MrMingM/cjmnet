# V2X-R E0—E3：Mix3 与配对晴天辅助生成监督

本目录交付四组训练、射线标签、逐轮三天气评测、选模、恢复和报告代码。真实数据、Torch/OpenCOOD、HIP 检查和实验都只在服务器执行。尚无服务器运行记录，也没有预填 AP。

E0 训练原生纯 LiDAR AttFuse；E1 增加普通二值采样命中扩散监督；E2 仅监督有保守射线证据的位置；E3 再把正表面分成 ego 支持和仅邻车支持两组求平均。E1—E3 的模型、噪声输入、窗口、预算和 lambda=.1 一致。该系数为预登记起点。推理使用严格匹配官方类的 detector-only 权重。

读取依据：用户本轮提示词、用户提供的 AGENTS.md、AI_CONTEXT.md 最新环境/数据记录，以及旧训练包、D−1C 预测/GT/AP 接口和 D−1A runtime 隔离代码。具体阅读记录、坐标、边界、公式和选择见 protocol.json；实际读取到的只读源码/配置版本见 PINS.json。服务器以源码 SHA256 核验，无 Git、分支或 commit 运行前提。旧历史 AP/Oracle 不用于证明本轮假设。

## 文件与只读依赖

新增文件都在本目录，另有项目根 launch_v2xr_e0e3.sh。未修改旧源码、旧 manifest、checkpoint、结果、AI_CONTEXT.md；没有创建分支、worktree 或提交。

|文件|职责|
|---|---|
|config.json、protocol.json|共享配方、实验定义、几何及损失分母预登记|
|common.py、paths.py、runtime.py|原子写、哈希、输出隔离、官方导入和冻结 GT 核验|
|data.py|同帧 Mix3、独立随机流、背景样本及空协作车适配|
|geometry.py、cache.py|有限射线、小窗口、来源分组、共享标签缓存|
|model.py、losses.py|条件 UNet、相同扩散输入、三组损失|
|training.py、evaluation.py、report.py|逐轮训练/恢复、官方预测/重放、选模和报告|
|preflight.py、smoke.py|一次共享输入核验、真实四组 smoke、显存选择和恢复重放|
|controller.py、worker.py、stop.py、run_all.sh|串行调度、进程隔离、内核锁、安全停止|
|check_static.py、tests_local.py|本地静态及纯 NumPy 合成检查|
|PINS.json、source_snapshot.json|只读依赖和本次新增源码哈希|

保留同级只读目录 **v2xr_train_attfuse/**、**v2xr_protocol_audit/**、**v2xr_official_repro/** 的完整原版本，含原 source_snapshot、PINS、TRAIN_DATA_ACCEPTANCE、UPSTREAM 及 upstream。准确文件哈希在本目录 PINS.json；运行时不会更新旧门禁。保留服务器上已验收的 train 提取记录、原 inventory 接受证据、D−1A/B/C 结果、官方 model/config 与 runtime/native 扩展。无需重新解压或安装依赖。

Windows 本地在项目根同步新增实现（将 SERVER 改成 SSH 目标）：

```powershell
scp -r v2xr_raygen_e0e3 launch_v2xr_e0e3.sh SERVER:/home/cjm/OpenCOOD-main/cjmnet/
```

若服务器缺少上述三个只读依赖目录，将完整目录一并同步；保留原字节，避免文本工具转换旧源码换行。新增源码统一 LF。不要把原数据或大缓存复制进代码目录。

## 首次后台启动

服务器执行，GPU=0 改成你选定的空闲物理卡号：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
GPU=0 sh launch_v2xr_e0e3.sh
```

默认 BACKGROUND=1，入口内部使用 nohup，输出绝对 OUTPUT_ROOT、PID 和日志。首次 OUTPUT_ROOT 自动为数据盘时间戳目录，也可显式指定全新目录。无需额外套 nohup。MODE=all 顺序：预检 → 合成检查 → 局部真实标签与显存检查 → 四组两轮短训练/三天气 O/O*/F/恢复重放/报告 → 全量天气解码核验 → 共享完整 train 射线及全部 epoch 窗口标签 → E0 完整训练/逐轮三天气评测/选模/报告 → E1 → E2 → E3 → 总报告。任何组的 AP 不提升都继续后续组。

只设置 ROCR_VISIBLE_DEVICES，脚本清除 HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES。每次只使用指定单卡；没有自动选卡。默认 FP32，AMP=1 明确报错。没有蒸馏、GSPR、雷达/图像模型输入、时序、定位噪声或点云回灌。

|环境变量|默认值或规则|
|---|---|
|PROJECT_ROOT|/home/cjm/OpenCOOD-main/cjmnet|
|PYTHON|/home/cjm/miniconda3/envs/opencood/bin/python|
|GPU|all/smoke 必填，单个物理整数卡号|
|MODE|all；另有 preflight、labels、smoke、report|
|SEED、EPOCHS、AMP|20261019、20、0；SEED 必须 uint32，启动控制器之前设置 PYTHONHASHSEED|
|BACKGROUND、RESUME|1、0|
|OUTPUT_ROOT|首次生成数据盘时间戳；恢复必须显式指定旧目录|
|TRAIN_ROOT|/data/cjm/datasets/v2xr-official-train-v2；也接受该目录下 train/|
|VALIDATE_ROOT|/data/cjm/datasets/v2xr-official-validate-v1/validate|
|PAIRED_VALIDATE_ROOT|/data/cjm/datasets/v2xr-lidar-weather-validate-v1|
|RUNTIME_ROOT|/data/cjm/datasets/v2xr-official-runtime-v1|
|MODEL_DIR|/data/cjm/datasets/logs/v2xr-official-attfuse-model-v1|
|D1A_ROOT、GT_AUDIT_ROOT、D1C_ROOT|沿用旧训练包的官方默认目录，可显式覆盖；必须匹配冻结证据|
|TMPDIR|/data/cjm/datasets/tmp/v2xr_raygen_e0e3|

不同 SEED/EPOCHS 可以在首次启动前显式设定并冻结；恢复不可改变。默认20轮属于首轮受控预算。可用 microbatch8/4/2/1 和累积1/2/4/8，保持有效 batch8；E3 预检测到第一个可用配置后所有组统一，BN 使用相同 microbatch 统计。正式训练 OOM 立即失败，不改变单组配方。

## 调试、恢复和日志

前台短调试：

```sh
GPU=0 MODE=smoke BACKGROUND=0 OUTPUT_ROOT=/data/cjm/datasets/logs/raygen-smoke-debug sh launch_v2xr_e0e3.sh
```

仅 smoke 后台（自动生成目录）：

```sh
GPU=0 MODE=smoke sh launch_v2xr_e0e3.sh
```

同目录恢复（RUN 替换成第一次打印的绝对目录，保留原 SEED/EPOCHS/路径及选定 GPU）：

```sh
RUN=/data/cjm/datasets/logs/实际运行目录
GPU=0 MODE=all RESUME=1 OUTPUT_ROOT="$RUN" sh launch_v2xr_e0e3.sh
```

仅 CPU 准备完整标签，也可能很久，默认后台运行：

```sh
MODE=labels OUTPUT_ROOT=/data/cjm/datasets/logs/raygen-labels-prepared sh launch_v2xr_e0e3.sh
```

标签模式会进行路径/环境/输入预检及完整天气内容核验，使用 CPU Torch/原生 ROI 操作，没有 GPU 前向。随后同 OUTPUT_ROOT 加 RESUME=1、MODE=all、GPU 即可继续；显卡身份在首次 capacity 阶段冻结。preflight 模式同理。标签计算期间无需数据加载到 GPU。

只重建报告，不需要原数据或 GPU，读取输出目录内的已提交证据：

```sh
MODE=report RESUME=1 OUTPUT_ROOT="$RUN" sh launch_v2xr_e0e3.sh
```

查看统一日志和某阶段进度：

```sh
tail -f "$RUN/launcher.log"
cat "$RUN/current_stage.json"
cat "$RUN/stage_status.json"
tail -f "$RUN/logs/E1_train_01.log"
```

缓存进度位于 labels/progress.json、labels/window_progress.json；天气解码进度在 preflight/content_progress.json；逐轮进度含当前 sample UID、曝光数和更新数。控制器每20秒报告阶段耗时和子日志位置，子日志打印实际样本进度。

安全停止：

```sh
/home/cjm/miniconda3/envs/opencood/bin/python -B v2xr_raygen_e0e3/stop.py "$RUN"
```

停止工具核对 /proc 的启动身份和完整控制器参数后才发 TERM。控制器只转发给自己创建的子进程组，并保留恢复文件。锁使用 fcntl.flock；PID 文本不能代替锁。若旧控制器被 kill -9 而留下子进程，worker 内核锁会阻止第二个 worker 并发写入；先确认该任务原子进程退出再恢复。不要按进程名批量 kill。

## 数据、标签与公平性

沿用已接受的8083协作帧、32有数据场景、92 scene-CAV、各天气24088单车帧。默认路径只是假设，preflight 必须解析实际 manifest、原8083接受证据和三天气配对。validate829协作帧；F 必须与现有 D−1C 清单逐 UID/顺序/ID/几何一致，开发运行前冻结。禁止访问正式 test。

每协作帧按1/3概率抽 Clean/Fog/Snow，同帧所有车同天气。顺序、天气、点顺序 shuffle、窗口、扩散各用独立稳定哈希随机流；生成头不会挪动其他组的输入。每轮共同丢弃末3帧，曝光8080，默认1010次有效更新。保留零GT帧作为背景，原生非空GT的天气/FOV/点命中筛选规则不变。空CAV使用固定全零稀疏体素占位保持原生 scatter 的 batch ID，原投影点仍为空，不能产生标签。

原生 loader 没有调用配置中的世界翻转/旋转/缩放。本轮四组统一使用其实际恒等几何；发现增强调用会报错。缓存保持未裁断的原点-回波关系，将两者用同一原生位姿投影一次，再查询窗口；不使用天气/晴天点逐行对应关系。生成窗口的 GT 来自 train Clean 原生最终目标清单，是额外训练监督，检测GT仍来自当前天气。空CAV的稀疏占位在scatter之前显式将柱特征置零，不把占位当真实点过滤或补成回波；该适配没有权重参数。

O_raw 为实际端点；E_raw 只取最近回波前的有限射线，距终点留.8m边界，最短穿格.1m；所有空域格角点须位于允许范围及 FOV。O/E 冲突全部归 U。没有记录的射线、回波后方不确认空域。角度量化1e-5rad只用于合并记录回波，缺少 ring/束ID 和多径真值的边界已在 protocol 中说明。这些是局部射线采样近似标签。

窗口32×32×8，固定12.8×12.8×4m，z范围[-3,1]，水平中心在GT附近抖动±1.6m，最多8个；不按GT尺寸或朝向变换。来源分组 ego 优先，ego与邻车同时支持只算一次。x0 在 O_raw 为+1，其他均-1；E2/E3 的 U 仅忽略损失，不改变输入值或通道。

每个非空类先组内均值，两类各占.5；缺类时在剩余类重归一化。E3正类内的非空来源组平分正类总权重。全U窗口辅助loss=0，检测样本保留。详细公式在 protocol.json，并由纯NumPy测试覆盖。

缓存：每样本 NPZ 保存完整允许回波及原点、最近回波标记、来源和覆盖所有抖动的目标候选射线。逐样本/epoch预生成4个packbits掩码和窗口中心，E1—E3共享，避免每组重复射线追踪。没有全场景稠密3D网格。预估原始数组约50字节/回波加候选索引，4掩码每窗口4096字节，压缩后实际用量由 manifest 报告。部分帧和部分frame-epoch可分别恢复。输入提取已有CRC/验收结果直接复用，新需要解码的天气PCD共享核验一次；以后只比较文件身份/元数据，变化就停止，要求新输出重新核验。

## 验证、选模与产物

每轮保存不可变 detector.pth 和完整 resume.pth，训练进程退出释放显存后，独立进程对完整829帧三天气各推理一次。复用 v2xr_protocol_audit.predict.generate/evaluate.replay，未调用旧 Clean-only 选模或 MODE=final。三协议共享同一份预测：O保持None跳帧；O*与F保留None帧GT；F为已冻结Clean目标。指标为 BEV AP30/50/70，VOC2010、global_sort=False，严格score>.20、NMS=.15。推理只读当前天气点云，生成模型和训练标签不导入评测进程。

每组全部epoch完成后，找最高Clean O AP70，保留与最高值相差≤.005的epoch，选择Fog/Snow O AP70均值最高者，未舍入比较，完全并列选最早。报告包括三天气三协议完整AP、分母、框数、None/empty、E1−E0/E2−E1/E3−E2/E3−E0百分点差值。只有O改善而F未改善不能称固定目标检测能力改善。没有性能停机门槛。

|位置|主要产物|
|---|---|
|根目录|request/contract/protocol/config/source_snapshot/stage_status/data_check/label_check/execution/cost_estimate.json|
|preflight/|实际训练manifest、验证证据、输入元数据、一次解码日志、smoke样本清单|
|labels/|不可变共享契约、逐样本原始射线、逐frame-epoch窗口掩码、manifest及进度|
|smoke/、smoke_replay/、probe/|四组两轮短训练/三天气/报告、从第一轮恢复的第二轮重放、显存检查；全部明确smoke，不可作为正式complete|
|groups/E0…E3/|config、初始化tensor哈希和初始权重、参数量、完整history、selected、summary、RESULTS.md|
|groups/组/epochs/NN/|sampling、训练进度、detector/resume.pth、train_complete；evaluation/下三天气逐帧预测和三协议结果|
|根目录完成后|RESULTS_E0_E3.md、results.csv、summary.json、complete.json|

本epoch训练提交与评测提交分开。评测失败时不重训已提交权重；中途训练从最近训练epoch边界重跑，记录RNG、optimizer/scheduler、criterion、sampler、global step 和累积位置0。没有逐step恢复承诺。四组smoke会从真实epoch1状态重跑epoch2并比较模型/辅助头/Adam/scheduler/RNG及输入日程；HIP恢复容差rtol1e-4、atol1e-5，超过就阻断。E0的首次smoke评测还会在Clean缓存提交后有意退出43，再恢复Fog/Snow并核对训练提交、权重、Clean缓存均未改变。该故障注入只在独立smoke目录执行。普通原生前向等价检查容差为rtol1e-5、atol1e-6。

## 阻断与尚待服务器确认

必要数据缺失、计数/UID/配对/接受证据不符、F不匹配、源码/配置/输入发生变化、混入其他OpenCOOD/native模块、环境不符、同输出并发锁、磁盘不足、标签缺必要O/E/U/ego/仅邻车类别、NaN/Inf、梯度断开、恢复漂移、最终评测覆盖不完整或训练OOM，都会停止后续训练。failure.json保留阶段/组/epoch/子日志异常；当前sample UID在对应progress和异常堆栈。容量预检最多尝试8/4/2/1，只对明确torch OOM换统一microbatch；其他错误立即停止。

服务器尚待验证：真实路径/完整数据内容、原生导入/空输入兼容、普通样本和原生模型等价、各组输入与初始化一致、真实标签类别、扩散损失到融合特征/参数的梯度、严格detector导出、HIP显存/数值、真实恢复重放、三天气O/O*/F及报告。显存预检是基于实测重样本的工程检查，不保证穷尽所有极端帧；正式期任何OOM会留下失败并停止。

本地检查方式（均为静态或纯NumPy合成检查）：

```sh
python -B v2xr_raygen_e0e3/check_static.py
python -B -m v2xr_raygen_e0e3.tests_local
sh -n launch_v2xr_e0e3.sh
sh -n v2xr_raygen_e0e3/run_all.sh
```

本地实际检查记录见 LOCAL_VALIDATION.md。此实现借鉴生成辅助监督思想，不称完整 DiGS 复现；单种子首轮结果不代表SOTA或已收敛。本次不扩展三种子、正式test或其他模块。

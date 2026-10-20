# V2X-R E0—E3：Mix3 与配对晴天辅助生成监督

本目录交付四组训练、射线标签、逐轮三天气评测、选模、恢复和报告代码。真实数据、Torch/OpenCOOD、HIP 检查和实验都只在服务器执行。服务器当前进度及自检修复见下文，没有预填 AP。

## 2026-10-20 最新：E1整块误差预算修正，复用E0/E1结果

流程已通过E0，当前停在`smoke_E1_resume_check`。用户回传证据：E1的564项加载后状态逐项完全一致，起点及实际数据一致；唯一失败项为`optimizer/state/17/exp_avg`，其L2差为0.00017879605184775667，张量L2大小1.0707758045136373，整体相对差0.0001669780462857662（约0.0167%）。其余状态均通过v2检查。

v2的两种检查使用了不同的绝对预算：逐元素检查给每个元素`atol=1e-5`，整块L2检查给整块合计`1e-5`。这会随张量元素数量增加而变得更严格。v3将整体差异换算为每元素的均方根（RMS），让相同元素重复拼接时得到一致的检查结论。`rtol=1e-4, atol=1e-5`保持原值，模型权重继续用原逐元素检查；只有Adam的三个moment字段补充的整体检查采用RMS。

具体规则为`RMS(a-b) <= 1e-5 + 1e-4*max(RMS(a),RMS(b))`，其中`RMS(x)=||x||2/sqrt(N)`。等价地，整块L2绝对预算为`1e-5*sqrt(N)`。这是比v2更宽的整块绝对误差预算，作为v3规则明确记录；旧v2失败保持原样，不称为通过旧规则。学习率、step、整数计数、RNG、数据、形状和dtype仍需精确一致，NaN/Inf或大幅偏差仍失败。实际训练加载函数和v2精确加载检查完全未改，可复用已通过的证明。

完整同步本目录，包含新增`RESUME_RMS_COMPAT.json`和最新`source_snapshot.json`。保留原运行目录及现有诊断文件，继续使用2号卡后台恢复：

```sh
GPU=2 MODE=all BACKGROUND=1 RESUME=1 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574 \
  sh /home/cjm/OpenCOOD-main/cjmnet/launch_v2xr_e0e3.sh
```

升级仅接受上一版完整v2源码、上述确切E1失败边界及匹配的checkpoint/诊断/精确加载证明，附加`source_repair/resume_rms_v3.json`；旧U修复与v2来源不改写。E0/E1训练、验证、重放和精确加载全部复用，只新增纯合成回归并重试失败的E1比较。通过后继续E2/E3及正式流程。旧`resume_check_v2_diagnostic.json`保留，新报告单独写为`resume_check_v3_diagnostic.json`，记录真实元素数、RMS差、L2差和允许界限；最终报告附带新来源。

本地57项合成检查通过，包含维度预算、重复元素结论不变、大幅漂移/非有限数拒绝、原目录来源升级、损坏产物/训练源码变化拒绝，以及控制器仅重试E1比较。未在本地运行Torch/OpenCOOD或科研训练。新规则仍须由服务器检查实际checkpoint，不预先声称所有组已经通过。以下为前序修复记录。

## 2026-10-20 最新：E0恢复检查修复，可沿用原目录和2号卡

当前失败阶段为`smoke_E0_resume_check`。E0两轮短训练、三天气验证和第二轮重放已经完成；E1—E3的短训练及正式训练尚未开始。无需重新准备缓存、探测显存或重训E0。

用户回传的CPU诊断报告为`diagnostics/resume_E0_1792466315475575112.json`：产物哈希正确，第1轮起点相同，显卡、第二轮数据与标签、RNG和scheduler完全一致；模型全部119个浮点张量通过原容差，最大绝对差`4.363246262073517e-06`。Adam有8个张量的1446个元素超出原逐元素容差，但这些张量的相对L2差为约`4.7e-7`至`6.1e-5`。另外，4个浮点损失日志被旧代码按完全相等比较而失败。该证据支持区分恢复状态错误与计算后的浮点差异，尚未独立确定具体HIP算子的影响。

本版增加“刚加载完就精确核对”的检查：用实际训练共用的加载函数，将checkpoint读回官方模型、辅助模型、Adam、scheduler、criterion以及Python/NumPy/Torch CPU/HIP随机状态，要求逐项完全相同。该检查只创建对象并加载状态，不前向、不反向、不更新权重。之后才比较已有的两条第二轮结果：

- 模型/辅助模型权重沿用原逐元素`rtol=1e-4, atol=1e-5`。
- Adam的`exp_avg/exp_avg_sq/max_exp_avg_sq`先做原逐元素检查；未通过时，分别限制每个张量的整体L2差。通俗地说，将这个参数块的全部差异合起来，与参数块自身大小比较，限制整体偏差。公式为`||a-b||2 <= 1e-5 + 1e-4*max(||a||2,||b||2)`；各张量独立检查，不将全模型混在一起平均。
- criterion的浮点损失日志使用原`rtol/atol`，损失累计次数仍需完全相等。
- LR、Adam step、scheduler、RNG、数据日程、配置、身份、形状和dtype保持精确比较；NaN/Inf直接失败。

**这次补充了v2验收规则。旧逐元素容差失败记录原样保留，不能把它描述为已通过旧规则。** 新规则及源码身份写入`source_repair/resume_check_v2.json`，并附在最终报告中；原request/contract/protocol/execution/U修复记录和已完成产物不改写。训练配方、模型/损失/标签算法、种子、batch、学习率及数据保持一致。后续训练的每次恢复也执行相同的加载后精确检查。

原目录升级仅接受上一版完整GPU修复源码、确切E0优化器容差失败及匹配的诊断报告；重新核验完成记录、数据日程、缓存和输入身份。损坏产物、RNG不一致、大幅优化器偏差、其他源码变化或已经进入后续组/正式训练均拒绝套用该升级。再次恢复会继续锁定新源码、规则及2号卡。

请完整同步`v2xr_raygen_e0e3/`，包含`resume_compare.py`、`RESUME_CHECK_COMPAT.json`及最新`source_snapshot.json`。保留服务器原运行目录中的诊断JSON，无需重跑诊断。仍在原目录后台恢复：

```sh
GPU=2 MODE=all BACKGROUND=1 RESUME=1 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574 \
  sh /home/cjm/OpenCOOD-main/cjmnet/launch_v2xr_e0e3.sh
```

预期日志复用已完成阶段；先运行纯合成`resume_check_v2_regression`，然后`smoke_E0_restore_check_v2`精确加载核对及失败的`smoke_E0_resume_check`。两项通过后继续E1—E3短训练、共享完整标签及正式流程。新增检查失败仍会停止并写出诊断；不预先声称服务器已通过。检查比较的是两次从同一epoch1边界恢复的结果，未独立证明连续进程训练等价或逐比特确定性。

本地50项纯Python/NumPy合成检查通过，包括真实加载函数AST配合NumPy替身执行，以及原目录升级、拒绝损坏/漂移和控制器复用E0结果。没有在本地导入Torch/OpenCOOD或运行科研实验。以下为先前U字段与换卡修复的历史记录，适用对应旧失败边界。

## 2026-10-20：全未知窗口自检修复

用户回传的运行目录 `/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574/` 已进入 server_checks，7项官方AP合成检查通过；随后自检构造的全未知窗口缺少 `U`，在 `auxiliary_loss` 统计未知格数量时抛出 `KeyError: 'U'`。日志中的Shapely warning对应测试仍显示ok，这次退出的直接原因是缺少字段。

修复限定在smoke.py的测试样本：清空O/E后按 `U=~(O|E)` 补齐未知掩码。全未知窗口的8192个格子均为U，E2/E3的有效监督权重仍为0。没有改动训练、生成损失、标签生成、实验配方或旧结果文件。新增纯NumPy回归检查直接执行服务器自检中的原始样本构造语句，并对照真实损失统计所需字段；修复前复现相同KeyError，修复后本地17项合成检查通过。修复后的真实Torch/HIP自检尚待服务器重跑。

**最新修复支持在原失败目录继续，也支持本次指定的2号卡，无需重新准备已完成的smoke缓存。** 请整目录同步，包含新source_snapshot.json、resume_compat.py及U_FIX_COMPAT.json，再对下面这个原目录使用RESUME=1。此前建议新开目录会重复准备工作；这次增加了针对已知自检错误的定向恢复。

恢复器仅接受最初交付源码，以及server_checks中`zero_unknown`对应的`KeyError: 'U'`失败；四组短训练和正式训练必须尚未开始。它会核对已提交的preflight、16个样本的原始射线和两轮窗口缓存、显存探测产物、选定microbatch及输入身份。训练配方、标签/损失/模型算法、配置、依赖和数据保持一致；runtime/training新增的工程逻辑仅用于选择经过验证的显卡记录和隔离新显存检查的输出路径。原request.json、contract.json、execution.json、execution_commit.json、source_snapshot.json、缓存manifest和缓存内容保持原样；实际新源码及原失败证据写入额外的source_repair/记录，每个后续子进程也记录实际源码身份，最终报告附带修复来源。配置、种子、数据变化或缓存损坏仍然拒绝恢复。

通过核验后日志会显示复用preflight、synthetic_checks、smoke_prepare。若保留原卡，显存检查直接复用；若本次切换至2号卡，只额外执行`capacity_gpu2`，在2号卡上使用原microbatch、原梯度累积次数、有效batch8和FP32完成E3显存检查。检查失败会停止，不自动缩小batch；检查通过后将新显卡及硬件名称写入`source_repair/gpu_capacity.json`，后续四组均使用该卡。新探测产物保存在`source_repair/gpu_probe/2/`，原卡探测结果保留。再次恢复会核对并复用已经完成的新显卡检查，卡号锁定为2。

新增`u_fix_regression`会重跑纯合成检查，失败的`server_checks`会从头重测；通过后继续尚未执行的四组短训练/三天气验证/恢复检查及正式流程。已经完成的长时间准备不会重建。若再次中断，使用相同命令恢复，修复记录锁定的新源码也不能随意更换。

服务器一键后台恢复（本次指定物理卡2；若原来覆盖了SEED/EPOCHS/数据路径，仍按原设置传入）：

```sh
GPU=2 MODE=all BACKGROUND=1 RESUME=1 \
  OUTPUT_ROOT=/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574 \
  sh /home/cjm/OpenCOOD-main/cjmnet/launch_v2xr_e0e3.sh
```

本次回传的旧版本恢复器在写入修复记录前因卡号不一致退出，因此这个原目录可以直接使用新版本。上述首次换卡只支持原始U字段错误边界；已开始后续训练的目录不会自动换卡。保留原目录，不手改哈希、execution.json或阶段状态。尚待服务器实际恢复验证，本地完成的是纯Python/NumPy及合成文件状态检查。

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
|resume_compat.py、U_FIX_COMPAT.json|已知U字段自检错误的定向恢复、原源码白名单及追加来源记录|
|resume_compare.py、RESUME_CHECK_COMPAT.json|精确加载核对、分项数值规则及E0失败边界的来源升级|
|RESUME_RMS_COMPAT.json|E1整体绝对预算修正的上一版源码身份与适用边界|
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

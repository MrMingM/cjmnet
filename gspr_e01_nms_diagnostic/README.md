# E01 smoke 第2轮预测框除零诊断

本工具针对用户回传的 `smoke_T1_C_train` → `epochs/02/clean_ap_6b8abd…` 失败。

核心意思：模型生成的两个预测框，在去掉重复框的 NMS 环节计算合并面积时得到了0，原版 IoU 计算发生除零。NMS 是筛掉重复预测框的步骤；它在 AP 评分之前执行。这段报错还不能区分框尺寸已经变成0、尺寸太小导致角点重合、坐标变换异常等具体来源。

这次只复用已经保存的第2轮检测权重，观察第一个验证帧。不训练、不创建优化器、不更改学习率，不删框、不裁剪回归值、不改变 IoU/NMS/AP，不修改旧源码、合同或完成结果，也不自动继续原训练。

默认模式及 `TRACE_LAYERS=1` 为原计算的被动观察。用户于2026-10-23明确选择“做独立 BN 对照诊断”后新增 `BN_COMPARE=1`：仅其 `current_batch` 诊断分支临时改变 BN 的统计量来源，明确标记计算干预、禁止计入 AP；详见下文。

## 上传

将整个 `gspr_e01_nms_diagnostic/` 放到服务器 `/home/cjm/OpenCOOD-main/cjmnet/` 下，与 `gspr_e01/` 并列。**不要把本目录的文件放进旧 `gspr_e01/`**，否则旧目录的源码快照会变化，影响其恢复身份。

原 `gspr_e01/`、E0、冻结训练和评分包保持原样。本工具检查旧目录快照和原运行 `request.json` 的源码身份；不需要 Git。权重默认使用报错日志中的 SHA256：

`6b8abd19b6fd7db5f6e53e90fda7b628c68452ce00991de71d9de5bf9ac7f49d`

## 后台诊断命令

确认原控制器已经退出后复制执行；它会检查原控制器、训练 worker 和 GPU 租约。仍有旧进程时会退出，不删除锁或终止旧实验。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
RUN=/data/cjm/datasets/logs/gspr-e01-v1
OUT=/data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/$(date -u +%Y%m%dT%H%M%SZ)_$$
mkdir -p "$OUT"
RUN_ROOT="$RUN" OUTPUT_ROOT="$OUT" nohup sh gspr_e01_nms_diagnostic/run_server.sh \
  > "$OUT/launcher.log" 2>&1 < /dev/null &
printf 'PID=%s\nOUT=%s\nLOG=%s\n' "$!" "$OUT" "$OUT/launcher.log"
tail -f "$OUT/launcher.log"
```

物理卡号自动读取原 `request.json` 的 `E1_gpu`，保持原串行/并行启动时的真实选择；不猜卡号。还原其 Python hash seed 和线程环境。若设置 `GPU`，必须与原 E1 卡号一致。

固定 Python 默认为 `/home/cjm/miniconda3/envs/opencood/bin/python`，必须与原实验一致。可通过 `PYTHON` 指定同一个解释器。`CHECKPOINT` 可显式指定原 `epochs/02/checkpoint_<实际SHA>.pth`；不接受其他模型、初始权重、只按最新时间猜出的文件。

旧来源和 validation 输入有完整 SHA 核验，开始前可能需要几分钟；每20秒有心跳。DataLoader 保持原本的设置和12帧前缀，但模型只执行第一个帧的前向，然后停止，不计算 AP。不会跑完整829帧，更不会训练。

### 2026-10-22 信号处理修正

旧版将 SIGHUP、SIGINT、SIGTERM 都转成 `KeyboardInterrupt`，还会覆盖 `nohup` 或后台 shell 原本的忽略设置。因此，旧日志中的 `Diagnostic interrupted` 只能证明收到中断信号，不能确定是哪一种，也不能说明第一帧已复现除零。

本版保留启动时已忽略的信号；其他终止信号仍会停止本次诊断子进程，等待它退出，再释放租约。明确记录 `INTERRUPTED signal=...` 及独立 `diagnostic_status.json`，避免把中断伪装成诊断完成。不会忽略明确的 SIGTERM 来强行持续运行。

重新同步整个目录（含 `source_snapshot.json`），保持原中断输出不变，使用新的诊断目录。若不方便复制多行变量命令，可逐条执行下面两条**各自单行**的命令。默认生成新的输出目录，并在日志中打印其路径；下列日志路径的父目录已由上次诊断建立。

```sh
RUN_ROOT=/data/cjm/datasets/logs/gspr-e01-v1 OUTPUT_ROOT= nohup sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e01_nms_diagnostic/run_server.sh >> /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/signal_fix_launcher.log 2>&1 < /dev/null &
```

```sh
tail -f /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/signal_fix_launcher.log
```

`RUNNING first_frame` 是每20秒的存活提示，时间到了仍会继续等待，并非20秒超时退出。查看日志的 `tail -f` 可用 Ctrl+C 退出；后台诊断继续运行。若仍收到中断，回传新的 `INTERRUPTED signal=...`、状态文件及 `first_frame.log`。

### 2026-10-22 临时通信路径修正

用户随后回传的 `first_frame.log` 显示 `OSError: AF_UNIX path too long`。这是数据加载子进程传递数据时的通信错误：旧版将 `TMPDIR` 设为较长的输出目录再加 `/tmp`，Python 还会在下面加 `pymp-*` 和 `listener-*`，最终无法创建 socket。它解释了为什么进程持续等待；尚不能据此判断原模型预测框除零的原因。

本版用 `tempfile.TemporaryDirectory(prefix='gnms-', dir='/tmp')` 为每次诊断创建短且独立的私有目录，同时设置子进程的 `TMPDIR`、`TEMP`、`TMP`，覆盖继承来的长路径。权重、数据和正式诊断报告仍在原定位置，数据加载 worker 数量、共享策略、随机种子和全部评分设置不变。诊断子进程退出后才清理本次独有的临时目录；不清理任何旧输出。

加载 Torch/OpenCOOD 和模型前，子进程通过标准库 `multiprocessing.connection.Listener(family='AF_UNIX')` 实际测试一次 socket 创建，核验 Python 真实选用的临时目录和地址长度，保存 `ipc_preflight.json`。失败时直接退出并保留错误，避免由数据加载队列线程报错后主进程一直等待。日志新增 `IPC_TMPDIR=` 和 `IPC_PREFLIGHT=complete socket_path_bytes=...`。

Linux 路径型 UNIX socket 的 `sun_path` 为108字节，程序为结尾空字符保留1字节，按文件系统编码后的字节数检查。依据：[Linux unix(7)](https://man7.org/linux/man-pages/man7/unix.7.html)；临时目录选择与缓存见 [Python 3.10 tempfile](https://docs.python.org/3.10/library/tempfile.html#tempfile.gettempdir)。

本包包含上面的信号修正。同步整个新目录（含源码快照）后，只需逐条执行下面两条单行命令；不要只设置外层 `TMPDIR` 来运行旧包，旧包会将它覆盖。

```sh
RUN_ROOT=/data/cjm/datasets/logs/gspr-e01-v1 OUTPUT_ROOT= nohup sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e01_nms_diagnostic/run_server.sh >> /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/ipc_fix_launcher.log 2>&1 < /dev/null &
```

```sh
tail -f /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/ipc_fix_launcher.log
```

使用默认新诊断目录，不覆盖中断记录。仍有旧诊断占用锁时，新诊断会拒绝启动；不会删除锁或自动杀死其他进程。若旧进程仍在运行，先回传其 PID 和命令确认身份。

### 2026-10-23 被动逐层定位

用户回传的服务器报告已确认 `outcome=zero_union_reproduced`，第一帧为 Clean 的 `2024_06_28_17_45_09/000060`。进入 NMS 的30215个预测框中有48个零面积框。某个原生回归值为 `-171.0901641845703`，对应长度解码为0；另一些正的极小尺寸在生成角点时也已丢失面积。分类原始输出绝对最大约767万、回归输出约1095万，异常出现在后处理之前。

checkpoint 未发现 NaN/Inf，BN 保存的方差为正且不为0，计数为8。训练日志仅显示每轮第一批损失从113.051437降到15.547859，第1轮 Clean O AP70为0。这些事实不足以认定优化器发散、BN 已损坏或训练一定正常。官方 BN 的 momentum 为0.01，只经历8次更新时其运行统计量可能仍未稳定；还需要观察具体数值在哪一步放大。

可用 `TRACE_LAYERS=1` 开启新增的被动记录。它沿用原第2轮权重、首帧和原前向函数，观察卷积、线性层、BN、注意力融合和主要特征层，保存各次输出的范围。对 BN 记录本帧通道均值/总体方差与保存统计量的差距，并在第0个批次、每通道最多8个位置以 CPU float64 抽样核对原 eval 公式。抽样不覆盖整个算子，不能单独证明 GPU 完全正确或错误。前向前后逐字节哈希核对 BN 的 running_mean、running_var、num_batches_tracked；缺失证据时明确返回 null。

观察器不切到训练模式、不使用当前批次统计量替换原计算、不重置或校准 BN、不更改权重、不额外执行模型前向、不创建优化器。钩子返回 None，保留原返回对象，观察失败只记录错误。阈值100/1000/1000000仅用于报告首次观察到的量级，不作为训练、AP或恢复门禁。被动归约和 CPU 读数可能改变执行时序，不能据此保证逐位复现。

本包包含前面的信号和短路径修正。覆盖整个诊断目录（含源码快照），保留原 E01 和旧报告，逐条执行两条单行命令：

```sh
TRACE_LAYERS=1 RUN_ROOT=/data/cjm/datasets/logs/gspr-e01-v1 OUTPUT_ROOT= nohup sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e01_nms_diagnostic/run_server.sh >> /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/layer_trace_launcher.log 2>&1 < /dev/null &
```

```sh
tail -f /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/layer_trace_launcher.log
```

本轮不用重新训练。默认自动分配新的输出目录；正式结果、已完成第1轮和已有诊断均保留。新增 `layer_trace.json` 及终端 `LAYER_SUMMARY=`，和原 `SUMMARY=` 一起提供。`layer_trace.json` 包含全部逐层记录与观察错误；终端只汇总首次跨越量级的层和主要 BN 差距。

### 2026-10-23 经用户确认的独立 BN 对照

后续服务器逐层报告已回传：54次层调用、23次 BN 观察，抽样公式检查无失败，BN 保存状态在前向前后完全相同。卷积输出首次过100、1000、100万的层分别为 `backbone.blocks.1.4`、`backbone.blocks.1.16`、`backbone.blocks.2.16`；`backbone.blocks.2.25` 最高约6430万。某层当前特征标准差与保存统计量对应尺度之比约510万，这个比值描述统计不匹配，不能解释成单个 BN 操作把输入乘了510万倍。

官方 momentum=0.01 且仅更新8批，初始运行统计量的递推系数仍为 `0.99**8 ≈ 92.27%`。这为短 smoke 的历史统计尚未稳定提供了线索；单帧被动观察尚未证明唯一根因，也不能排除其他计算问题。递推规则与 eval 使用保存统计量的行为见 [PyTorch BN 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html)。

用户已确认只做独立对照。本模式顺序运行两个新子进程，各读取同一份原始 checkpoint，保持同一卡号、原输入/数据加载/线程/随机种子、模型 eval 和原 `torch.no_grad()`：

- `native/`：保留原 BN 的历史统计量。
- `current_batch/`：仅对临时模型实例中的 BN forward 使用 `functional.batch_norm` 的当前批次均值/总体方差，传入 `running_mean=None`、`running_var=None` 和原 gamma/beta/eps。该函数的 `training=True` 参数选择当前批次统计量；没有反向、优化器、模型 train 模式或状态更新。不会写回新的模型权重。

两路共用同一组原控制器/worker/GPU租约，按顺序执行，各只有一个模型前向；没有额外校准遍历。对照前逐 tensor 核验实际模型精确等于 checkpoint，前后哈希核验全部参数和缓冲区、BN计数器不变。模型上的 forward 只在进程内临时替换，退出时恢复原有实例描述。

配对报告严格要求两路 checkpoint/配置/代码/原运行身份、物理 GPU、线程设置、首帧身份与原后处理参数相同，processed LiDAR、record_len、坐标变换矩阵的 dtype/shape/字节哈希完全相同，起始模型状态哈希相同，前向后原状态不变。BN公式核验分别采用原历史统计量和当前批次统计量，后者不会误标成原 eval 公式通过。任何核验缺失、观察失败或两路身份不符，均不宣布配对完成。

`current_batch/report.json` 明确标记 `protocol_changes=true`、`BN_current_batch_counterfactual=true`、`eligible_for_AP_or_selection=false`；评分/GT/NMS/AP函数没有改动。两路均在首帧完成或捕获原异常后退出，原冻结生成器不会接受预测缓存或计算 AP。即使异常消失，也不能用该模型或结果替代正式评测，不会自动更改原 smoke 的门槛或完成状态。

同步整个诊断目录（含新增 `nms_bn.py`、源码快照），逐条执行下面两条单行命令。默认自动创建新的输出目录，不覆盖已有被动诊断或正式实验。

```sh
BN_COMPARE=1 RUN_ROOT=/data/cjm/datasets/logs/gspr-e01-v1 OUTPUT_ROOT= nohup sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e01_nms_diagnostic/run_server.sh >> /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/bn_compare_launcher.log 2>&1 < /dev/null &
```

```sh
tail -f /data/cjm/datasets/logs/gspr-e01-nms-diagnostic-v1/bn_compare_launcher.log
```

会先后看到 `START BN_COMPARE mode=native/current_batch`。本模式包含两次原数据身份核验与两次首帧前向，耗时会超过单次观察，运行期间每20秒有心跳。结果保存在：

- `comparison_request.json`、`comparison_status.json`。
- `native/`、`current_batch/`：各自 request、IPC自检、原始数组、逐层记录、诊断状态和 report。
- `bn_comparison.json`：逐项配对核验、两路原始分类/回归范围、输出量级比值、NMS实际结果及全部输出哈希。

终端汇总为 `BN_COMPARISON=`，路径为 `BN_COMPARISON_REPORT=`。回传这两行即可先判断方向。没有执行 NMS 时 `NMS_geometry` 为 null，会保留原 None/空预测信息，不能把它当作实际测到了0个零面积框。任何结果都只说明这个首帧在这次干预下的表现；不能据此宣布正式修复、AP提高或所有帧安全。新增对照流程仍需服务器验证。

## 运行后提供这些输出

```sh
cat "$OUT/launcher.log"
cat "$OUT/request.json"
cat "$OUT/report.json"
```

先贴 `LAST_COMMITTED_EPOCH=`、`SUMMARY=` 和 `REPORT=` 行。`report.json` 较长，可回传该文件。`first_frame_arrays.npz` 留在服务器，保存精确的原始数组，后续需要时再读取，不必先传大文件。

另有 `ipc_preflight.json`，只记录通信路径自检，不代表模型验证或 AP 已通过；其中的临时 socket 路径会在自检关闭时释放，临时目录在诊断进程退出后清理。

结果解释：

- `outcome=zero_union_reproduced`：在同一权重、同一卡、第一帧上捕获了原版错误及合并面积为0的框。
- `outcome=not_reproduced_first_frame`：本次第一帧没有复现；这不证明问题修复，也不证明之后的帧安全。
- `outcome=other_failure` 或进程非零：查看 `first_frame.log`、`report.json` 和 `original_traceback.txt`，保留新的实际异常。

进程返回0表示“诊断取证完成”，不表示 smoke 或科学评测完成。所有输出都标记 `eligible_for_AP_or_selection=false`；工具不会生成原实验的成功收据。

## 记录内容与不变性

保持原 `e1_runtime.install` → 冻结 `evaluation.evaluate` / `predict.generate` → 官方 `inference.main`，沿用原模型、数据加载、FOV、阈值、随机种子和后处理。Checkpoint 读取使用 `weights_only=True`，继续由冻结生成器检查 key/shape/dtype 和实际加载值。

只在独立进程中安装被动观察器，原操作照常调用、原返回对象照常返回；原 IoU 除零也原样抛出。观察器不向 IoU 分母加小数，不丢弃或替换任何框，不替换预测 None，不重写 AP。没有改动旧文件内容。

保存的证据包括：

- checkpoint 所有 tensor 的有限值和范围，BN（归一化层）保存的统计量。
- 第一个原始预处理输入、分类输出 `psm`、回归输出 `rm`。
- 原生 sigmoid 概率、解码后的中心/高宽长/角度、生成的角点、坐标变换结果。
- 原先两个过滤器的真实 mask、进入 NMS 的框和分数、原生多边形面积。
- 实际出错的 NMS 配对索引、两个多边形、合并面积及其原始解码值。

只有从原生概率选框、原有过滤器到 NMS 的数组逐项完全对应时，才将问题框映射回 anchor 和原始回归值；无法证明对应关系时明确标注 unavailable，不猜测。

`nms_numeric.py` 中的诊断面积仅用于定位数值在哪一步丢失，不用于计分。错误取证使用原 Shapely 多边形和原 IoU。被动 CPU 拷贝会影响运行时序，报告明确保留这一局限；没有承诺 GPU 逐位重现。

## 已有进度怎样保留

从真实代码看，保存顺序是：本轮训练 → 检测权重 → 验证 loss → Clean AP → scheduler → 完整 Adam/随机状态 → `complete.json`。

本次报错发生在 Clean AP，因此可以读已经存在的第2轮检测权重做诊断。完整恢复必须以实际 `epochs/NN/complete.json` 及其 checkpoint、全状态、Clean summary、采样轨迹的哈希为准。工具只读核验这些记录，并在 `request.json` 的 `resume_audit` 中报告最后完整提交轮次。

如果只有第1轮完整提交，第2轮的检测权重不能代替第2轮 Adam、scheduler 和随机状态。后续继续原训练通常要从第1轮完整状态重做未提交的第2轮；已完成 E0 和其他有效阶段可由原总入口按哈希复用。是否需要新的计算或协议修复，要先看本诊断结果；不要直接反复重启训练来碰运气。

## 本地工程检查

```sh
python -B gspr_e01_nms_diagnostic/nms_main.py --self-test
```

仅运行 NumPy/标准库合成检查和来源哈希，不导入 Torch/OpenCOOD，不读取真实权重或点云。本地覆盖原版除零重现、观察器返回值/异常不变、精确索引映射、不同数值失败位置、完整状态提交判断、staticmethod 恢复和禁止训练调用。

服务器已经回传第2轮 checkpoint 的第一帧除零和逐层被动观察证据，见上面的2026-10-23说明。新增独立 BN 对照尚未在服务器验证；模型输出异常的因果来源仍待确认，不能宣称计算修复或 smoke 通过。

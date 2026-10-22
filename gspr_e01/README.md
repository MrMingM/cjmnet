# GSPR E0 + E1 一键实验

代码交付；真实科研运行全部在Linux ROCm服务器进行。本地仅执行静态与合成工程检查。没有预写实验分数或完成状态。

## 复用范围

已检查对话 **sota模型（UECP）打法** 的最新交付与实际19个文件：`gspr_e0_diagnosis/` 已实现完整E0，其原始源码快照核验一致，本次未修改。它覆盖原C/D等价性、R0–R3、r/u/evidence分布、三天气配对点云、目标错误、原版三协议和恢复。

E0代码已完成，服务器实验是否完成必须以真实收据为准。默认总入口调用原E0总脚本。新增代码集中在E1和统一调度，不重写E0、模型、检测损失、AP或整个训练框架。

E0原文件名称映射：

| 本任务要求 | 原E0文件 |
|---|---|
| `E0_RESULTS.json` | `E0_FINAL_RESULTS.json` |
| `E0_RELIABILITY.json` | `E0_RELIABILITY_STATS.json` |
| `E0_TARGET_ANALYSIS.json` | `E0_ERROR_ANALYSIS.json` |
| `E0_AUDIT.json` | `E0_EXECUTION_AUDIT.json` |

总入口核验全部原E0阶段收据后，将JSON按新名称发布到新输出目录；Markdown和图像指向完整原报告。原实验权重、评测缓存与历史代码只读。新E0的底层证据仍在 `E0/stages/`，已有独立E0则保留在其原目录，并登记路径及SHA256。

## 上传

将整个 `gspr_e01/` 和已经交付的整个 `gspr_e0_diagnosis/` 同步到 `/home/cjm/OpenCOOD-main/cjmnet/`。还需服务器现有冻结包 `v2xr_weather_l012/`、`v2xr_train_attfuse/`、`v2xr_gspr_clean/`、`v2xr_protocol_audit/`、`v2xr_official_repro/`、`attfuse_gspr/`；不得只上传若干`.py`。`PINS.json`与`source_snapshot.json`随包上传。服务器无需Git。

默认输入：已验收训练提取 `/data/cjm/datasets/v2xr-official-train-v2/`，历史T0 `/data/cjm/datasets/logs/v2xr-weather-l012-v1/`，其余官方runtime/validation/D−1A/B/C路径继承冻结包。不会解压、下载权重或读取正式test。

T0优先读取仓库 `logs/train/FINAL_RESULTS.json`。服务器未同步该文件时，使用E0随包提供的同一文件原字节副本，校验其SHA，并要求服务器C/D完整summary与副本逐字段相同；preflight及报告明确记录来源。不会拿提示词中的舍入数字补齐结果。

## 唯一后台启动入口

从头执行，单卡串行，默认物理GPU 2。整段复制到服务器：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
ROOT=/data/cjm/datasets/logs/gspr-e01-v1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/launcher_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
RUN_MODE=serial GPU=2 RESUME=0 nohup sh gspr_e01/run_all.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
```

断点恢复：使用完全相同的卡号、路径、模式、代码、线程环境，只将 `RESUME=0` 改为 `RESUME=1`：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
ROOT=/data/cjm/datasets/logs/gspr-e01-v1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/resume_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
RUN_MODE=serial GPU=2 RESUME=1 nohup sh gspr_e01/run_all.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
```

每次启动先重新核验真实输入与资源；成功阶段通过字节哈希后复用。训练从最近完整提交的epoch恢复模型、Adam、scheduler、criterion累计量及Python/NumPy/Torch CPU/GPU RNG；未提交epoch重跑，已提交epoch保留。评测沿用冻结逐帧缓存，完整预测帧不再前向。输出目录存在不代表完成；正式完成文件为顶层 `completion.json`，同时必须具有所有阶段与报告的有效哈希。

两张独立空闲卡的可选并行入口，使用同一总脚本。首次运行也可以采用下面这套配置；恢复时保留该配置：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
ROOT=/data/cjm/datasets/logs/gspr-e01-v1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/launcher_parallel_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
RUN_MODE=parallel GPU_E0=2 GPU_E1=0 RESUME=0 nohup sh gspr_e01/run_all.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
```

E0/E1分别使用子进程、GPU、目录、日志。E1内部四个模型顺序运行。任一分支技术失败时保留另一分支，等待其正常完成；两组均完整通过才生成综合结论。串行模式也保留E0失败后独立执行E1的能力。科学上的低分不会改变T2、轮数、batch、GT或验收门槛。

只设置 `ROCR_VISIBLE_DEVICES`，脚本在各子进程分别指定卡号；清除HIP/CUDA冲突变量。默认保留原线程环境。资源门槛在 `settings.json` 预先登记：E0/E1分别至少8/24 GiB可用显存，每张卡至少95%显存空闲；串行/并行至少32/64 GiB可用CPU内存，输出盘至少200 GiB空余。真实smoke继续验证实际链路；运行时资源变化会如实报错，不缩小实验。新控制器有输出锁和物理GPU租约，旧E0仍保留自身锁。其他项目若占卡，由显存空闲检查拒绝本次启动。

## 避免重复运行已经完成的E0

如果独立E0已在服务器完整结束，首次启动总任务时，在上述启动命令前增加：

```sh
E0_EXISTING_ROOT=/data/cjm/datasets/logs/gspr-e0-diagnosis-v1 \
RUN_MODE=serial GPU=2 RESUME=0 nohup sh gspr_e01/run_all.sh > "$LOG" 2>&1 < /dev/null &
```

恢复时继续保留 `E0_EXISTING_ROOT` 并使用 `RESUME=1`。程序核验原E0源码、原C/D身份、每一阶段及产物哈希，完整通过才导入；旧E0当时使用的GPU按其原记录保留，不要求在新GPU重跑。未完成的独立E0不能冒充完成。默认未提供该参数时，在 `gspr-e01-v1/E0/` 运行原诊断链。

## 调试模式

`MODE=all` 为默认，自动包含全部self-test与smoke，无需第二条启动命令。

可以用 `MODE=self-test` 或 `MODE=smoke` 配合**独立** `OUTPUT_ROOT` 后台运行相同入口。self-test包括原E0真实等价检查和E1真实数据forward/backward；smoke还完成四组训练→保存/精确加载→Clean验证/选模→三天气推理→原版AP→报告。每个模型2轮×32帧、batch8，共8次更新；每种天气12帧。输出标记无正式结论资格，不与完整T0 AP混比。该调试目录不能改成all模式继续；正式all内部另建 `smoke/E1/`，与四组正式训练严格隔离。

## E1实现与配对

- T1为Clean/Fog/Snow=3/7、3/7、1/7；T2为3/7、2/7、2/7。保留原 `random.randint(0,6)` 每CAV一次调用，仅替换该训练dataset实例的LiDAR文件后缀。原GT/FOV/COM范围、空GT重采样、radar/camera辅助读取和validation行为不变。原生函数的AST结构必须符合已审查模式；不匹配立即停止。
- `adapters.py` 对被冻结的训练函数只做6处机械调整：三处smoke epoch上限1→2、两处smoke子集16→32、一处计数加入Snow。正式20轮、模型/损失/反向/优化/验证/scheduler及提交顺序均沿用原函数。未启用其旧BN/FiLM分支。
- 四组正式训练的batch8、Adam(.002,eps1e-10,wd1e-4)、MultiStep([10,15],gamma.1)、seed20261017与原实验一致。8083训练帧，drop_last后每轮8080帧/1010次更新；每轮完整829帧Clean O AP70选模，精确平分取最早轮。保存epoch20和全部训练状态，不训练T0。
- D创建时逐tensor检查共享初始骨干与匹配C完全相同。逐轮比较真实读路径/SHA、CAV天气轨迹、worker ID/seed及Python/NumPy/Torch随机状态、最终输入/GT、重采样与增强调用；任一差异在D该轮提交前作为技术错误停止。
- 实际原加载函数执行完整状态roundtrip并要求精确一致；smoke额外检查已有Adam动量的两轮状态。它验证状态正确恢复，不承诺训练更新逐位相同，也不调整ROCm算子或放宽历史评测容差。
- 所有最终评测继续调用 `v2xr_train_attfuse/evaluation.py:evaluate` →冻结 `predict.generate` / `evaluate.replay`。同一缓存回放O/O*/F。保留829帧、固定6224 GT、各天气真实GT身份、score .20/NMS .15等检查。新实验身份独立登记，旧快照SHA不变。
- 最终评测的被动观测器记录真实post-FOV/pre-voxel LiDAR，在原生补点前取点，按固定Clean框统计观测点数；检测输入与预测未修改。距离、目标体积、点数分组报告原匹配目标召回。FP不强制归到某个GT组，也没有重写分组AP。模型对比的新增/丢失目标从冻结匹配身份读取。

## 日志与结果

```sh
tail -f "$LOG"
cat /data/cjm/datasets/logs/gspr-e01-v1/current_stage.json
cat /data/cjm/datasets/logs/gspr-e01-v1/completion.json
cat /data/cjm/datasets/logs/gspr-e01-v1/reports/E01_FINAL_CONCLUSIONS.md
```

若重新登录后 `$LOG` 未定义，使用启动时打印的真实日志路径。各阶段详细日志在 `logs/`；每轮Clean验证日志在该模型epoch目录旁。`failure.json`保留上次失败证据，当前状态与本轮launcher日志用于判断是否正在运行。

E0报告在 `E0/`。E1四组在 `E1/T1/C,D/`、`E1/T2/C,D/`，保留config、全部weights/full states、20轮history、selected、采样/配对审计及最终缓存。E1根目录发布要求的七个JSON/Markdown报告；AP与差值保留完整精度。综合报告在 `reports/E01_FINAL_CONCLUSIONS.md`。工程筛选门槛预先固定，单种子变化不宣称统计显著或严格交互因果。

## 检查与待服务器验证项

本地 `python -B gspr_e01/check_static.py` 仅运行静态和合成测试，不导入Torch/OpenCOOD、不读取权重/点云、不运行任何科研实验。另检查Pyflakes与 `sh -n`。旧E0的独立合成测试及快照检查亦保持有效。

交付检查：新增42项合成测试、原E0的27项合成测试、Python 3.10语法、Pyflakes、POSIX `sh -n`、LF/UTF-8/JSON和源码哈希均通过。Windows沙箱的旧E0测试默认临时目录限制，通过将测试临时目录指向工作区解决；没有修改其生产代码。状态恢复测试的本地替身只使用标准库，真实Adam/RNG恢复由服务器smoke另外验证。

服务器总入口依次检查：真实输入/原模型/冻结缓存、设备和空间、原E0 self-test/smoke、E1模型/损失/梯度、完整状态恢复与四组真实smoke，随后运行完整科学实验。当前本地检查不能证明服务器训练、829帧推理、运行耗时或科研结论已经通过；这些需由服务器收据给出。技术失败保持未完成状态。

本轮不实现E2/E3。最终建议链接E0干预/可靠性/覆盖与E1配对结果；论文最终主张仍需独立评测集及多随机种子。

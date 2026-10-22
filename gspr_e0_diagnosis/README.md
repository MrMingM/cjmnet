# GSPR E0 机制诊断

状态：代码交付；本地只做静态和纯 NumPy 合成测试。真实模型 self-test、smoke 和完整 E0 尚待服务器执行。没有预写 E0 实验分数或结论。

只诊断现有 C/D，不训练、不保存修改后的权重、不读取正式 test。完成后生成研究决策报告，停止在 E0，不自动实施 E1–E4。

## 上传与唯一启动命令

把整个 `gspr_e0_diagnosis/` 目录（包括两个哈希清单和历史参考副本）放到：

`/home/cjm/OpenCOOD-main/cjmnet/gspr_e0_diagnosis/`

使用用户指定的物理 GPU **1**。在服务器执行下面整段命令，后台顺序运行全部阶段：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
ROOT=/data/cjm/datasets/logs/gspr-e0-diagnosis-v1
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/launcher_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
PHYSICAL_GPU=1 nohup sh gspr_e0_diagnosis/run_all.sh > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
```

**中断恢复：重新执行同一段命令。** 不更换代码、输出目录、卡号、Python 或线程环境。已完成阶段先核验全部登记哈希，逐帧缓存和配对分析可以续跑。上一阶段失败时，下游阶段不会执行。并发启动会被文件锁拒绝；旧工作进程存活时也不会开始第二个写入者。

查看总日志：使用启动命令打印的真实 `LOG` 路径执行 `tail -f "$LOG"`。总日志会显示当前阶段的详细日志路径，可再 `tail -f <打印的阶段日志路径>`。不要根据文件名猜测正在运行的版本或日志。

最终结果目录：

`/data/cjm/datasets/logs/gspr-e0-diagnosis-v1/`

该目录顶层发布九个要求的报告文件及 `figures/` 的链接，实际不可变报告保存在 `stages/final/`。逐帧证据、输入/输出哈希及失败历史保留在各阶段目录和 `failures/`。

## 自动执行顺序

1. **Preflight**：复核冻结源码、原实验、完整输入哈希、两模型选模契约/20轮 Clean 候选、真实 checkpoint SHA256、全部 key/shape/dtype/加载值、完整三天气预测缓存及 GT 身份。检查 ROCm、版本、GPU 和空间。
2. **Self-test**：启动时的静态/合成测试通过后，C/D 各三天气、每场景前两帧，共12个配对协作帧；比较无观测器的重复原始前向、R0 被动观测前向、分类/回归、最终框/分数/None、GT、一对一匹配和O/O*/F小样本 AP。D 额外执行 R1/R2/R3，检查真正进入 PFN 的张量。
3. **Smoke**：配对数据审计覆盖每场景两帧及推理前12帧的并集；推理对三天气前12帧顺序执行 R0–R3，完成统计、三协议评测、匹配与错误分析。这里的结果没有正式结论资格。
4. **Baseline**：只复制并重放原 C/D 完整预测缓存，AP/计数必须与历史 `FINAL_RESULTS.json` 完全一致。
5. **Paired Data Audit**：完整829帧×三天气，核对实际车辆、传感器姿态及原版变换/处理规则，保存真实点覆盖、原生体素占用、近邻代理和目标内几何支持。
6. **Reliability Audit**：D/R0 完整三天气推理并观察全部真实有效体素槽位。逐帧最终预测须复现原 D；完整 AP 须通过固定容差。
7. **Interventions**：同一 D 权重，顺序完整运行 R1/R2/R3；输入张量、车辆顺序、后处理与原版一致，逐帧比较 R0 输入哈希。
8. **Error Analysis**：复用冻结 C/D 最终缓存与官方一对一匹配，链接固定 Clean 目标、覆盖/可靠性及挽救/丢失身份，报告分数/定位的最终输出层观察。
9. **Final Report**：再次验证历史输入/模型/缓存没有变化，输出完整报告和七个研究决策问题。

一个 POSIX `sh` 总脚本完成上述流程；内部阶段单独开进程以保留官方源码隔离和推理初始化顺序。无需操作者逐个启动，也不依赖服务器存在 Git。

## 实验约定

- 实际执行的是 `v2xr_gspr_clean/integration.py` 的官方模型适配器。冻结的 `attfuse_gspr/pillar_vfe.py` 已阅读，但它不是本次 V2X-R 的执行类。
- 模型、官方数据集、16 worker、随机种子、GT、score/NMS 和 AP 都复用已冻结路径。只包装 GSPR 返回值，并通过两个 PFN 前置钩子检查加权前后张量。参数和 buffer 在前后以内容哈希核验。
- R0 原样返回；R1 有效点为1、padding为0；R2 为每辆车的有效点均值；R3 使用私有 NumPy PCG64，只置换车内有效点。种子由 `[20261021, weather, scene, timestamp, CAV ID]` 的 SHA256 前16个十六进制字符构造。保存输入顺序、种子、置换索引哈希、权重多重集合和均值审计；不推进原推理的全局 RNG。
- 预设容差写在 `protocol.json`：head `atol=1e-5, rtol=1e-4`；最终框绝对误差 `1e-4m`、分数 `1e-5`；AP `1e-6`。框数量、None、dtype、GT 身份及小样本匹配身份需要一致。超差立即终止，代码不会放宽容差。重复原前向的差异单独记录，不能作为模块效应。
- R0 为提取全量统计需要重新前向；旧 C/D 缓存用于只读基线和逐帧复核。原缓存不会被修改，新 R0 不是悄悄接受一个不同的基线。
- 可靠性统计区分 sensor/ego 距离，使用 `[0,20)、[20,40)、[40,70)、[70,+∞)`，不丢弃远处观测。分位数使用1000格直方图近似，记录误差格宽；均值/方差用全部真实槽位。密度、距离、强度与邻域柱体占用的条件相关性仅作描述。
- 点云退化同时保留原始点、实际 post-FOV/pre-voxel 点和原生体素占用。原版临时补点单独标记，排除出真实几何与可靠性统计。CPU配对分析精确沿用冻结D-1B的逐帧种子：每个天气前重置Python/NumPy/Torch为20261016+frame_index；实际推理仍执行原16-worker路径，真实推理槽位统计单独保存。跨场景self-test比较同一真实输入上的原模型与观测模型；完整R0和原缓存另做严格复核。
- 最近邻代理固定为0.05/0.1/0.2m，0.1m用于可靠性关联；不把未匹配点自动称为雪噪声。官方加载器读取绿色通道，并用 `label==0` 构造 `de_lidar_np`；当前C/D未启用这条分支，也未用于GSPR点监督。另行保存该作者掩码、以1e-4m容差关联到真实体素点，明确区分零值、非零、歧义和未映射，报告条件相关；未验证生成语义前不作为物理噪声真值。
- C/D 错误分析保留历史缓存中固定Clean GT的原字节角点。最终框跨模型对应仅是唯一双向IoU≥0.5的几何代理，歧义单列。另有保持C最终框集合、只换入对应D分数的排序敏感性回放，仍调用原AP；这是NMS之后的分析，不是模型新性能或完整内部因果分解。
- scene bootstrap 只描述挽救减丢失的目标数区间，不重定义逐帧排序 AP。单训练种子、六场景、一次 R3 置换及重复使用 validation 的限制始终保留。

## 输出索引

- `E0_FINAL_RESULTS.json`：真实完整四组×三天气×O/O*/F 的全部 AP、差值和 TP/FP/FN。
- `E0_RESULTS.md`、`E0_CONCLUSIONS.md`：结果表、限制、七个问题、证据与研究决策表。
- `E0_PROTOCOL.json`：预先固定设置、实际输入路径和原模型选择身份。
- `E0_RELIABILITY_STATS.json`：r/u/evidence、pillar、距离/密度/CAV/框内外/近邻代理与条件相关。
- `E0_POINT_DEGRADATION.json`：完整配对覆盖、点数及逐目标证据索引；原生体素统计在逐帧记录。
- `E0_INTERVENTIONS.json`：各 arm 三协议 AP/差值/计数、相对R0挽救与丢失。
- `E0_ERROR_ANALYSIS.json`、`stages/error_analysis/objects.jsonl.gz`：C/D目标级错误、分组样本量、分数/定位和可观测性关联。
- `E0_EXECUTION_AUDIT.json`：来源、加载、阶段收据、版本、GPU、失败历史、全部图像哈希。
- 各 arm 的 `all_protocol_matches.jsonl.gz`：O/O*/F逐帧三阈值匹配；`per_frame_metrics.jsonl`、`predictions/` 和 `observations/` 保存原始预测与紧凑统计。
- `stages/paired/clean_clouds/*.npz`：复核近邻代理所需的真实 Clean 点；空间热图只保存预设样本的压缩子采样，不生成逐点 JSON。

## 新增文件与测试

| 文件 | 用途 |
|---|---|
| `run_all.sh` | 唯一服务器总入口 |
| `e0_main.py`、`e0_common.py` | 阶段依赖、日志、锁、身份和恢复 |
| `e0_runtime.py`、`e0_checks.py` | 原包隔离、选模/缓存核验、真实模型等价性 |
| `e0_numeric.py`、`e0_observe.py` | 权重干预、PFN验证及全量统计 |
| `e0_data.py` | 配对点云、原生体素与目标几何分析 |
| `e0_inference.py` | 原版推理包装、观察证据与缓存事务恢复 |
| `e0_analysis.py`、`e0_report.py` | 冻结匹配、错误分析、报告和图像 |
| `protocol.json` | 固定协议和容差 |
| `original_FINAL_RESULTS.json` | 仓库已有历史结果原字节副本，作为只读参照；不是本轮 E0 结果 |
| `PINS.json`、`source_snapshot.json` | 冻结依赖及交付代码哈希 |
| `tests_local.py`、`check_static.py` | 无 Torch/OpenCOOD 的合成与静态检查 |
| `SOURCE_REVIEW.md`、`README.md` | 真实实现核对和操作说明 |

本地已运行27项纯NumPy合成测试，全部通过；Python 3.10 AST、Pyflakes、POSIX `sh -n`、LF/UTF-8/JSON和冻结哈希检查通过。检查命令为 `python -B gspr_e0_diagnosis/check_static.py`，不会运行实际科研实验。Windows沙箱对临时文件原子重命名的限制在测试时通过受审批的隔离外执行解决，没有改变生产原子写入逻辑。

尚待真实服务器验证：当前GPU 1可用性、依赖/原始产物是否完整、实际 `.pth` 加载、829帧全量配对/GT、数值容差、运行时间和空间、真实 self-test/smoke 与所有 E0 结论。代码中的自动检查会逐项核验，失败会保留错误并停止。

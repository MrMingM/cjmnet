# V2X-R 天气混合训练 L0 → L1 → L2

状态：实现及本地静态/纯 Python 合成检查已完成；真实模型自检、smoke、正式训练与三天气结果待服务器运行。没有预填性能结果。

所有新增实现都在本目录；未修改旧训练包、官方 runtime、既有权重/结果或 AI_CONTEXT.md。无需分支、worktree、Git 环境，也不会下载、解压或重提取数据。请同步**整个目录（包括 source_snapshot.json 与 PINS.json）**，并保留仓库中原有依赖目录。

新增文件明细：

| 文件 | 作用 |
|---|---|
| run_all.sh、orchestrator.py、main.py | 唯一总入口、串行阶段控制、内部子进程入口 |
| preflight.py、auditlib.py、train_acceptance.py | 数据/环境/冻结身份、旧8083帧接受契约桥接 |
| data.py、models.py、training.py、checkpoint.py | 实际读取审计、C/D及FiLM工厂、训练、完整状态恢复 |
| calibration.py、evaluation.py、gates.py、results.py | BN校准、冻结官方评测桥接、固定门槛和全部报告 |
| tests_local.py、server_tests.py、check_static.py、test_shell.sh | 本地合成、服务器模型/接口、静态及shell测试 |
| settings.json、PINS.json、source_snapshot.json | 预注册配置、冻结依赖、独立源码身份 |
| README.md、PROTOCOL.md、LOCAL_VALIDATION.md | 操作说明、源码协议核对、本地测试边界 |

重要协议：作者代码按每辆 CAV 随机选择，Fog 概率 **3/7**，Clean **4/7**，同一协作帧允许混合天气。已严格保留，不改成 50/50。完整设计和源码证据见 [PROTOCOL.md](PROTOCOL.md)。

此外，冻结的 IntermediateFusionDataset 实际没有调用 YAML 中的 world flip/rotation/scaling。实现保留原路径，并记录真实调用次数；仍有点顺序 shuffle、天气抽样等随机操作。

## 一次启动

在服务器项目目录 `/home/cjm/OpenCOOD-main/cjmnet`、已配置好的 opencood 环境中执行。GPU=0 需换成可用的物理卡号；首次输出目录必须为空或不存在。

```sh
nohup env MODE=all GPU=0 RESUME=0 sh v2xr_weather_l012/run_all.sh > /data/cjm/datasets/logs/v2xr_weather_l012_all.log 2>&1 &
```

执行顺序：Preflight → Self-test → C/D 训练 smoke → C/D BN smoke → C/D FiLM smoke → L0 20轮/Clean O选模/三天气评测 → L0 gate → L1 20轮/Clean O选模/三天气评测 → L1 gate → C_BN/D_BN全训练Fog校准及评测 → C_FiLM/D_FiLM各3轮适配及评测 → 六组总报告。

所有阶段严格串行。任何 smoke 或技术检查失败都阻止正式训练。L0 预设 +5 pp Fog 门槛失败停止；L1 科研门槛失败仍继续 L2，不需要手动启动后续阶段。

任意安全提交点中断后，使用同一物理 GPU、输入、输出、源码和环境恢复：

```sh
nohup env MODE=all GPU=0 RESUME=1 sh v2xr_weather_l012/run_all.sh >> /data/cjm/datasets/logs/v2xr_weather_l012_all.log 2>&1 &
```

模型/优化器/调度器/损失累积状态及所有 RNG 均恢复到最后完整 epoch；未完整提交的一轮重跑。已完成阶段及评测帧按哈希复用。模型代码或数据变动会失败，不能静默恢复。`kill -TERM` 会结束正在运行的阶段进程组；若用了 `kill -9`，先确认旧工作进程已退出。新流程中的 worker 锁也会阻止同一输出上的并发写入。

查看当前阶段和统一日志：

```sh
cat /data/cjm/datasets/logs/v2xr-weather-l012-v1/current_stage.json; tail -n 80 -f /data/cjm/datasets/logs/v2xr_weather_l012_all.log
```

每阶段另有 `logs/<stage>.log`；每轮 Clean AP 日志在模型目录 `epochs/NN/clean_ap_<sha>.log`。数据 SHA 检查及全训练集 Fog 校准可能较久，均已包含在上述后台任务内。

## 配置与前提

环境变量：

| 变量 | 默认值 / 用途 |
|---|---|
| MODE | all；另支持 preflight、self-test、smoke，均先执行前置检查 |
| RESUME | 0；恢复同一根目录用1 |
| GPU | 0；只能一个物理卡号，仅导出 ROCR_VISIBLE_DEVICES |
| PYTHON | /home/cjm/miniconda3/envs/opencood/bin/python |
| OUTPUT_ROOT | /data/cjm/datasets/logs/v2xr-weather-l012-v1 |
| TRAIN_ROOT | /data/cjm/datasets/v2xr-official-train-v2/train；也接受其父提取根 |
| TMPDIR | /data/cjm/datasets/tmp/v2xr_weather_l012 |
| RUNTIME_ROOT | /data/cjm/datasets/v2xr-official-runtime-v1 |
| BASELINE_ROOT | /data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1 |
| MODEL_DIR | /data/cjm/datasets/logs/v2xr-official-attfuse-model-v1 |
| D1A_ROOT | /data/cjm/datasets/logs/v2xr-official-attfuse-clean-repro-v1 |
| GT_AUDIT_ROOT | /data/cjm/datasets/logs/v2xr-official-gt-audit-v1 |
| D1C_ROOT | /data/cjm/datasets/logs/v2xr-dminus1c-protocol-v1 |
| VIEW_ROOT | /data/cjm/datasets/v2xr-official-validate-v1 |

保留旧验收 manifest、plan、data_acceptance、records、原始 train 归档及接受证据；沿用旧完整性检查，不重提取。历史 D−1A/B/C 的固定产物、官方模型别名/配置和验证天气 overlay 必须还在。原始归档只作哈希核对，不读取或扫描正式 test。所有 train/validation 路径及软链接目标继续使用旧 no_test 检查。

固定配方在 settings.json；不得根据 Fog/Snow 分数临时改 lr、epoch、batch、门槛或选模规则。BN 与 FiLM 各自从 C/D 的原选中 checkpoint 开始，四个 L2 变体独立保存。

`MODE=preflight` 等模式会在相同 OUTPUT_ROOT 保存前置阶段。之后继续 `MODE=all` 时加 `RESUME=1`。正式/smoke 输出永远相隔；本脚本没有从 smoke 恢复正式训练的入口。

## 输出和状态

根目录包含 `preflight/`、`self_test/`、`smoke/`、`L0_C/`、`L1_D/`、`L2_C_BN/`、`L2_D_BN/`、`L2_C_FiLM/`、`L2_D_FiLM/`、`stages/`、`reports/`、`logs/`。

`stages/<stage>/` 包含 `request.json`、`stage_state.json`、`complete.json`、失败时的 `failure.json`。完成文件记录输入/代码/配置依赖、产物哈希、时间、PID及执行身份。目录存在不代表完成。旧失败 traceback 保留供审计，当前状态看 `current_stage.json`；成功恢复后旧 STOP 归档到 stop_history。

模型目录：`contract.json`、`config.yaml`、`selection_policy.json`、`resume_states/`、`epochs/NN/{complete.json,sampling.jsonl,checkpoint_<sha>.pth,clean_ap_<sha>/}`、`history.json`、`selected.json`、`training_summary.json`。BN 有 `artifact.json`、`adaptation.json`、`calibrated.pth`、完整校准读取日志。FiLM 有冻结主干审计、参数量和逐 batch gamma/beta 分布。

各模型 `final_evaluation/` 保留原官方输出、逐帧预测、GT及 O/O*/F AP/分母/None 审计。最终查看：

```text
/data/cjm/datasets/logs/v2xr-weather-l012-v1/reports/FINAL_RESULTS.md
/data/cjm/datasets/logs/v2xr-weather-l012-v1/reports/FINAL_RESULTS.json
/data/cjm/datasets/logs/v2xr-weather-l012-v1/reports/EXPERIMENT_AUDIT.json
/data/cjm/datasets/logs/v2xr-weather-l012-v1/reports/CONCLUSIONS.md
```

L0 未过门槛时写 `STOP.json`（L0_GATE_FAILED）、`reports/L0_DIAGNOSIS.md/.json` 及实际已完成的部分报告，进程退出20。技术失败非零退出并保存 traceback，不生成伪造的科研负结果。L1 gate 为 L1_GATE_FAILED 时，只要技术有效仍正常进入 L2。

## 测试边界

本地只执行不依赖 Torch/OpenCOOD 的静态和合成测试，不做任何科研实验。记录见 [LOCAL_VALIDATION.md](LOCAL_VALIDATION.md)。

总脚本会在服务器自动运行：官方 C 等价、GSPR 共享初始化/RNG/单位权重等价/掩码/检测梯度/正则、C/D BN 仅 buffer 修改及输出变化、C/D FiLM 恒等/梯度/冻结/严格重载、Adam/scheduler/criterion/Python/NumPy/Torch CPU+CUDA 恢复后的下一步一致性、真实多 worker 数据观察器与原数据集逐项等价、冻结官方 AP 的 None/GT/一对一匹配测试。随后才跑真实数据短 smoke。没有服务器执行记录前，不宣称真实模型测试通过。

单种子 validation 结果不能声称统计显著或正式 test 泛化。原版训练的 GT 依赖天气、空 GT 重抽样、空体素临时点等行为保留；BN 采用按 batch 累计统计，FiLM 仅训练新增模块，这些边界在正式报告中继续保留。

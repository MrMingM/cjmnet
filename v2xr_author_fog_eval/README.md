# 作者 Fog AttFuse 固定权重同集评测

本目录只做推理和计分。作者权重固定为 `AttFuse_LiDAR_train_w_fog.pth`，不训练、不微调，不创建优化器，不读取 test，不下载权重。所有真实模型和数据操作只在服务器执行。

## 上传内容

把整个 `v2xr_author_fog_eval/` 同步到服务器的 `/home/cjm/OpenCOOD-main/cjmnet/`。包括 `reference/`、`DEPENDENCIES.json`、`source_snapshot.json`，不要只上传 `.py`。

本目录依赖服务器原有的冻结 `v2xr_train_attfuse/`、`v2xr_protocol_audit/`、`v2xr_official_repro/` 及其哈希所引用的文件。`DEPENDENCIES.json` 固定所读历史源文件；历史目录已经匹配时不需要重复上传或改动。完全不依赖 `.git`、分支或 commit 命令。

`reference/FINAL_RESULTS.json` 是本地 `logs/train/FINAL_RESULTS.json` 的逐字节副本，保留完整精度，来源与 SHA256 在 `reference/provenance.json`。优先读取服务器 `logs/train/FINAL_RESULTS.json`；该文件缺失时读取随本目录同步的真实归档副本，报告会注明实际来源。已有文件的哈希不一致会停止，不能悄悄改用预置数字。

当前本地 `AI_CONTEXT.md` 结束于 §49.6，没有 §50。本次未修改它，也没有提前写入作者评测成绩。

## 一次启动全部评测

沿用之前使用的物理 2 号卡；需要时只改 `GPU=2`。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
OUT=/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1
mkdir -p "$OUT"
nohup env MODE=all RESUME=0 GPU=2 OUTPUT_ROOT="$OUT" \
  CHECKPOINT=/home/cjm/OpenCOOD-main/cjmnet/logs/AttFuse_LiDAR_train_w_fog.pth \
  sh v2xr_author_fog_eval/run_server.sh > "$OUT/launcher.log" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$OUT/launcher.log"
```

运行顺序：输入及冻结来源检查 → 作者权重严格加载自检和原版七项评分合成检查 → 独立三天气各12帧 smoke → 三天气各829帧推理 → 同预测 O/O*/F 回放 → C/D 对比与报告。

总脚本会按阶段每20秒输出心跳。每个 GPU 阶段使用独立 Python 进程，避免官方 runtime 重复初始化或混入本地 OpenCOOD。首次数据 SHA256 核验可能较久，此时没有训练或模型前向。

默认 Python 为 `/home/cjm/miniconda3/envs/opencood/bin/python`，不要求先激活环境。脚本只设置 `ROCR_VISIBLE_DEVICES`，清除冲突的 CUDA/HIP 可见设备和分布式变量；Python hash seed 固定20261016。它不会强制新的确定性内核设置。

## 断点恢复

确认先前控制器已退出后运行；不删除锁文件，不另开同目录并行运行。

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
OUT=/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1
nohup env MODE=all RESUME=1 GPU=2 OUTPUT_ROOT="$OUT" \
  CHECKPOINT=/home/cjm/OpenCOOD-main/cjmnet/logs/AttFuse_LiDAR_train_w_fog.pth \
  sh v2xr_author_fog_eval/run_server.sh >> "$OUT/launcher.log" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$OUT/launcher.log"
```

恢复会重新核验来源、输入、checkpoint、加载自检以及完成记录的哈希。同一 checkpoint SHA、代码、配置、数据、物理 GPU、Python 路径和运行身份必须一致；变化会停止。

冻结生成器逐帧校验缓存 SHA256 和记录身份。已完成条件不前向；部分条件继续按原顺序读取数据，以维持 DataLoader/随机顺序，缓存帧只重用预测，缺少的帧才前向。中断发生在原子保存预测后、登记缓存前时，冻结生成器会校验后接纳该帧。损坏的已登记缓存会报错，不能当作缺帧静默重算。

`smoke/` 和 `author_evaluation/` 拥有不同的完整运行合同和帧集合，缓存不能相互替代。目录存在、旧日志中的 COMPLETE 或单个 JSON 存在，都不构成完整评测证明。

## 其他模式与可选路径

`MODE=self-test` 只运行可脱离 GPU/真实数据/真实权重的合成检查。`MODE=smoke` 运行 preflight、加载审计与三天气各12帧，结束时明确标记非完整评测。单独 smoke 后可用相同目录 `MODE=all RESUME=1` 继续。

服务器的独立自检也可后台运行：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
OUT=/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1
mkdir -p "$OUT/logs"
nohup env MODE=self-test GPU=2 OUTPUT_ROOT="$OUT" sh v2xr_author_fog_eval/run_server.sh \
  > "$OUT/logs/self-test.log" 2>&1 < /dev/null &
```

可选环境变量：`PYTHON`、`RUNTIME_ROOT`、`MODEL_DIR`、`D1A_ROOT`、`GT_AUDIT_ROOT`、`D1C_ROOT`、`VIEW_ROOT`、`CD_RESULTS`、`CD_ROOT`、`TMPDIR`。`TMPDIR` 必须在 `OUTPUT_ROOT` 内；默认 `$OUTPUT_ROOT/tmp`。

默认 `CD_ROOT=/data/cjm/datasets/logs/v2xr-weather-l012-v1`，从其 `L0_C/final_evaluation/` 与 `L1_D/final_evaluation/` 读取历史逐帧审计。若实际目录不同，首次运行前设置 `CD_ROOT`，恢复保持一致。

## 查看进度及结果

```sh
OUT=/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1
tail -f "$OUT/launcher.log"
```

详细推理日志：

```sh
tail -f /data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1/logs/author-evaluation.log
```

完成后：

```sh
OUT=/data/cjm/datasets/logs/v2xr-author-fog-attfuse-val-v1
cat "$OUT/completion.json"
cat "$OUT/author_evaluation/RESULTS.md"
cat "$OUT/reports/MATCHED_COMPARISON.md"
cat "$OUT/reports/CONCLUSIONS.md"
```

根 `completion.json` 的 `mode=all`、`full_evaluation=true` 及其哈希链通过后，才是本次总流程完整完成。阶段失败看 `failure.json`、`failure_<stage>.json` 和对应 `logs/<stage>.log`。先前失败历史在成功恢复后保留并注明已解决。

## 权重加载如何保证正确

原文件必须非空，实际 SHA256 必须严格等于：

`195a873a814cfeb9e392ea7e3b6ae12a142b009719194d3c0a5980cbb4d089f3`

来源：[作者 Hugging Face 原权重](https://huggingface.co/hx24/V2X-R_Benchmark/blob/main/benchmark/l%2Br_train_with_fog_sim/AttFuse_LiDAR_train_w_fog.pth)。不移动、覆盖或重新保存原文件。

只以 `torch.load(..., map_location='cpu', weights_only=True)` 读取；不安全回退和任意 pickle 全局注册均禁用。允许普通参数字典，或唯一明确的 `state_dict` / `model_state_dict` 包装。只允许全部 key 一致带有 `module.` 时做一对一前缀去除，记录全部映射和未使用的包装字段。混合前缀、其他映射、冲突或未知格式立即停止。

从已核验的官方纯 LiDAR YAML 创建 `PointPillarIntermediate`。逐 key 比较 shape/dtype，拒绝任何缺失、多余或不匹配；严格加载 `strict=True`，逐 tensor `torch.equal`。失败详情保存到 `checkpoint_audit.json`，不会改网络或进入完整推理。

旧 loader 原本允许跳过层，本入口在调用冻结 `evaluate` 的动态范围内替换为上述严格加载函数。冻结生成器原来的宽松 `torch.load` 请求只获得已安全解析和审计的相同 tensor 字典，不执行宽松反序列化。模型类、原始路径和两个权重符号链接仍会逐次检查。完整模型值检查仍由冻结生成器再做一遍。

新输出内复制原配置，`0.pth` / `net_epoch0.pth` 都是指向上传原权重的符号链接。配置仅沿用冻结代码已有的根路径及天气选择适配，保留模型、后处理、FOV、max_cav=7、COM_RANGE=70 等设置。模型为 eval，参数不累积梯度，前向观察器还会核验 eval/no_grad 及检测输出有限值。

## 与旧训练入口的关系

直接调用实际接口：

```python
v2xr_train_attfuse.evaluation.evaluate(
    args, own, checkpoint, destination,
    conditions=('clean', 'fog', 'snow'), limit=0)
```

此函数内部不调用 `selection` / `weather_checkpoint_gate`，可用于外部固定 checkpoint。本入口不会调用训练选模函数。根目录不建立旧训练用的 `contract.json`；使用 `run_identity.json` 记录外部模型身份，避免被解释成本地训练模型。

预测与评分继续调用未修改的 `v2xr_protocol_audit.predict.generate` 与 `v2xr_protocol_audit.evaluate.replay`。原版 `inference.main`、官方 dataset/GT/FOV/参与车辆/时间戳/辅助文件读取、天气路径、后处理和 AP 不变。seed=20261016，batch=1、workers=16、shuffle=false、pin_memory=false、drop_last=false。保持 frozen global_sort=False。

冻结 D−1A/B/C 检查仍要求原先的官方模型目录及旧 checkpoint 存在，以验证历史证据链；这些旧权重只作文件哈希检查，不加载为本次模型。缺失则停止，不下载其他权重。

## 报告能说明什么

完整三天气各829帧，检查固定 GT 各6224，官方 GT 为6224/3673/5854；逐帧身份、顺序、几何均与冻结 D−1C 对照。这些值只做检查，绝不修改 GT 凑数。

绝对 AP 和作者−C、D−作者、D−C 的全部81项差值保存完整浮点精度到 `MATCHED_COMPARISON.json`，Markdown 仅为阅读显示而舍入。JSON 还包含 Fog 预测框、原版 F 一对一 TP/FP/FN、保留/移除目标计数和协议影响分解。

历史逐帧记录必须通过 C/D 已归档 summary 中的哈希、原版协议定义及冻结 GT 检查，才生成不同帧受益/退化、具体目标新增/不再检出的分布及 `frame_gains_*.jsonl`。缺失会逐项说明；不从 AP 反推目标数量。保留/移除 TP 是同一轮 F 匹配的分组，不能混作 O 的独立匹配结果。

论文公开数字单列。[论文 v5 Table 5](https://arxiv.org/html/2411.08402v5#S5.T5) 与[作者仓库](https://github.com/ylwhxht/V2X-R)的纯 LiDAR Normal/Fog/Snow AP70 为 .7609/.6152/.4554；Table 5 本身没有明示 split，相关 Table 7/8 标明 testing。本次只有829帧 validation，表格明确不把两者当作同集复现。论文表称3D mAP，本次实现是冻结 BEV IoU/AP；不为对齐公开数值调整协议。LiDAR+4D Radar+MDD 不纳入本次同模态对照。

同集成绩可以比较固定模型性能；C/D 经 validation 选模，作者训练与选模细节没有在本次独立验证。单次差值不能证明统计显著性或训练机制因果关系。

## 检查与服务器待验证事项

本地仅执行 Python 静态解析、POSIX `sh -n` 及 stdlib/NumPy/PyYAML 合成与配置检查；测试不会导入 Torch/OpenCOOD、读取真实数据或 checkpoint。合成评分测试执行冻结原版 AP/一对一累计函数的代码，IoU 使用已知的玩具对应关系；恢复测试执行原生成器的缓存分支，停止在 GPU 推理部分之前。真实多边形/torch tensor 路径由服务器加载自检中的原版七项检查覆盖。

本地通过不代表服务器推理通过。仍需服务器确认：上传文件真实 SHA 和格式、官方结构兼容性、HIP/native 环境、冻结视图与辅助文件完整性、12帧 smoke、829帧输出和历史逐帧审计可用性。尚无作者同集实测 AP；正式结果只由服务器成功运行后生成。

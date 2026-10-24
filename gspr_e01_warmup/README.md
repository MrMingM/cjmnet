# E1：正式前2轮作为预运行

本入口按用户确认的方案执行：T1-C、T1-D、T2-C、T2-D 先各完成正式第1—2轮，四组检查全部通过后，从各自第2轮完整状态继续到第20轮。前2轮计入20轮，总轮数不增加。

代码放在新的 `gspr_e01_warmup/`，输出放在新的 `/data/cjm/datasets/logs/gspr-e01-formal-warmup-v1/`。原 `gspr_e01/`、旧短 smoke、诊断、历史权重和结果保持原样。同步时上传整个新目录，与原目录并列；同时保留服务器上已有的原代码和冻结依赖。

## 运行内容

1. 校验源码哈希、原 E01 请求、数据和已完成 E0 的所有阶段凭据。复用该 E0，缺少完整 E0 时停止。
2. 检查 GPU、内存、磁盘和完整训练/验证输入；执行原有真实数据自检。
3. 四组顺序执行正式第1—2轮。每轮使用8083帧官方训练集；原 `drop_last` 设置实际消耗8080帧、1010次更新。每轮保留原验证损失流程和完整829帧 Clean O/O*/F 评测。
4. 检查第0、1、2轮实际完整状态加载：模型、Adam、调度器、损失记录和随机状态必须精确一致。核对 C/D 初始公共权重和逐轮数据轨迹；校验每帧预测缓存、GT 身份与评测文件哈希。
5. 四组全部通过后，各自接续第3—20轮。沿用20轮完整 Clean O AP70 选模和最早并列规则，随后执行完整三天气评测、目标分析及原 E0/E1 总报告。

保持原模型、BN、学习率、优化器、采样、GT、NMS 和评分规则。没有最低 AP 通过线。当前批次 BN 对照不进入正式计算。此前8次更新的短 smoke 权重不会作为新正式训练的起点；四组按原种子重新初始化并检查公共初始权重。已有完整 E0 经核验复用。

前2轮的完整加载检查使用原 `install_restore_check`，仅扩大其审计范围；所有训练与评测参数中的 `smoke` 均为 `False`。两个新增提前返回点分别位于“已提交轮次读取后”和“完整轮次、历史、进度全部提交后”。去掉这两个返回点，训练函数语法树与原 E01 适配后的函数完全相同。

## 服务器后台启动

首次启动，默认物理 GPU 为2。以下是一整条命令，在服务器执行：

```sh
GPU=2 RESUME=0 nohup sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e01_warmup/run_server.sh >> /data/cjm/datasets/logs/gspr-e01-formal-warmup-launcher.log 2>&1 < /dev/null &
```

可覆盖 `PYTHON`、`OUTPUT_ROOT`、`LEGACY_RUN_ROOT`、`ORIGINAL_ROOT`、`TRAIN_ROOT`、`E0_EXISTING_ROOT`。默认 Python 为 `/home/cjm/miniconda3/envs/opencood/bin/python`；默认旧 E01 为 `/data/cjm/datasets/logs/gspr-e01-v1`；原 C/D 为 `/data/cjm/datasets/logs/v2xr-weather-l012-v1`；训练数据为 `/data/cjm/datasets/v2xr-official-train-v2`。

默认从旧 E01 的 `request.json` 找到外部 E0 路径，或使用其自身 `E0/`。显式指定 `E0_EXISTING_ROOT` 仍须通过同一完整性检查。运行不要求 Git 仓库、分支或 commit 命令；上游原评测记录中的身份字段保留。

默认资源门槛沿用原串行 E1：空闲 GPU 显存至少24 GiB且空闲比例至少95%，可用 RAM 至少32 GiB、磁盘至少200 GiB。不自动缩减 batch、worker 或评测帧数。

## 中断恢复

先确认原进程已退出，再执行这一整条后台命令：

```sh
GPU=2 RESUME=1 nohup sh /home/cjm/OpenCOOD-main/cjmnet/gspr_e01_warmup/run_server.sh >> /data/cjm/datasets/logs/gspr-e01-formal-warmup-launcher.log 2>&1 < /dev/null &
```

恢复需保持同一新输出目录、源码、数据身份、配置、物理 GPU、Python 和线程环境。每次重新校验输入，完整阶段凭据和文件哈希通过后复用。模型从最后提交的完整轮次状态恢复；未提交轮次重新执行。损坏凭据会停止，不会跳过或覆盖为“通过”。

预运行凭据只绑定前2轮的不可变文件。后续历史和进度更新不会使它失效，也不会重做已经通过的预运行。短临时目录解决多进程 socket 路径过长问题；后台启动保留 `nohup` 继承的忽略挂断信号设置。模型评测在独立子进程运行，避免训练锁与评测锁互相等待。

## 查看进度与结果

查看总日志：

```sh
tail -f /data/cjm/datasets/logs/gspr-e01-formal-warmup-launcher.log
```

查看当前阶段：

```sh
cat /data/cjm/datasets/logs/gspr-e01-formal-warmup-v1/current_stage.json
```

四组前2轮全部通过后，出现 `formal_warmup_gate.json`；它仅表示预运行完成。每组凭据在 `E1/T1/C/formal_warmup/receipt.json` 等对应目录。单组训练详细日志如 `logs/gate_T1_C.log`、`logs/T1_C_train.log`；每轮验证日志保存在对应 `epochs/NN/clean_ap_*.log`。

全部20轮及报告完成后，出现 `completion.json`，总报告为 `reports/E01_FINAL_CONCLUSIONS.md`，完整机器可读证据为 `reports/E01_EVIDENCE.json`。每组仍保留完整轮次提交、恢复状态、Clean 评测、选模和三天气预测缓存。失败信息保存在 `failure.json` 和相应阶段日志。

## 检查范围与仍待验证的事项

本地只做 Python 3.10 语法、静态、POSIX shell 语法、源码身份及标准库合成测试，详见 `LOCAL_CHECKS.md`。可在服务器用 `SELF_TEST=1 sh .../run_server.sh` 单独运行这些工程检查，不加载模型。

本次流程调整尚未在服务器真实运行。较长预运行让原 BN 统计量经历完整训练，但仍不能预先保证模型数值、NMS 或 AP 正常。若原验证在任何轮次报错，按原错误停止并保留证据。已有第一帧 BN 对照只支持解释该帧的数值异常贡献，不构成正式修复或 AP 改善证据。

`PINS.json` 固定原 E01 的17个文件，原 E01 再验证445份冻结依赖；`source_snapshot.json` 固定本新目录。科研执行只允许 Linux 服务器。本地未读取真实权重或点云，未启动训练、推理、BN 校准或 AP 实验。

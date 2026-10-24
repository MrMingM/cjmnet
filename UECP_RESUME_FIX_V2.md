# UECP 恢复检查修复包 v2

2026-10-20。用户已选择“保留原网络，修复恢复检查”。此包直接继续 `/data/cjm/datasets/logs/uecp_gspr_v2xr_v1`，仅适用于该次已完成预检和九组smoke、停在旧E0恢复检查的实验。

## 改了什么

- 每次训练真正加载检查点后，立即将模型、Adam状态、学习率、随机数、计数和采样进度与保存值作零容差核对。跨epoch时另外检查新epoch的计数清零。
- E0的中间恢复门禁使用已经执行的同一公共检查点对照；独立从头重训的旧失败记录保留。
- E0/E2/E3_joint的epoch2重放统一复制原分支的同一epoch1检查点，写入新的 `smoke_replay_v2/`，保留旧 `smoke_replay/`。
- 模型张量继续采用原逐元素容差。只对Adam动量增加逐张量整体误差检查，沿用 `atol=1e-5, rtol=1e-4`。optimizer step、参数组、LR、RNG、输入哈希、计数精确相等。梯度差异保留为诊断信息。
- 训练网络、GPU算子、损失、优化器超参数、数据划分、20轮预算、选模和原生AP不变。原生dev/final保留原 `PYTHONHASHSEED=20261016`；训练使用原配置种子。

这是独立ZIP运行器，不覆盖项目源码、PINS、request、contract或源码快照。运行时只调整重放目录、worker入口、报告采用的恢复门禁，并增加加载后的检查。原封存训练函数的计算语句保持一致。

已知旧依赖training.py的 b6577e... → cc5f52... 版本变化，沿用此前已核实的四个函数和模块级语句一致性检查；其余依赖继续严格校验。修复包自身按ZIP SHA256、成员清单和成员SHA256固定，不依赖Git。

## 上传并后台继续

上传 `uecp_gspr_resume_fix_v2.zip` 到 `/home/cjm/`。无需解压。服务器执行：

```sh
ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1
ZIP=/home/cjm/uecp_gspr_resume_fix_v2.zip
EVIDENCE="$ROOT/diagnostics/controlled_resume_20261020T083508_121163Z_3625228/RESULTS.json"
if [ -f "$ZIP" ] && [ -f "$EVIDENCE" ]; then
    LOG=$(mktemp "$ROOT/diagnostics/resume_v2_launcher.XXXXXX")
    nohup /home/cjm/miniconda3/envs/opencood/bin/python -B "$ZIP" run \
      --output "$ROOT" --evidence "$EVIDENCE" --mode all \
      > "$LOG" 2>&1 < /dev/null &
    echo "PID=$! LOG=$LOG"
    tail -f "$LOG"
else
    echo "请检查ZIP已上传，且原始RESULTS.json仍在指定路径"
fi
```

`--mode all` 会先复用已完成的九组smoke，完成新的恢复检查和smoke报告，再按原顺序执行内容校验、正式E0、E1、其余检测器和结果报告。新检查失败时停止，不会修改阈值继续运行。仅检查smoke可把 `--mode all` 改成 `--mode smoke`。

后续断点继续也使用这个ZIP和同一个OUTPUT_ROOT。已有有效修复记录时无需重复 `--evidence`；程序校验记录后复用已提交阶段。原启动脚本仍按旧检查规则运行。

## 保留与新增记录

- 原request、contract、源码快照和项目源码保持原字节。
- 最初failure.json、controller_failure.json和pipeline_status.json另存于 `source_repair/uecp_resume_v2_original/`。原失败阶段和failure_history保留；当前进度文件按后续执行正常更新。
- `source_repair/uecp_resume_v2.json` 记录用户选择、检查规则、ZIP身份、原始契约身份、已核实的源码差异、原诊断证据与受保护文件的哈希/时间戳。
- `smoke/resume_v2/` 保存新的中间恢复和epoch2重放门禁，旧失败门禁文件不伪装为通过。
- 每个新训练epoch的 `restore_checks_v2/` 保存实际加载检查，作为train_complete.json的受哈希保护输出；新commit标记执行修复身份。
- 新的FINAL/PARTIAL报告明确列出恢复检查修订。已完成的最终报告保持不可覆盖；失败报告使用PARTIAL前缀。

首次运行会在服务器CPU上重新核验已提供的受控对照检查点与新比较规则，然后登记修复。此操作读取已有检查点，不重跑原受控对照。原实验的进程锁被占用时停止，避免同时运行两个控制器。

## 验证与限制

本地只执行纯Python/NumPy合成检查、AST/静态检查和ZIP完整性检查。没有导入Torch/OpenCOOD，没有读取真实点云，没有训练或推理。实际E2/E3重放及完整科研流水线仍需服务器运行确认。

本修复验证保存、加载和有限步重放，不承诺训练逐位确定、长期轨迹等价或检测AP提升。原模型容差超出、NaN、输入/RNG/计数不一致或优化器整体误差超出时继续停止。

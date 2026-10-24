# E0 恢复与重复计算对照

服务器已验证：原 E0 和独立重跑 E0 的 epoch1 检查点，分别加载后全部560个浮点数组及检查的进度、随机数状态完全一致。原来的跨运行比较仍有61个模型张量和84个优化器张量超出注册容差，不能据此放行正式训练。

本工具进一步回答：从同一个中间检查点出发，多算一遍本身会产生多大差异，退出进程后恢复是否带来额外差异。

## 固定对照

- continuous：调用原封不动的 `uecp_gspr_v2xr.train.run`，完成原smoke的两次更新。第一次更新后的状态另存为公共检查点，保存后与当前内存逐项零容差比较。
- resume1、resume2、resume3：分别使用独立进程，读取同一个公共检查点，只执行剩余的一次更新。通过包外观察函数，在原训练函数真正完成加载后、下一帧计算前，与公共检查点逐项零容差比较。
- strict：同样加载公共检查点，加载检查通过后启用 `torch.use_deterministic_algorithms(True)`，记录不支持确定性计算的具体错误，或记录该次诊断完成。这个分支不作为普通运行的数值对照。

每次有效更新仍为原定的8个协作帧；四帧真实子集、天气、输入与标签哈希、初始模型、FP32、AdamW、学习率、密度辅助损失和原UECP前向均沿用原配置。总计普通对照5次更新，严格检查至多1次更新。原训练循环、模型、损失和数据模块不改动。

额外保存第二次更新第一小批的前向输出和更新前的完整梯度，帮助定位差异出现在哪一步。CPU线程设置不改动，计算相关开关及线程数单独记录，并检查连续分支与恢复分支设置相同。

PyTorch文档说明，确定性算法模式会尽量选择确定性实现，并对已知不支持的操作报错；一次严格运行成功不能证明所有计算都完全可重复。参见版本2.5源码文档：

- https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst
- https://github.com/pytorch/pytorch/blob/v2.5.1/torch/__init__.py

## 上传与运行

上传 `uecp_gspr_controlled_resume_probe.zip` 到 `/home/cjm/`。不用解压，不覆盖项目目录，不需要Git。

```sh
ROOT=/data/cjm/datasets/logs/uecp_gspr_v2xr_v1
LOG="$ROOT/diagnostics/controlled_launcher_$(date +%Y%m%d_%H%M%S).log"
nohup /home/cjm/miniconda3/envs/opencood/bin/python -B \
  /home/cjm/uecp_gspr_controlled_resume_probe.zip --output "$ROOT" \
  > "$LOG" 2>&1 < /dev/null &
echo "PID=$! LOG=$LOG"
tail -f "$LOG"
```

工具读取原request中的路径和物理GPU，进程启动前恢复同一哈希种子；只在原GPU显存空闲至少90%时运行。控制器持有原输出的进程锁，原流水线/worker正在运行时直接退出，不抢占。临时诊断预留空间为24倍原resume.pth大小再加1GiB；不足则停止。

新内容全部写到 `OUTPUT_ROOT/diagnostics/controlled_resume_时间_进程号/`。arm目录复制必要的小型JSON元数据；数据与旧检查点只读，不访问正式test、不评AP。原源码/配置/失败记录/检查点的受保护哈希在结束时再次核对。

`RESULTS.json` 保存完整对照，日志中 `PROBE_SUMMARY_BEGIN` 到 `PROBE_SUMMARY_END` 保存可粘贴摘要。失败时记录 `ERROR.json` 和对应arm日志。

## 结果边界

本工具始终记录 `formal_recovery_gate_passed=false`。原流水线门禁、PINS、request、contract、源码快照和容差不改动。

只在独立诊断中允许已核实的旧依赖 `v2xr_raygen_e0e3/training.py` 从 b6577e... 到 cc5f52... 的完整文件版本差异，且再次检查本轮复用的四个函数和所有模块级语句与原交付版一致；其余347项依赖严格匹配。该例外不赋予原流水线继续运行的权限。

三次恢复属于极小规模工程诊断，无法建立正式统计等价性、长训练稳定性或放宽原容差的依据。本地只运行静态、文件及纯Python/NumPy合成检查，未导入Torch/OpenCOOD、读取真实PCD或执行科研实验。

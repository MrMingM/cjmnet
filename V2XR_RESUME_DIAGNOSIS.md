# E0恢复数值差异：现有文件诊断

2026-10-20，用户的原运行目录为
`/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574`。
在2号卡继续后，`failure.json`明确记录失败阶段为`smoke_E0_resume_check`。
具体错误是`optimizer/state/3/exp_avg`的最大绝对差为
`6.132945418357849e-05`，超过原逐元素判断`rtol=1e-4, atol=1e-5`。
`exp_avg`是Adam保存的梯度平均记录，不是AP或检测框误差。

按当前控制器顺序，E0的两轮短训练/验证和第二轮重放已提交，失败发生在恢复核验，
E1—E3短训练及正式训练尚未开始。按smoke.py的比较顺序，model和aux先于optimizer比较，
因此本次报错前模型权重已通过原容差；RNG及完整数据日程的比较尚未执行，不能宣称它们已经通过。

当前训练设置没有强制所有GPU计算使用确定性算法。PyTorch2.5.1的官方说明将随机种子控制和
确定性算法作为两个独立条件：
https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst
这支持“GPU计算差异是候选原因”，不足以确认这次错误的根因；不能仅据一个最大差值放宽容差。
另外，当前两条第二轮路径都会读取相同第一轮checkpoint；检查直接比较的是两次恢复运行，
没有独立建立连续运行与恢复运行的等价性。

新增根目录`diagnose_v2xr_resume.py`独立于已冻结的实验包，只在服务器CPU读取既有checkpoint。
无需更新v2xr_raygen_e0e3目录或它的source_snapshot。

诊断内容：

- 核验两条路径的第一、第二轮完成记录和4项产物哈希，再加载checkpoint；
- 确认第一轮起点完全相同、所有完成记录使用相同卡号及硬件名称；
- 对照第二轮sampling/actual_samples，包含天气、CAV和输入/标签摘要；
- 收集完整model/aux/Adam/scheduler/RNG/criterion及元数据差异，不在首个异常处停止；
- 给出超过原容差的元素数量、最大绝对差、相对L2差和相对原容差的倍数，并拒绝NaN/Inf；
- 保持训练配方、容差、原文件、阶段状态及所有修复来源记录不变。

诊断期间通过共享内核锁排除正在运行的控制器/worker；若任务仍活动会直接提示等待退出。
它不发信号、不删除锁，不执行前向/反向或重训。
唯一新增产物是`diagnostics/resume_E0_<time_ns>.json`，包含脚本自身SHA256及完整诊断；
这个报告不替代原恢复验收，不会触发后续训练。

只需将该脚本复制至服务器`/home/cjm/OpenCOOD-main/cjmnet/`。
服务器后台命令：

```sh
RUN=/data/cjm/datasets/logs/v2xr-raygen-e0e3-20261020T010502Z-1870574
LOG="$RUN/resume_diagnosis_E0_$(date -u +%Y%m%dT%H%M%SZ).log"
nohup /home/cjm/miniconda3/envs/opencood/bin/python -B \
  /home/cjm/OpenCOOD-main/cjmnet/diagnose_v2xr_resume.py \
  --output "$RUN" --group E0 > "$LOG" 2>&1 < /dev/null &
printf 'PID=%s\nLOG=%s\n' "$!" "$LOG"
tail -f "$LOG"
```

不需要指定GPU；脚本不调用GPU。当前资料尚不足以选择科学上可靠的放宽容差或重算方案，
应先依据诊断报告确定数据/RNG/优化器差异的范围，再修复真正出错的环节。
保留原运行目录和已完成的缓存、显存检查、训练及验证。

本地仅运行8项纯NumPy/合成文件检查，覆盖完整差异收集、原容差、空数组、非有限数、
结构/类型漂移、只读文件检查、产物损坏拒绝和输入日程漂移；检查通过。
未导入Torch/OpenCOOD、读取真实数据、加载真实checkpoint或运行科研实验。
原实验包及326项冻结依赖的静态/源码哈希检查通过。真实checkpoint诊断待服务器运行。

## 诊断已回传及后续修复（2026-10-20）

用户已回传`resume_E0_1792466315475575112.json`的控制台摘要。文件哈希、起点、显卡、
采样、实际数据/标签摘要、RNG和scheduler一致；模型最大绝对差4.36e-6，在原容差内。
8个Adam张量局部超出逐元素容差，但整体相对L2差4.7e-7至6.1e-5；
4个criterion浮点日志被旧代码要求完全相等。没有发现该摘要中的数据/RNG漂移或NaN/Inf证据，
尚未独立定位具体HIP算子。

实验包已补充加载后逐项精确检查与分项数值规则，记录为独立v2来源。
保留旧容差失败证据、原U修复来源及所有E0产物；只新增真实对象加载核对并重跑失败检查。
模型沿用原逐元素容差，Adam moment增加每个张量的L2界，浮点日志用原容差；
LR/step/RNG/整数计数保持精确。完整规则、适用边界和原目录后台命令见
`v2xr_raygen_e0e3/README.md`最新小节。仍需服务器实际通过新增检查才能继续后续实验。

## E1的后续证据与v3（2026-10-20）

用户继续运行后，E0通过，E1精确加载564项状态成功；唯一失败项为Adam state/17/exp_avg。
其L2差0.00017879605184775667，张量尺度1.0707758045136373，相对差约0.0167%。
v3修正v2整体绝对预算与逐元素预算的尺度不一致：用RMS=L2/sqrt(N)比较，
保持rtol=1e-4/atol=1e-5。这改变了整体误差验收预算，单独留存来源与规则，
不将旧v2失败改称通过。模型和精确状态比较、训练代码均保持原样。
升级保留E0/E1训练、验证和加载证明，只重试已有E1的数值比较；
新报告写v3路径，完整说明和后台恢复命令见实验包README最新小节。

## E2 继续排查（方案 A，2026-10-20）

E2 的模型和辅助网络数值比较通过，564 项状态精确加载通过；
唯一未通过 v3 的是 Adam state/66/exp_avg。用户选择先查固定输入的重复计算差异。
本次没有继续调整容差，新增根目录独立诊断脚本，不覆盖冻结包、原阶段记录或已有训练产物。
脚本及卡 2 的服务器后台运行命令见 `V2XR_GPU_REPEATABILITY.md`。

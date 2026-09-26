# 方案 A：冻结选择器的未用验证帧复验

这一步直接检验第二轮小实验的选择器是否值得继续研究。它**不重新训练**，也不访问 OPV2V-W。脚本读取第二轮已经完成的 `selector.pth`，核对对应的 v3 checkpoint 和源码哈希。报错表明第二轮已用到 validation 的全部场景（配置最多抽 12 个，实际总数以运行 manifest 为准），无法再做场景隔离；本版排除第二轮用过的每一帧及其前后各 2 帧，在余下的验证帧中每场景最多固定抽 24 帧。clean、fog、rain、snow 用同一批帧。

固定候选生成、v3 residual / peer-query 动作、tile 尺寸和阈值。比较原融合模型、学习排序 top-1/top-4、简单置信度排序 top-1/top-4、学习排序允许 KEEP，以及**每帧执行与学习 KEEP 完全相同动作数**的置信度排序。最后两者是选择方法的关键公平对照；动作数量相同仍可能使用不同动作类型，因此同时记录每种方法实际推理耗时。最终按整帧 NMS 后 AP30/AP50/AP70 及恢复、原 TP 损失、新增 FP 身份统计。

本次复验的事先判读规则：若学习 KEEP 在 Snow 上比配平的简单置信度排序至少高 **0.5 个 AP70 百分点**，并且 Clean/Rain 相对原融合的 AP70 降幅都不超过 **0.2 个百分点**，且原 TP 损失、新增 FP 不显著恶化，方案 A 才值得投入完整方法开发；否则优先转向简单保护型方案。这个门槛只是研究资源决策规则，不是统计显著性保证。逐场景 AP70 只用于查看收益是否集中；由于场景与第二轮相同，不能把它解释为跨场景泛化。Fog 也报告，但不作为事后调整门槛的理由。若置信度配平对照与学习 KEEP 差不多，不能声称学习选择器比简单已有规则有价值。

这是此前没用过的 **validation 帧**，比第二轮 72 帧大，但与之共享全部或部分场景，且官方验证划分已反复用于研究；不称为独立测试。排除邻近帧只能减少直接时序重叠，不能消除同场景相关性。不能依据这次结果反复调门槛后在同一场景上宣布验证成功。

## 服务器后台运行

先同步本目录到 `/home/cjm/OpenCOOD-main/cjmnet`。先前报错的运行目录不含复验结果，不要在它上面 `RESUME=1`；请新建 `RUN`。下面第一行查找已完成的第二轮运行目录；请核对打印的路径确实是刚才那次 6876 训练动作的运行。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export PILOT_V2_RUN=$(find /data/cjm/datasets/logs -mindepth 2 -maxdepth 2 -name decision_pilot_v2_results.json -printf '%h\n' | sort | tail -n 1)
echo "PILOT_V2_RUN=$PILOT_V2_RUN"
export V3_RUN=/data/cjm/datasets/logs/local_fusion_v3_20260922_182832
export RUN=/data/cjm/datasets/logs/decision_confirm_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_decision_confirm/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
```

如果查找命令得到空值或不是刚才的运行，手动把 `PILOT_V2_RUN` 设成正确目录。运行中看 `tail -f "${RUN}.launcher.log"` 或 `$RUN/progress.json`。中断可用相同的 `RUN`，加 `RESUME=1` 重启；已完成的帧会跳过。结果是 `$RUN/confirmation_results.json` 和各天气的 `*_confirmation.json`，源码快照与抽样场景保存在运行目录中。

同一服务器/GPU，第二轮约半小时；本轮不训练和制单动作标签，但场景数更大、仍需逐动作跑检测头。预估 **45–90 分钟**，若保留场景接近上限或天气生成变慢，可能约 **2 小时**。跑完第一个天气的 20–40 帧后，可用日志里的帧速率更准确估计。新增 FP 是相对原输出出现的新 FP 身份，不是总 FP 净增量。计时排除数据加载和 GT 指标计算，包含候选动作预计算以及各方案的检测头/后处理。

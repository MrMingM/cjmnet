# K2：仅禁用 E 候选的排列修复

fog/frame608/target11 的六个候选在新旧轨迹中全部属于 E、全部 `allowed=false`，全部 outcome8 等于当步 KEEP；最终均选 KEEP。source 对及数量相同，query:0/query:1 在列表中互换。这些候选不可能被 A_ONLY 采用，换位不改变该步更新。

这是额外的逐字段复现检查对无效列表排列的误报。§42 的扰动、样本资格、稳定性阈值和 AP 门禁均保持原样。本工具不把 K2 判为 PASS，也不修改原动作排序或标签。

## 严格限定

先执行 v1 原检查。只有拒绝项全部是候选审计里的 cls_source/reg_source 名称时，才检查例外。每个涉及的目标必须同时满足：

- 新旧都选 KEEP；selected_key 等于自己的 keep_key，所选记录的 cls/reg 均为 KEEP、changed=false、expansion=null。
- 新旧**全部**候选都是 E、allowed=false，每个 outcome8 与自己当步 KEEP 完全相同。
- 新旧 source-pair 多重集合完全相同，保留重复次数。
- 只在比较用的副本中恢复旧候选名称后，完整 v1 检查通过，包括全部其他字段、原 S0 数值约定和 full9 排序。

原始推理、列表、source、ROI、GT、标签不改；实际新旧轨迹及所有差异仍保存。A/B、合法候选、胜者、候选数量、数值超限、其他排序/字段变化一律保留拒绝。该证明不要求给重复 source 任意配 ROI：全部禁用动作均不能改变状态。

## 来源与缓存

只同步本新目录，不改原 Stage 2、v1 修复、历史模型/报告或 AI_CONTEXT。新 RUN 从 v1 协议继承所有科学设置、数据、源代码和输入；追加本目录的源码哈希及这项窄工程例外，在计算前冻结协议。

启动时核对用户已检查的 608 帧证据 SHA256。前一 RUN 的四天气×Same/Task Source 复现必须 complete，八项条件凭证和原始数据哈希全部核验，并重新比较历史 AP30/50/70，容差仍 1e-6。通过后仅引用这些凭证，不重复 Source 前向。前一 RUN 的相关产物哈希加入新协议，必须保持旧 RUN 原样。

旧 K2 部分结果不移植。新 RUN 中 K2 仍按帧原子保存/resume，记录原始差异与例外证明。旧 K1 继续使用原 RUN，完成后按原停止逻辑汇总。K3 继续冻结。

## 服务器后台启动

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
conda activate opencood
export ROCR_VISIBLE_DEVICES=3
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export PREVIOUS_RUN=/data/cjm/datasets/logs/source_kill_stage2_k2_repair_20261014_154622
unset RUN
sh local_fusion_source_kill_stage2_replay_fix_v2/launch.sh
```

记录输出的新 RUN。查看日志和状态、恢复：

```sh
tail -f <NEW_RUN>/driver.log
python -m local_fusion_source_kill_stage2_replay_fix.status --run <NEW_RUN>
RUN=<NEW_RUN> sh local_fusion_source_kill_stage2_replay_fix_v2/launch.sh
```

Same GPU/software environment required; 原 resource stop 不变。不要重启旧 repair RUN 或修改其已完成 Source 产物。

原 K1 和新 K2 都完成后，可再次执行 CPU 汇总：

```sh
python -m local_fusion_source_kill_stage2_replay_fix_v2.driver --run <NEW_RUN> --summarize-only
```

本地仅检查代码和纯虚构记录，科研计算只在服务器执行。输出仍为 PROTOCOL/SOURCE_SNAPSHOT、K2_LABEL_STABILITY、k2_flip_records、bootstrap、逐帧 reproduction 证明及 STAGE2_FINAL；所有大结果放外部 RUN。

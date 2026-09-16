# 第一轮无收益后的固定支持诊断

只使用已训练的 reviewer_best.pth，不更新任何网络或覆盖原结果。上传更新后的 gspr_review 后运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_diagnostics.sh
```

脚本仅在日志目录下找到**唯一一个**有 reviewer_best.pth 的 gspr_review_seed* 训练目录时自动使用它。若存在多个，会列出候选并停止，避免选错模型；使用 `--run-dir` 指定其中的完整训练目录。也可加 `--max-frames 5` 先检查 5 帧/分支，再去掉限制跑完整开发集。前端路径读取该次训练 manifest，模型配置读取该次保存的 YAML。资源位置仍支持 FOG_LOOKUP_DIR。

## 对照怎样做到公平

每个 Clean/Weather 样本只执行一次真实通信与正常推理，捕获已解码的证据、进入复核 MLP 的输入和原本可修改点的掩码。之后在原掩码上重放 MLP，固定本地点信息、请求、支持范围及消息载荷：

- normal：原始证据。重放权重必须与模型实际输出一致，否则报错。
- permuted：在有自车待复核点的有效几何格之间循环交换四维证据。相同格内多个点仍使用同一证据。交换保持证据集合不变，但打破位置关联。
- constant：有效格证据的均值，移除位置相关的内容差异。
- zero：证据输入清零，仍强制保留原始可修改点掩码。此为离线敏感性测试，**不代表部署时没有消息也可以修正点**；清零组合可能超出训练分布。

这些都是对固定已收到消息的事后分析，不是新增通信模式，不重新评估 AP，也不能将其推断为可部署新模型的性能。诊断不改动 model.py、evidence.py 或训练脚本。

记录 `mean_peer_input_change_vs_normal`，以避免实际各格证据近乎一致、根本没有充分扰动时错误认定网络不依赖证据。

## 目标附近与柱特征

默认使用 GT 旋转 BEV 框向外扩展 1 米，按点 XY 位置划分目标附近；按柱中心 XY 位置划分柱所在区域。两种分组略有边界差异，且不限制高度。“outside_target_near”只是标注目标邻域外，不等同于有真值保证的纯背景或噪声。GT 只用于统计，不传入模型/修正器。

对每组记录有效点数、可修改点数、上调/下调点数、权重改动和相对正常证据的差异。重新运行冻结 PillarVFE，比较修正前后的柱特征，报告可修改柱中实际变化的比例、绝对变化及相对 L1 变化。还重复一次未修改权重的 PillarVFE，报告基线重复误差；变化应结合该误差及阈值解释。默认判定阈值 1e-6，可通过 `--epsilon` 修改。

输出目录为原训练目录加 `_diagnostics_时间戳`，包含 protocol.json、逐帧 frames.jsonl 和 summary.json。汇总采用有效点/特征元素数加权，不用逐帧均值再次平均。没有有效样本的均值为 null。

## 怎样读 summary.json

每个分支的 `variants.normal.all` 是正常复核的整体情况，`target_near` 与 `outside_target_near` 是分组情况。

1. 先看干预组 `mean_peer_input_change_vs_normal`：输入是否真的变了。
2. 再看 `all.mean_abs_change_vs_normal`：输入变了，输出权重是否跟着变化。将其与正常组 mean_abs_weight_change 比较。不能仅凭两者都小就声称网络完全忽略证据。
3. 对比正常组目标附近与外部的 changed_points、eligible_points：修正集中在哪里，目标附近本来是否有可修正点。
4. 看正常组 changed_pillar_fraction 和 relative_pillar_l1_change：点权重修改是否传递到柱特征。柱特征变了也不代表变得更好；此诊断不提供点级正确率或因果归因。

完成后提供 summary.json 即可进一步判断。单元测试中的 NumPy 区域/汇总测试可在本地运行；固定支持张量测试与真实重放一致性检查需要服务器 PyTorch。本地未执行真实模型诊断。

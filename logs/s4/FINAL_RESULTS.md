# Direction B S4 Source Evidence Audit

## 一句话结论

Geometry evidence 本轮没找到足够稳定的信号；Semantic evidence 本轮没找到足够稳定的信号。**Case 5**：本轮冻结证据与线性模型均未达到门槛；不能据此宣称来源效用不可学习。下一步检查非线性交互与联合动作。

GEO：GEO_EVIDENCE_WEAK；SEM：SEM_EVIDENCE_WEAK；Proposal：REPLAY_FAIL。Assisted-Task 恶劣天气平均 ΔAP70 = +0.1754 pp。

## S4-GEO

| Features | Fog Pair Acc | Rain | Snow | Mean | Regret | Verdict |
|---|---:|---:|---:|---:|---:|---|
| output_only | 0.815776 | 0.812693 | 0.831736 | 0.820068 | 0.127281 | GEO_EVIDENCE_WEAK |
| output_plus_geometry | 0.810174 | 0.809379 | 0.828876 | 0.816143 | 0.126536 | GEO_EVIDENCE_WEAK |
| output_plus_semantic | 0.812727 | 0.810083 | 0.828760 | 0.817190 | 0.126491 | GEO_EVIDENCE_WEAK |
| output_plus_gspr | 0.814802 | 0.811909 | 0.830661 | 0.819124 | 0.126311 | GEO_EVIDENCE_WEAK |
| output_plus_geometry_semantic | 0.809647 | 0.807121 | 0.827471 | 0.814746 | 0.126392 | GEO_EVIDENCE_WEAK |
| all_evidence | 0.807380 | 0.804352 | 0.826132 | 0.812621 | 0.126953 | GEO_EVIDENCE_WEAK |
| all_evidence_without_loo | 0.809759 | 0.807234 | 0.827603 | 0.814865 | 0.127046 | GEO_EVIDENCE_WEAK |
| all_evidence_without_gspr | 0.809647 | 0.807121 | 0.827471 | 0.814746 | 0.126392 | GEO_EVIDENCE_WEAK |

## S4-SEM

| Features | Fog Pair Acc | Rain | Snow | Mean | Regret | Verdict |
|---|---:|---:|---:|---:|---:|---|
| output_only | 0.845267 | 0.848923 | 0.836690 | 0.843626 | 0.175226 | SEM_EVIDENCE_WEAK |
| output_plus_geometry | 0.844399 | 0.849679 | 0.835177 | 0.843085 | 0.176500 | SEM_EVIDENCE_WEAK |
| output_plus_semantic | 0.853682 | 0.860607 | 0.841115 | 0.851801 | 0.155693 | SEM_EVIDENCE_WEAK |
| output_plus_gspr | 0.846863 | 0.851377 | 0.837899 | 0.845380 | 0.165068 | SEM_EVIDENCE_WEAK |
| output_plus_geometry_semantic | 0.854829 | 0.862598 | 0.839205 | 0.852211 | 0.147179 | SEM_EVIDENCE_WEAK |
| all_evidence | 0.852334 | 0.859573 | 0.838807 | 0.850238 | 0.151047 | SEM_EVIDENCE_WEAK |
| all_evidence_without_loo | 0.851714 | 0.857242 | 0.840176 | 0.849711 | 0.155564 | SEM_EVIDENCE_WEAK |
| all_evidence_without_gspr | 0.854829 | 0.862598 | 0.839205 | 0.852211 | 0.147179 | SEM_EVIDENCE_WEAK |

## Evidence ablation

主配置固定 all_evidence。output_only、+geometry、+semantic、+gspr、+geometry+semantic、all-no-LOO 和 all-no-GSPR 均采用同一训练配方。

| Task/comparison | Pairwise gain | Regret difference | Relative regret reduction |
|---|---:|---:|---:|
| GEO/all_evidence-vs-all_evidence_without_loo | -0.002244 | -0.000093 | 0.000733 |
| SEM/all_evidence-vs-all_evidence_without_loo | 0.000527 | -0.004517 | 0.029034 |
| GEO/all_evidence-vs-all_evidence_without_gspr | -0.002125 | 0.000561 | -0.004439 |
| GEO/output_plus_gspr-vs-output_only | -0.000944 | -0.000970 | 0.007621 |
| SEM/all_evidence-vs-all_evidence_without_gspr | -0.001973 | 0.003868 | -0.026283 |
| SEM/output_plus_gspr-vs-output_only | 0.001753 | -0.010158 | 0.057969 |

## Replay

| Method | Clean AP70 | Fog | Rain | Snow | adverse ΔAP70 pp | recovered/lost/newFP | changed |
|---|---:|---:|---:|---:|---:|---|---:|
| Shared | 0.822254 | 0.785814 | 0.812677 | 0.601223 | 0.000000 | 0/0/0 | 0 |
| Evidence-Assisted-Cls | 0.823646 | 0.787343 | 0.812926 | 0.600294 | 0.028284 | 11/9/20 | 2595 |
| Evidence-Assisted-Reg | 0.822849 | 0.785814 | 0.812677 | 0.604846 | 0.120770 | 5/0/0 | 772 |
| Evidence-Assisted-Task | 0.824239 | 0.787343 | 0.812926 | 0.604706 | 0.175352 | 16/9/19 | 2626 |
| Evidence-Proposal-Cls-Greedy | 0.817099 | 0.779646 | 0.806460 | 0.593726 | -0.662766 | 19/35/89 | 8155 |
| Evidence-Proposal-Reg-Greedy | 0.820828 | 0.785059 | 0.812677 | 0.603476 | 0.049902 | 5/2/2 | 1784 |
| Evidence-Proposal-Task-Greedy | 0.815726 | 0.778683 | 0.806364 | 0.597680 | -0.566243 | 24/36/90 | 8289 |
| Evidence-Proposal-Task-Conservative | 0.822254 | 0.785814 | 0.812677 | 0.601223 | 0.000000 | 0/0/0 | 0 |

计数汇总 Fog/Rain/Snow；Clean 单独受掉点门槛约束。全部天气 AP30/50/70、动作数、冲突和 cell 数见 S4_REPLAY_RESULTS.md/json。

## 机制判断

GEO_EVIDENCE_WEAK / SEM_EVIDENCE_WEAK / REPLAY_FAIL。主方法没有按 validation 重新选择。标准化权重只表示模型依赖或关联，不能解释为因果。

| Task | output | geometry | gspr | semantic | leave-one-out | single-query | competition |
|---|---:|---:|---:|---:|---:|---:|---:|
| GEO | 2.684902 | 1.158131 | 4.848048 | 5.747897 | 4.246657 | 2.172455 | 0.993123 |
| SEM | 7.513434 | 1.912769 | 3.336523 | 9.156234 | 3.931068 | 3.022374 | 3.015549 |

## 目前能得出什么结论

本轮冻结证据与线性模型均未达到门槛；不能据此宣称来源效用不可学习。下一步检查非线性交互与联合动作。

## 还不能得出什么结论

不能把线性失败解释为不可学习，不能把删除后的变化幅度解释为正确贡献，不能从权重大小推出因果。不能把单动作排序提升直接等同于联合 AP 收益，也没有正式 test 结论。

## 下一步应该验证什么

本轮冻结证据与线性模型均未达到门槛；不能据此宣称来源效用不可学习。下一步检查非线性交互与联合动作。

完整 ranking、LOO/GSPR 消融和整帧 AP 分别见 S4_GEO_RESULTS.md、S4_SEM_RESULTS.md、S4_ABLATIONS.md、S4_REPLAY_RESULTS.md。

本轮只增加来源证据。旧 S0 标签、候选、动作、fit/calibration/validation 场景、线性层、30 个 epoch 和训练超参数保持一致；R0 先复现旧 S1/S2。归一化只看 fit，阈值只看 train calibration。

GT 用于旧标签、Assisted 候选关联和最终评价；inference feature GT=0。S4 不访问正式 test。使用在线天气，不能当作正式固定天气 benchmark。

几何统计使用 Shared 预测框中的保留点和 pillar；没有重建体素化丢弃的原始点。空 ROI、零向量 cosine、无其他来源和不可定义的移除均有 validity 标记。B0 learned Shared 的 LOO 与原始 AttFuse LOO 分开记录。


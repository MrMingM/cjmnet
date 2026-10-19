# V2X-R Clean训练AttFuse：最终天气validation结果

本报告依据已归档的服务器RESULTS.md、summary.json及训练选择记录整理；属于D−1D新训练模型，日志中的“D-1C replay”表示复用冻结评测器。
已核对summary.status=complete、full_validation=true、每条件829帧、权重与训练/GT身份关联、RESULTS.md哈希及AP/GT计数计算。未重新运行模型或读取真实点云/预测缓存。

## 模型与数据

- 模型：官方LiDAR AttFuse，随机初始化，在V2X-R Clean train上训练20轮，实际8083协作帧；公开8084与实际8083的差值此前已明确接受。
- 选模：只看完整Clean validation F AP70，选中epoch20；Fog/Snow不参与训练或epoch选择。
- validation：6场景、829个配对协作帧；官方已有fog_0.060、snow_2.5_2.0点云，没有Rain条件。
- 选中权重：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1/net_epoch20.pth`。
- 权重SHA256（来自已回传selected.json）：`ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3`。
- 服务器最终输出：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1/final_evaluation/`。
- 指标：BEV IoU，官方VOC2010 AP，global_sort=False；表中AP为0–1，pp为百分点。

## 三种评测协议

- **O**：每种天气使用其官方GT；prediction=None时按官方代码跳过整帧。
- **O***：GT与O相同；prediction=None按零预测计入，GT保留在分母。
- **F**：三条件始终评测同帧官方Clean选择出的相同目标，天气不再删除这些GT；prediction=None按零预测计入。作为本研究天气鲁棒性主结果。

三协议复用同一模型预测，差异来自GT集合与None帧处理；F仍继承官方Clean的FOV/范围等条件，不等于原始所有YAML目标。

## AP30 / AP50 / AP70

| 条件 | 协议 | AP30 | AP50 | AP70 |
|---|---|---:|---:|---:|
| clean | O | 0.87655091 | 0.84707141 | 0.70539766 |
| clean | O* | 0.87612841 | 0.84666312 | 0.70505766 |
| clean | F | 0.87612841 | 0.84666312 | 0.70505766 |
| fog | O | 0.56354597 | 0.55077779 | 0.44212603 |
| fog | O* | 0.55480050 | 0.54223046 | 0.43526483 |
| fog | F | 0.32740717 | 0.31998915 | 0.25686499 |
| snow | O | 0.46624621 | 0.43523409 | 0.28892218 |
| snow | O* | 0.45708695 | 0.42668405 | 0.28324639 |
| snow | F | 0.42991436 | 0.40131884 | 0.26640816 |

## GT、空预测与预测框计数

| 条件 | 官方GT | 固定GT | 天气过滤GT | None帧 | None帧内官方GT | None帧内固定GT | 预测框总数 | O实际GT分母 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 6224 | 6224 | 0 | 43 | 3 | 3 | 6718 | 6221 |
| Fog | 3673 | 6224 | 2551 | 164 | 57 | 169 | 2239 | 3616 |
| Snow | 5854 | 6224 | 370 | 114 | 115 | 117 | 3482 | 5739 |

Fog/Snow分别过滤固定Clean目标的40.9865%/5.9447%。GT是跨帧目标实例累计数，不等于去重车辆数。
O*分母为6224/3673/5854，F各6224；O跳过None帧后分母如表。None帧数不能直接解释为漏检目标数量。
Snow另有1个empty non-None帧，仍计入官方AP，不属于114个None帧；Clean/Fog没有此类帧。
Fog/Snow被过滤目标的removed_detected在IoU .3/.5/.7均为0；仅描述该权重在本次一对一匹配下的结果。

## AP70与天气下降

天气下降定义为Clean AP70减天气AP70，再换算为百分点。

| 协议 | Clean AP70 | Fog AP70 | Snow AP70 | Clean−Fog / pp | Clean−Snow / pp |
|---|---:|---:|---:|---:|---:|
| O | 0.70539766 | 0.44212603 | 0.28892218 | 26.3272 | 41.6475 |
| O* | 0.70505766 | 0.43526483 | 0.28324639 | 26.9793 | 42.1811 |
| F | 0.70505766 | 0.25686499 | 0.26640816 | 44.8193 | 43.8649 |

## 评测协议对AP70的影响

| 天气 | 跳过None帧 O−O* / pp | 改变天气GT集合 O*−F / pp | 固定GT比官方多显示的天气下降 / pp |
|---|---:|---:|---:|
| fog | 0.6861 | 17.8400 | 18.4921 |
| snow | 0.5676 | 1.6838 | 2.2174 |

最后一列为F天气下降减O天气下降，包含Clean基准自身约0.0340 pp的O−F差异，不等于直接O−F。

## 当前结论与证据边界

1. 固定同一批目标后，Clean AP70=70.5058%，Fog=25.6865%，Snow=26.6408%，下降44.8193/43.8649 pp。本次Clean域内训练后仍有明显天气退化。
2. Fog官方O AP70=44.2126%，F为25.6865%。保持同一预测和None处理时，O*与F的差为17.8400 pp；本模型下，天气GT集合变化对Fog评分的影响较大。
3. Snow的O*−F为1.6838 pp；跳过None帧的影响Fog/Snow为0.6861/0.5676 pp。以上是给定模型预测下的描述性协议差异，不解释为可直接获得的模型性能提升。
4. 本结果是单种子validation结果；未评测正式test。尚无同协议OPV2V权重对照，不能由旧v2xr_attfuse分数直接推算域内训练收益或域偏移贡献。
5. None/GT/预测计数直接取自已归档summary，不由AP反推。Fog保留框最大角点差为2.384185791015625e-07 m，未逐位相同；Clean/Snow逐位一致，保留该浮点差异记录。

## 归档与核验范围

- 原始报告：同目录RESULTS.md，SHA256=`bec4695339c2429c3603dc80a81f53e17f31238b2a33916fccdc890466696a06`，与summary记录一致。
- 最终summary：同目录summary.json，SHA256=`89312cdef48b02443455b411f9f4adba63f89b4424900c8f017410720276ac34`。
- 冻结D−1C GT身份：`d5244c4a8fa78dc1d26df1a7ee1f4592c1dd24f91a3f9c575313a081bbeab776`，与已归档训练protocol一致。
- 最终summary中checkpoint与selected.json一致，关联training_contract哈希一致；9组AP的天气gap及GT分母计算核对一致。
- 未读取实际checkpoint、最终逐帧预测/GT manifest或重算匹配；服务器保留这些原始证据。同目录protocol.json为训练协议，不能冒充最终评测protocol.json。

本文件为人工整理报告，不替代服务器原始RESULTS.md或完成标记。


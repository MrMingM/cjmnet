# D−1C：同一预测下的官方评测协议差异

范围：每条件 829/829 帧；**READY_FOR_INTERPRETATION**。
同一个官方 V2X-R LiDAR AttFuse checkpoint；每条件每帧只生成一份最终预测，O/O*/F共用其缓存。
官方 dataset/model/native ROI/postprocess/eval 源码保持原样。没有训练或阈值调整。

| Condition | Official GT | Fixed GT | Removed by weather | None frames | Official GT in None frames | Fixed GT in None frames |
|---|---:|---:|---:|---:|---:|---:|
| clean | 6224 | 6224 | 0 | 43 | 1 | 1 |
| fog | 3673 | 6224 | 2551 | 221 | 165 | 323 |
| snow | 5854 | 6224 | 370 | 98 | 111 | 112 |

| Condition | Protocol | AP30 | AP50 | AP70 | Counted GT |
|---|---|---:|---:|---:|---:|
| clean | O | 0.85993605 | 0.82279434 | 0.66625452 | 6223 |
| clean | O* | 0.85979789 | 0.82266214 | 0.66614748 | 6224 |
| clean | F | 0.85979789 | 0.82266214 | 0.66614748 | 6224 |
| fog | O | 0.44374745 | 0.41430254 | 0.27543253 | 3508 |
| fog | O* | 0.42381324 | 0.39569107 | 0.26305944 | 3673 |
| fog | F | 0.25010701 | 0.23351114 | 0.15524057 | 6224 |
| snow | O | 0.46365775 | 0.42977816 | 0.27207667 | 5743 |
| snow | O* | 0.45486615 | 0.42162897 | 0.26691772 | 5854 |
| snow | F | 0.42782559 | 0.39656426 | 0.25105018 | 6224 |

下表每格为 **AP差 / 百分点pp**，保留正负号。

| Condition | Metric | AP30 / pp | AP50 / pp | AP70 / pp |
|---|---|---:|---:|---:|
| clean | NoneSkipEffect | +0.00013816 / +0.0138 | +0.00013220 / +0.0132 | +0.00010705 / +0.0107 |
| clean | GTFilterEffect | +0.00000000 / +0.0000 | +0.00000000 / +0.0000 | +0.00000000 / +0.0000 |
| fog | NoneSkipEffect | +0.01993420 / +1.9934 | +0.01861147 / +1.8611 | +0.01237309 / +1.2373 |
| fog | GTFilterEffect | +0.17370623 / +17.3706 | +0.16217994 / +16.2180 | +0.10781887 / +10.7819 |
| fog | OfficialWeatherGap | +0.41618860 / +41.6189 | +0.40849180 / +40.8492 | +0.39082199 / +39.0822 |
| fog | FixedWeatherGap | +0.60969088 / +60.9691 | +0.58915100 / +58.9151 | +0.51090691 / +51.0907 |
| fog | GapMasking | +0.19350227 / +19.3502 | +0.18065921 / +18.0659 | +0.12008491 / +12.0085 |
| snow | NoneSkipEffect | +0.00879160 / +0.8792 | +0.00814919 / +0.8149 | +0.00515895 / +0.5159 |
| snow | GTFilterEffect | +0.02704056 / +2.7041 | +0.02506470 / +2.5065 | +0.01586754 / +1.5868 |
| snow | OfficialWeatherGap | +0.39627830 / +39.6278 | +0.39301618 / +39.3016 | +0.39417785 / +39.4178 |
| snow | FixedWeatherGap | +0.43197230 / +43.1972 | +0.42609788 / +42.6098 | +0.41509730 / +41.5097 |
| snow | GapMasking | +0.03569400 / +3.5694 | +0.03308170 / +3.3082 | +0.02091944 / +2.0919 |

删除/保留目标的检出率：都取同一次 Protocol F 全部 GT 的官方一对一匹配；同一预测不会在两组重复使用。

| Condition | GT group | Total | IoU≥0.3 detected / recall | IoU≥0.5 detected / recall | IoU≥0.7 detected / recall |
|---|---|---:|---:|---:|---:|
| fog | removed | 2551 | 0 / 0.0000% | 0 / 0.0000% | 0 / 0.0000% |
| fog | retained | 3673 | 1584 / 43.1255% | 1534 / 41.7642% | 1250 / 34.0321% |
| snow | removed | 370 | 0 / 0.0000% | 0 / 0.0000% | 0 / 0.0000% |
| snow | retained | 5854 | 2864 / 48.9238% | 2737 / 46.7544% | 2121 / 36.2316% |

协议精确定义：
- O / OFFICIAL_EXACT：使用该条件原版最终GT，预测None时整帧跳过；非None空数组正常计分。
- O* / OFFICIAL_GT_NO_SKIP：使用与O逐帧同一GT，None视为0预测，GT正常进入分母。
- F / OFFICIAL_CLEAN_FIXED_GT：固定同帧官方Clean已选择的目标ID和几何，所有天气使用同一GT；None计0预测且保留GT分母。
三者均直接调用固定版本eval_utils_opv2v，BEV IoU、VOC2010、global_sort=False、原帧顺序和逐帧分数排序。

NoneSkipEffect = O − O*；GTFilterEffect = O* − F。
OfficialWeatherGap = Clean_O − Weather_O；FixedWeatherGap = Clean_F − Weather_F。
GapMasking = FixedWeatherGap − OfficialWeatherGap。
因此 GapMasking = Weather_GTFilterEffect + Weather_NoneSkipEffect − Clean_NoneSkipEffect。
GapMasking同时包含两种协议差异，不能全部归因于GT过滤；GT过滤的直接对照是O*−F。

结果理解：
- fog AP70：None跳帧差 +1.2373 pp；GT集合差 +10.7819 pp；天气下降被遮蔽量 +12.0085 pp。
- snow AP70：None跳帧差 +0.5159 pp；GT集合差 +1.5868 pp；天气下降被遮蔽量 +2.0919 pp。
正且较大的GTFilterEffect和GapMasking支持官方天气评测明显低估在固定Clean目标上的退化。
若三档GTFilterEffect与GapMasking均接近0，则即使删除目标很多，当前预测下的AP实际协议差异仍小；需结合removed/retained recall与FP变化解释。
删除目标检出率明显大于0，说明当前预测中部分零框内点目标仍被检出；本实验无法据此确定具体依赖了上下文还是协同来源。
删除目标几乎全漏检也不能推出AP影响小：这些漏检是否进入分母仍可能显著影响AP。
不预设效应正负，不以删除GT比例代替AP效应，也不把描述性差异称为因果统计估计。

完整性检查：原版推理循环O与缓存回放O的TP/FP/GT/分数序列及AP一致；Clean O*=F；所有GT含None帧。
D−1A full Clean重复对齐容差预先固定1e-6；若未对齐，先检查运行证据，不更改GT定义/阈值或重生成已缓存预测。
D−1A Clean差异（当前O−历史D−1A）：{'ap30': {'AP': 0.0, 'pp': 0.0}, 'ap50': {'AP': 0.0, 'pp': 0.0}, 'ap70': {'AP': 0.0, 'pp': 0.0}}。
每帧共同目标的几何容差1e-4m；数值差与逐位一致状态见summary.results.*.counts。
保留GT原始顺序；匹配分数与最大IoU并列均遵循官方实现，不另设ID优先规则。
逐目标匹配与预测文件哈希见per_frame_metrics.jsonl、removed_gt_metrics.jsonl和prediction_manifest.json。

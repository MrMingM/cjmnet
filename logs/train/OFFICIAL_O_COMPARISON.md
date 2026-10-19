# V2X-R官方O协议：重训AttFuse与作者结果对照

本次输入为用户归档的OFFICIAL_O.md：FULL VALIDATION，每条件829帧，epoch20，来源inference_full/summary.json，权重SHA256为ccf0211d04f732246d53b70faf8c3f6eb494c19963177d9cedf3d2b106cf01d3。
本次报告中的9项AP按8位小数与上一轮logs/train/summary.json的O完全一致。
只读取报告和历史JSON/官方源码快照并计算差值，没有重新运行实验。
当前未回传该独立重新推理目录的OFFICIAL_O.json/summary.json，不声称已独立核验那轮原始缓存；上一轮D−1D完整summary已归档。

## 一、作者公布的普通LiDAR AttFuse Clean validation

官方冻结README的LiDAR-based Cooperative 3D Detector表，Validation列报告86.69/82.58/66.56。
本次也是validation，可进行分数对照。单位均为百分数，差值为本次减作者参考、单位pp。

| 指标 | 作者公布validation / % | 本次 / % | 差值 / pp |
|---|---:|---:|---:|
| AP30 | 86.6900 | 87.6551 | 0.9651 |
| AP50 | 82.5800 | 84.7071 | 2.1271 |
| AP70 | 66.5600 | 70.5398 | 3.9798 |

同表Testing列为91.21/89.51/80.01；本次没有运行正式test，不能拿validation AP70=70.54%与作者test=80.01%作为同集复现偏差。

## 二、作者发布的普通Clean训练checkpoint：同协议、同829帧复测

作者权重的结果来自logs/D/D1/RESULTS.md及summary.json的O，权重SHA256=4e92bff533cba4d8e2020f4176bd9f684b5a2337d59f853535f612edd6aa3e94。
这是我们用作者权重复测得到的结果，不冒称作者论文/README逐天气自行报告的原值。
两者使用相同验证帧、官方O GT/None处理、模型/输入/后处理/AP；不同权重导致None帧和最终计分分母可以不同，这是O协议本身的行为。

| 天气 | 指标 | 作者权重复测 / % | 本次重训 / % | 本次−作者权重 / pp |
|---|---|---:|---:|---:|
| clean | AP30 | 85.9936 | 87.6551 | 1.6615 |
| clean | AP50 | 82.2794 | 84.7071 | 2.4277 |
| clean | AP70 | 66.6255 | 70.5398 | 3.9143 |
| fog | AP30 | 44.3747 | 56.3546 | 11.9799 |
| fog | AP50 | 41.4303 | 55.0778 | 13.6475 |
| fog | AP70 | 27.5433 | 44.2126 | 16.6694 |
| snow | AP30 | 46.3658 | 46.6246 | 0.2588 |
| snow | AP50 | 42.9778 | 43.5234 | 0.5456 |
| snow | AP70 | 27.2077 | 28.8922 | 1.6846 |

本次官方O AP70的Clean−Fog/Snow下降为26.3272/41.6475 pp；
作者权重为39.0822/39.4178 pp。Fog天气下降缩小，Snow的下降更大，同时Snow绝对AP70仍提高1.6846 pp，
原因是本次Clean改善幅度比Snow更大；不能只看gap而忽略绝对AP。

## 三、作者README另一组天气结果：包含Fog模拟训练

冻结README章节名为“Multi-modal Diffusion Denoising Results (AttFuse train with fog simulation)”。
其中纯LiDAR行：Snow AP30/50/70=68.73/65.35/45.54%，Fog=81.71/78.48/61.52%，Normal=89.30/87.39/76.09%。
表头未在该天气表明确标注validation/test，不能凭相邻表或数值猜测split。
本次模型只用Clean训练，两组训练条件不同；下表仅列数字差距，不视为相同设置的复现误差。

| 条件 | README使用Fog模拟训练的LiDAR AP70 / % | 本次Clean训练 AP70 / % | 数值差 / pp |
|---|---:|---:|---:|
| normal | 76.0900 | 70.5398 | -5.5502 |
| fog | 61.5200 | 44.2126 | -17.3074 |
| snow | 45.5400 | 28.8922 | -16.6478 |

这组差距不能解释为“当前官方评测代码仍然错误”，也不能断言补上Fog训练一定能达到该值。
目前支持的结论是本次模型在相同829帧官方协议下优于作者发布的普通Clean训练权重，Fog提高明显；
尚未复现作者Fog模拟训练实验，不声称超过作者全部方法。

## 来源与解释边界

- 本次结果：logs/train/OFFICIAL_O.md；SHA256=6913d355b8e0d720de6bfa78152a81d672acd279f48d3236736246005f1a0f2a。
- 作者权重同集复测：logs/D/D1/RESULTS.md、summary.json的O。
- 作者公布表：v2xr_official_repro/upstream/readme.md的LiDAR表及Fog模拟训练章节。
- 上游：https://github.com/ylwhxht/V2X-R/blob/f3ebbad95561444a53a17358735b0f0dc4ba89f7/readme.md 。
- 本次为单种子、Clean validation选模的结果；分数提高不证明某个新方法有效，也不建立统计显著性或保证test泛化。


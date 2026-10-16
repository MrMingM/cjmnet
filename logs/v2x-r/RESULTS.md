# D−1A：官方 AttFuse Clean validation

**REFERENCE DISTANCE WITHIN 2pp; NOT A STATISTICAL PASS**

| 指标 | 本次 AP | 官方参考 | 绝对差值 / pp |
|---|---:|---:|---:|
| AP30 | 0.85993605 | 0.8669 | 0.6964 |
| AP50 | 0.82279434 | 0.8258 | 0.3006 |
| AP70 | 0.66625452 | 0.6656 | 0.0655 |

validation帧数：829 / 829；完整验证集：True。
GT总数：6224；预测总数：7034。
官方AP实际累计GT：6223；官方因预测None跳过帧数：43。
GT_total包含所有已读取帧；保留官方跳帧行为，没有补计这些帧的漏检。
实际dataset/model/eval、关键配置、完整源码与checkpoint哈希见protocol.json。
仅执行Clean；本工具没有后续天气分析或训练阶段。
2pp只用于提示复现偏差；接近参考值不等于统计意义PASS。
checkpoint SHA256：`4e92bff533cba4d8e2020f4176bd9f684b5a2337d59f853535f612edd6aa3e94`

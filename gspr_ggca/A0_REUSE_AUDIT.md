# A0复用审计规则与当前证据

指定唯一来源：`/data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4/B0/`。

2026-10-27已使用用户授权的cjm账户直接读取海光实际文件：training_summary为complete、20轮，history有20轮，selected选中epoch19，Clean O AP70=0.6790423426742578。实际选模权重为`B0/epochs/19/checkpoint_7c75ddef20d3042c27f57f07f9f3366f86d9758bcbab5a5e1730aec2bb00c1c6.pth`，实际计算SHA256与文件名及selected记录一致。初始状态实际SHA256=`a4b953d907a9b25f279e19731a5f191b673751ce1f976dafa20b0bfe69fed164`，与00.json一致。

正式三天气summary为complete、full_validation=true、每天气829帧，所选权重身份一致。O AP70：Clean=0.6790423426742578、Fog=0.7552091642488489、Snow=0.39197021920951053；F对应0.678715040773897、0.445432654556161、0.3675350577292191。这些是旧B0实际结果，不是GGCA新模型成绩。

上述只读检查尚未替代服务器preflight对全部逐轮产物、预测缓存、配置、数据和完整初始状态的严格审计。正式复用仍须由该步骤通过后写出baseline_reference.json。路径始终根据真实selected.json确认，不扫描其他目录寻找替代品。

自动核对：B0正式训练合同身份、20轮完整提交和1010更新/轮、逐轮Clean829帧O AP70、选模最优及最早并列规则、selected与训练完成记录、原GSPR/模型源哈希、训练配置/数据划分/抽样/随机种子、原生运行环境、固定GT和O/O*/F协议、实际checkpoint SHA256、完整初始00.pth及其保存标记。共同初始化必须来自原epoch0，不能用已训练最佳权重初始化A1—A3。

已有完整匹配三天气评测时只读复用。评测尚未完成或缺失时，使用同一选模权重在新实验A0/supplemental_evaluation补评测。已有完整结果若权重、协议或GT身份冲突，则停止，不能把冲突当作可忽略的缺失。

任何缺少合格选模权重、训练未完成、缺少可信初始状态或协议无法匹配都会停止正式比较，不重训A0，不找其他权重。旧B0全部文件只读。输出的baseline_reference.json、preflight/A0_REUSE_AUDIT.json及同名Markdown记录实际检查结论；本文件是规则说明，不是服务器通过证明。

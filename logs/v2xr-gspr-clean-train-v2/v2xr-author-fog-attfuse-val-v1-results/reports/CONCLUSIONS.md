# 结论

以下数值仅在服务器完整评测成功后生成。

clean：D−作者 O AP70 +8.118706 pp；固定 GT F 为 +8.107483 pp。在固定 Clean 目标并计入 None 帧后，D 仍领先。
None跳帧差异项 +0.011223 pp；GT筛选差异项 +0.000000 pp。

fog：D−作者 O AP70 +15.722703 pp；固定 GT F 为 +9.243959 pp。在固定 Clean 目标并计入 None 帧后，D 仍领先。
None跳帧差异项 +0.086704 pp；GT筛选差异项 +6.392040 pp。

snow：D−作者 O AP70 +5.500300 pp；固定 GT F 为 +5.104313 pp。在固定 Clean 目标并计入 None 帧后，D 仍领先。
None跳帧差异项 +0.036449 pp；GT筛选差异项 +0.359538 pp。

Fog 的预测框、正确匹配和漏检来自原版匹配器计数，见 MATCHED_COMPARISON.md。
逐帧增益仅在历史原始审计文件通过哈希和 GT 身份检查后输出；缺少时报告明确的缺失来源。

公平范围：同一 validation、同一冻结 GT 和评分协议的固定模型性能。C/D 在 validation 上选模；作者 checkpoint 的训练细节和选模过程未由本次验证。
论文 Table 5 的 split 未明确；相关消融表使用 testing，不能把论文公开数字视作同一 validation。
单次评测不能证明统计显著性、训练机制因果关系或正式 test 泛化；本次无训练、无阈值调整、无 test 访问。

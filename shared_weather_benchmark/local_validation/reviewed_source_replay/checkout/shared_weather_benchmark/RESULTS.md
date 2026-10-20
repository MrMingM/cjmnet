# 第一版交付结果

实现世界固定Fog场、双程路径积分、原版E0控制、point identity/sidecar、固定Clean GT审计、连续观测指标、逐scene比较和阶段恢复。

本地纯数值测试及总入口的实测状态见`local_validation/`。真实OPV2V的P3/P4未运行，当前没有数据集物理统计、观测互补或模型失败模式结论。

默认Layer 1不假定传感器接收阈值，保留点数并改变强度。非均匀散射只有数值核与明确禁用的生成接口。已有论文关系和全部近似分别见PRIOR_ART_AND_SCOPE.md、PHYSICAL_MODEL.md、PHYSICS_ASSUMPTIONS.md。

下一步按README在服务器先检查，再由用户手动启动小样本。是否扩大取决于跨scene可复现的互补差异、匹配残差和传感器依据；单纯生成成功或AP下降不能作为扩展理由。

# v3：局部空间来源融合

独立目录，不修改v2或冻结GSPR。两种模型使用相同空间编码器、完整训练集、8轮预算、分类/回归损失与权重修改惩罚：

- attention：普通可学习来源softmax，作为简单融合对照。
- residual：原AttFuse权重与新权重通过局部门控混合，门控上限0.5，初始约0.009。初始化接近原模型，并非精确恒等或不退化保证。

输入为当前接收并按原proj_first路径处理的来源特征。每来源共享1×1降维和3×3空间卷积，以来源自身、来源平均、原融合表示及原权重预测新权重。全BEV密集处理，不使用高分候选筛选、GT区域、有效peer或天气标签。尺度0/1修改，尺度2保留；分类与回归共享融合。

## 自动流程

1. 检查完整train/validation和四个测试目录、保存解析配置与实际依赖源码快照。
2. 单元测试。
3. 分别训练attention和residual：每帧在线生成模拟天气，clean/weather分别前向、累积梯度后更新一次；不生成v2单动作缓存。
4. 完整模拟天气validation联合输出校准，仅选择使用模型或KEEP。
5. 模拟天气development评价（与校准相同split，不是独立验证）。
6. 自动OPV2V clean test和固定OPV2V-W fog/rain/snow测试，关闭在线天气。
7. 汇总all_results.json。任何阶段失败停止后续阶段，日志在run目录。

训练冻结GSPR、backbone、deblocks和检测头参数，但允许新融合输出经deblocks和检测头反传；不重新训练整个检测器。所有训练帧均参与，单车帧没有来源选择梯度，不当作缺失帧。

checkpoint按混合clean/weather验证损失选择。校准门禁除v2的Clean AP70、lost和newFP限制外，增加各条件AP50下降不超过0.001；平均AP70增益须大于0.0001。只在validation选择，绝不按测试重新选。始终报告原始attention/residual及其*_deployed结果，若KEEP通过也不会掩盖模型原始表现。

## 服务器后台运行

在原opencood环境、原服务器项目目录中：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
export RUN=/data/cjm/datasets/logs/local_fusion_v3_$(date +%Y%m%d_%H%M%S)
nohup bash local_fusion_v3/run_all.sh > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$! RUN=$RUN"
tail -f "${RUN}.launcher.log"
```

支持PY、CONFIG、FRONTEND_ROOT、FRONTEND_CONFIG、FRONTEND_CHECKPOINT环境变量，与v2相同。RUN必须是新目录。此脚本自动执行最终OPV2V-W，无需训练后手动测试。训练期间可查看RUN/train_attention.log或train_residual.log。

## 范围与成本

本版实现推荐主方案及普通softmax公平对照。原v2结果作为历史参照；未实现空间输入+离散peer-query、加宽MLP、分类定位分离或检测端修复组合，不能用本轮对照单独归因输入和动作变化。

现有raw/full来源特征全部接收，不增加通信质量图，但不节省通信；raw_feature_bytes不含网络协议开销。计算计时包含共享编码与上下文、模块、检测头、后处理，排除加载、H2D及指标匹配；不是物理端到端延迟。每种模型都计入共享上下文成本。

局部连续权重仍不能从全部坏来源中创造证据。检测损失也不直接优化NMS或AP。必须同时检查AP30/50/70和恢复/误伤。新训练保留全帧梯度，比v2的缓存MLP训练更昂贵，不能沿用原十分钟训练估计。

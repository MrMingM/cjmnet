# 已生成天气数据的几何检查

目的：8帧 smoke 已完成，但出现 Shapely intersection 警告。原项目的 AP 匹配没有
显式拒绝 NaN IoU，需要复核本次检测框和重叠计算。检查脚本独立于 reference_weather，
保留其源码快照和已有生成身份。只在 Linux 科研服务器执行。

把本包三个文件同步到 `/home/cjm/OpenCOOD-main/cjmnet/`，保留原 reference_weather。
检查直接读取数据 manifest 中的参数、随机种子、精确权重；核对源数据/源码/资源哈希。
使用原项目的 NMS、AP 和 IoU 计算，额外检查模型输出、框坐标、多边形有效性、面积、
每次 IoU 是否为有限数且位于0–1。遇到无效值会停止并保存错误位置。
检查通过时另算 AP 并与原 smoke 对比。GPU浮点差异可能影响临界框，0.01pp 仅为重放
一致性的提示阈值。警告按每帧默认过滤规则记录，同一位置重复警告可能合并计数。

生成阶段的点数保留比例、回波替换/丢点数和点云平均强度也会汇总。
输出点数包含被天气回波替换的点，不能直接称为目标点保留率。
该检查不认证 Snow 的地面强度拟合或物理分布；仍需检查点云统计和后续完整 validate。

在服务器项目根目录后台启动，GPU换为空闲的物理卡号：

```sh
nohup env GPU=0 sh run_reference_weather_audit.sh \
  > /data/cjm/datasets/logs/reference_weather_geometry_audit_v1.log 2>&1 &
echo $!
```

输出位于原 smoke 数据目录的 `geometry_audit_<UTC时间>/audit.json`。
原 evaluation/ 结果保持原样，已下载的资源和生成的 PCD 直接复用。
重点查看 `status`、`geometry_checks_passed`、`replay_matches_within_0_01pp`、
`warning_counts`、`point_statistics`。失败时查看 `current_frame` 和 `traceback`。

本地仅做 AST 语法、文件完整性检查。实际数值验证在服务器完成。

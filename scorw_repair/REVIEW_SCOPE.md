# 本轮只读复核记录

没有全面重搜论文、重装环境或修改 P0–P7。

实际核对了：

- AI_CONTEXT.md 的最新尾部记录；本轮只处理 SCOR-W，未执行其 V2X-R 支线。
- shared_weather_benchmark 的 world_field、ray_integrator、generator、observability、dataset_io、geometry、audit、e0_adapter。
- reference_weather/UPSTREAM.json、vendor/fog/fog_simulation.py、adapters.py、common.py、config.yaml、evaluate.py、assets_lock.json。
- logs/sota/NEXT_STAGE_DECISION.md 与 logs/newdataset 的 P7_DECISION.json、RESULTS.md、P7_SCENE_SELECTION.json。
  两轮最终状态均为 INCONCLUSIVE；P7 建议停止将旧代理版本扩成全量天气数据集。
  本轮属于修复观测退化机制，不能把旧结论改写成成功结果。
- related_code/SeeingThroughFog 的 README、官方 split、校验和、terms_of_use、数据读取函数和示例元标签。
  官方旧校验和对应 SeeingThroughFogCompressed.z01–z18 及最终 ZIP，不能套到当前每传感器链接上。
- related_code/LiDAR_fog_sim 的 theory.py 和 generate_integral_lookup_table.py。
- 新增本地论文 Multi-Sensor_Alignment_2026.pdf 与 Impact_MultiScattered_LiDAR_2024.pdf 的前两页。
  仅以已读取部分说明统计对齐和单次散射限制，不宣称复现论文完整方法。
- AttFuse 的模型、scatter、voxel preprocess/postprocess、推理与 AP 实现，以及原 reference_weather 的冻结权重入口。
  R2 使用直接特征适配器，固定 Clean GT，不使用可能按天气删 GT 的历史 dataset loader。

定向访问官方 STF 下载介绍页返回403。本地没有使用临时签名 URL 发起网络请求；
实际 URL 有效性、大小、分卷以服务器 preflight 为准。
私有清单的文件名检查发现一个 lidar_hdl64_last.zip，以及 weather_station 和标定/标签链接；
没有 labeltool_labels.zip。清单带 Markdown 包装，新解析器兼容其独立 URL 行，过滤无关传感器。
未记录、打印或复制签名 URL 内容到本报告。

新增 official_stf/ 只复制必要的官方纯文本 split、旧校验和及许可证；没有复制数据载荷或私有链接。
inputs/ 只导出旧 P7 开发场景/帧身份，并记录原选择文件哈希，避免要求服务器再同步所有旧结果目录。

官方来源：

- [Seeing Through Fog 官方代码](https://github.com/princeton-computational-imaging/SeeingThroughFog)
- [Hahner 官方 fog simulator](https://github.com/MartinHahner/LiDAR_fog_sim)
- [Hevisov et al. 2024](https://doi.org/10.3390/s24165121)

第三方许可继续适用：STF 仅研究/教学、不可将原始数据再分发；Hahner 代码为 CC BY-NC 4.0。
小结果包仅包含派生统计、身份及报告，不包含原始点云、元数据载荷、图片或数组。

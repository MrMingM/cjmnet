# 已训练AttFuse：单独查看作者官方协议O

无需重新训练。当前epoch20的官方O结果已经存在于D−1D的final_evaluation中：

| 条件 | AP30 | AP50 | AP70 |
|---|---:|---:|---:|
| Clean | 0.87655091 | 0.84707141 | 0.70539766 |
| Fog | 0.56354597 | 0.55077779 | 0.44212603 |
| Snow | 0.46624621 | 0.43523409 | 0.28892218 |

本目录提供官方O单独报告，以及可选的独立重新推理入口。全部实际运行在服务器，不在本地执行评测。
仅新增本目录，既有v2xr_train_attfuse、v2xr_protocol_audit、v2xr_official_repro及其snapshot保持不变。
服务器无需Git；新目录整体按源码SHA256校验，继承的评测包继续使用原来的哈希/契约检查。

## 保留的协议

- 官方仓库：https://github.com/ylwhxht/V2X-R ，固定commit `f3ebbad95561444a53a17358735b0f0dc4ba89f7`。
- IntermediateFusionDatasetV2XR，实际class为opencood.data_utils.datasets.opv2v.intermediate_fusion_dataset.IntermediateFusionDataset。
- PointPillarIntermediate / processed_lidar；fov=true、proj_first=true、COM_RANGE=70、max_cav=7。
- 官方范围、camera0 FOV裁点、框内点数>0的逐CAV GT过滤和ID去重保持不变。
- score>0.20，NMS=0.15，BEV IoU，官方eval_utils_opv2v，VOC2010 AP，global_sort=False。
- **O / OFFICIAL_EXACT**：各天气使用自己的官方GT；prediction=None整帧跳过，其GT不计入AP分母。
- prediction为非None空数组时仍正常计分，不与None混淆。
- 使用已经按Clean锁定的权重，不按天气重新选epoch；仅validate，不读取正式test。

本工具不另写GT过滤、匹配、NMS或AP。rerun调用已验证的v2xr_train_attfuse/main.py evaluate-checkpoint入口，
后者通过冻结D−1C生成器运行原版official inference.main()，并校验O replay统计与原版统计完全相同。
底层仍保存O*/F审计文件；本工具的最终报告只选取O。额外审计不改变输入或预测。
读取已完成结果时也核对stat_O与official_runs/<weather>/official_result_stat.json一致。

## 路径

- 训练根目录：`/data/cjm/datasets/logs/v2xr-clean-attfuse-train-v1`。
- 默认读取的已完成评测：训练根目录下`final_evaluation/`。
- 报告目录：`/data/cjm/datasets/logs/v2xr-attfuse-official-protocol-v1`。
- 新推理缓存：报告目录下`inference_full/`；12帧检查在`inference_smoke12/`。
- TMPDIR：`/data/cjm/datasets/tmp/v2xr_official_selected_eval`。
- 默认权重从训练根目录selected.json自动读取，并核对训练完成记录和实际checkpoint SHA256。
- 可用TRAINING_ROOT、REPORT_ROOT、EVALUATION_ROOT改路径；新推理输出必须位于独立REPORT_ROOT内部。
- 继承评测依赖仍使用旧launcher的环境变量：TRAIN_ROOT/RUNTIME_ROOT/MODEL_DIR/D1A_ROOT/GT_AUDIT_ROOT/D1C_ROOT/VIEW_ROOT等；默认值与成功D−1D相同。

## 推荐：已有结果直接单独导出O

将整个v2xr_official_selected_eval目录同步到服务器项目根目录，运行：

```sh
cd /home/cjm/OpenCOOD-main/cjmnet
nohup env ACTION=export sh v2xr_official_selected_eval/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_O_export_v1.log 2>&1 &
echo $!
```

不需要GPU或重新推理，但核验权重/评测文件哈希可能耗时。确认服务器原final_evaluation完整保留；
只回传到本地的summary/RESULTS不足以执行服务器完整产物校验。

## 需要重新推理时

先做12帧运行检查（不能作为完整结果）：

```sh
nohup env ACTION=rerun GPU=0 LIMIT_FRAMES=12 \
  sh v2xr_official_selected_eval/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_O_smoke_v1.log 2>&1 &
echo $!
```

全829帧，Clean/Fog/Snow：

```sh
nohup env ACTION=rerun GPU=0 LIMIT_FRAMES=0 \
  sh v2xr_official_selected_eval/run_server.sh \
  > /data/cjm/datasets/logs/v2xr_official_O_rerun_v1.log 2>&1 &
echo $!
```

GPU=0替换为可用物理卡号。脚本只设置ROCR_VISIBLE_DEVICES，unset HIP_VISIBLE_DEVICES和CUDA_VISIBLE_DEVICES；内部cuda:0。
第一次使用独立inference_full目录会真正重新推理，不复用原final_evaluation的预测。
若该新目录已运行/中断，以相同参数加RESUME=1、日志改>>恢复；不得同时运行两个任务。
恢复会校验并复用已提交缓存；需要另一轮全新推理时显式指定新的REPORT_ROOT，旧缓存保留。

```sh
tail -n 80 -f /data/cjm/datasets/logs/v2xr_official_O_rerun_v1.log
```

完整报告为REPORT_ROOT/OFFICIAL_O.md和OFFICIAL_O.json；12帧为OFFICIAL_O_SMOKE12.md/.json。
JSON记录来源summary路径/哈希、checkpoint、帧数、官方GT与实际分母、None帧及源码哈希。
导出模式只验证/整理已有结果，不应称作一次新推理；重新推理的AP可能受HIP浮点非确定性影响，不强制与旧值逐位相同。
若原包、环境、数据或模型不兼容，继承入口会报错，不通过改配置/阈值或部分加载权重继续。

本地仅允许运行`python -B v2xr_official_selected_eval/check_static.py`做AST/LF/源码哈希检查。
没有新增数值测试；原有服务器自检及真实数据推理链已在D−1D运行完成，新的推理/真实文件核验仍仅在服务器运行。

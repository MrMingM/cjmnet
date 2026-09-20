# Stage-3C 三项离线分析

只读取已有 Stage-3C JSON，不加载 Torch、不运行模型，不改历史实验代码和清单。
同步整个 qa_stage3c_offline/ 到服务器项目根目录；依赖现有 qa_local_intervention/report.py。

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
/home/cjm/miniconda3/envs/opencood/bin/python -m unittest qa_stage3c_offline.test_analyze -v
RUN_NAME="qa_stage3c_offline_$(date +%Y%m%d_%H%M%S)"
nohup /home/cjm/miniconda3/envs/opencood/bin/python -u -m qa_stage3c_offline.analyze --stage3c-root /data/cjm/datasets/logs/qa_stage3c_local_20260919_195547 --output-dir "/data/cjm/datasets/logs/$RUN_NAME" > "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log" 2>&1 &
echo "PID=$! RUN_NAME=$RUN_NAME"
tail -f "/data/cjm/datasets/logs/${RUN_NAME}_launcher.log"
```

产物：
- 01_safety.md：候选/控制分开，按天气、动作、尺度、强度、区域和安全结果比较；分数/框分歧的P10、中位数、P90。
- 02_complementarity.md：按焦点目标去重，三种动作族互斥组合、两两交集、同peer交集、联合尺度独有事件。
- 03_unresolved.md：未恢复目标逐个排查；阶段转移、关键anchor区域包含关系、最佳IoU、分数余量。
- action_events.jsonl、target_overlap.jsonl、unresolved.jsonl：保留场景/距离、动作和逐框细节。
- analysis_results.json：完成标志、输入SHA和汇总；任何输入变化或不完整结果会报错，不能将半成品视为完成。

分数/框分歧从已有全局分数/几何交叉替换的同anchor日志读出，分别等于对应peer的输出分量。
这些数值在干预前已存在，但anchor、区域和有效peer选择依赖GT；不能据此宣称无GT触发器已验证。
未保存的anchor对比明确为缺失值，不填零。分位数按动作事件统计，同帧多peer不独立。
安全定义保持原TP不丢失且没有新增FP，分数下降单独统计。此脚本不训练选择器、不搜索阈值、不作显著性检验。

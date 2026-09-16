# 固定 contrast 权重的幅度扫描

本轮用户结果：legacy、contrast、contrast_gain 的正常/中性/置换消息检测 AP 均与 protocol 完全相同；contrast 平均调整约 0.000148，contrast_gain 约 0.000073，后者平均增益约 .4996。新结构未显示检测收益。本扫描检查已学出的调整方向是否仅受幅度限制，不重新训练。

同步更新的整个 `gspr_review` 后运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_amplitude_scan.sh \
  --run-dir /data/cjm/datasets/logs/gspr_cf_seed20260909_20260910_102406/contrast
```

从该训练目录的 manifest 加载固定前端路径；从保存的 experiment.yaml 加载相同数据和天气配置。使用旧 reviewer_best.pth，不修改模型参数、训练配置或旧结果；结束时复核输入文件哈希。

倍数为 0、1、10、50、100；0 倍仅运行 normal，其他每个倍数运行 normal/permuted。Clean/Weather 共 18 组。0 倍是不进行点修正的同协议对照；1 倍重现原方法。每组保留实际消息及原始可修改点掩码，按几何格置换消息内容。脚本逐帧检查样本、总字节及准入点数一致。

先计算原始 `max_adjustment*tanh(score)`，乘倍数后限制到原 `[-max_adjustment,+max_adjustment]`，最后限制权重在原 `[reliability_floor,1]`。因此触及边界时，实际调整不再严格为相应倍数，应结合输出中的 mean_mean_abs_weight_change 判断。scale 缺省为 1，保持原训练/评测行为；非 1 倍禁止在训练模式使用，不写入 checkpoint。

新输出位于 `contrast_amplitude_<时间戳>`。每组完成就写 scan_partial.json，全部完成写 scan_comparison.json。将最终文件回传即可；若中断，partial 只代表已完成的组，不能称为完整结果。

判读：真实消息放大后优于 0 倍且优于相同倍数的置换，支持调整方向具有任务价值；两者同时改善只能说明调整有作用，未证明正确对应关系有用；两者恶化说明盲目放大不能解决问题。先与本轮 0/1 倍结果比较，再参考同一运行中已有的 A0B0。该扫描使用反复观察的开发样本，不能据此选择倍数后直接宣称独立测试提升。

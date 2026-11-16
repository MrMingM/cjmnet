# 固定权重下的负向协同

E0 为原始 LR0，E1 为仅含 GSPR 的正式 LR-G，E2 为 GSPR 加 A1 的 F0。三组分别使用物理卡 4、3、5。默认入口核对三组正式权重；使用 `--with-lrg-training` 时，E0/E2 并行诊断，卡 3 复用 LR-G 原训练流程并自动接续负向协同诊断。可用 `LRG_RUN_DIR` 指向原 LR-G 正式产物目录。

主车模式从两种传感器的真实柱体中保留主车，并同步截取位置和主车标记，完整经过单车的原生结构。固定真值在单车和协同模式间共享。阈值敏感性使用官方分数阈值及其 0.75、1.25 倍，每次按原生流程重新去重。目标以官方 0.7 一对一匹配计数。负向目标追踪解码、分数、几何筛选、去重和范围裁剪，阶段标签属于候选层证据。点数分层使用框内主车激光雷达点数。

在服务器的工作目录后台运行：

```sh
out=/data/new_disk/cjm_logs/v2xr-negative-collaboration-v1
nohup env NEG_OUT="$out" sh v2xr_negative_collaboration/run_all.sh > "$out/launch.log" 2>&1 </dev/null &
```

先确认输出目录已存在且可写。每组保存 smoke.log、full.log、逐天气逐目标 JSONL 和汇总结果，最后生成 REPORT.md 和场景留一结果。

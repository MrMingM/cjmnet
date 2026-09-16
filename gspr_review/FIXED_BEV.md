# 固定 BEV 缩放对照（无需训练）

复用已完成的固定消息作用位置实验，从 paired_evaluation.json 定位已训练 BEV checkpoint，逐帧重放现有缓存。不会修改旧权重或缓存。运行：

```bash
cd /home/cjm/OpenCOOD-main/cjmnet
bash gspr_review/run_fixed_bev.sh \
  --location-run /data/cjm/datasets/logs/gspr_cf_seed20260909_20260910_102406/contrast_location_20260910_164020
```

先同步整个更新后的 gspr_review。四个对照为 zero（不校正）、fixed_005（特征乘 1.05）、fixed_010（特征乘 1.10）、learned（已训练校正）。每组都只改原有支持点映射出的三尺度 BEV 位置，未支持位置保持不变。固定组覆盖学习出的标量，不使用网络决定幅度；但仍使用相同证据支持掩码，因此不是完全不使用通信的规则。

Clean/Weather 共 8 组；打印组号与时间，结果存入新时间戳目录 fixed_partial.json / fixed_comparison.json。对照逐帧检查样本编号、字节、准入点数与消息 SHA256；缓存输入指纹不一致时停止。开始/结束核对旧权重、配置和缓存哈希。

两个正系数提前固定为 0.05 与 0.10；这不是穷尽规则搜索，也不做 AP 最优系数选择。若学习版本不优于这两个规则，没有证据表明当前学习带来额外收益；若优于，也只是开发集上的初步证据，不证明稳定泛化。当前只有约 54 帧每种环境，较小 AP 变化应结合漏检与误检统计解读。

模型加载和前向仍执行原头以保持路径、接口一致，但固定组不使用其分数产生最终校正，因此本实验不用于比较去掉网络后的速度。服务器不用额外安装依赖。当前本地 18 项相关 CPU 测试通过，Python/Bash 语法与冻结文件核验通过；真实 AP 待服务器运行。

# 2026-10-25 训练诊断记录修复

旧版在每轮训练开始时建立 module_rows，训练批次更新后添加诊断；进入验证时又执行 module_rows = []，因此 history.json 和 epochs/01/complete.json 中的 repe 始终为空。第一阶段汇总随后报 No real optimization/diagnostic records。训练和推理完成日志不能证明这份诊断记录保存成功。

本版移除验证前的清空操作，保留训练记录。第一阶段仍严格拒绝空记录，并在错误中列出具体模型及记录路径。没有更改网络、损失、优化器、数据、随机种子或验收条件。

旧版已提交的逐批模块统计没有保存下来，不能从权重或汇总损失还原。本修复不改写旧 history.json、阶段完成文件、源码身份或冻结依赖；旧目录不能直接使用新源码续跑。使用新输出目录重新运行预检、结构测试和五组小规模训练/评测，通过后才进入正式训练。

海光使用新 OUTPUT_ROOT，例如 /data/cjm/datasets/logs/gspr-repe-rocm-20261025-v4。CUDA 使用新交付包自身的 run_remote.sh 默认新目录及新 CUDA_INPUT_ROOT。旧 CUDA 输入审计也绑定源码身份，不指向旧输入目录。

部署前应确认没有进程正在读取待替换源码。若有正在运行的任务，保留原项目，在独立项目目录部署完整 CUDA 新包；不要边运行边覆盖源码。海光增量包依赖原项目冻结文件，确认海光旧流水线已退出后，将 gspr_repe 和 v2xr_pointcloud_only 移至旁边备份，再解压更新包。

test_training_records_static.py 只检查代码控制流程，并用临时报告样例检查通过/拒绝逻辑；不导入 Torch，不运行模型或真实数据。修复后的科研验收须在服务器执行。

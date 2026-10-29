# GGCA 几何修订2与保留A1的续跑

本包部署到项目根下gspr_ggca_geo_v2_r2/，原gspr_ggca/完全保留。几何初稿源码gspr_ggca_geo_v2/及其工程验收记录也保留，不覆盖。只用run_all.sh启动；MODE=preflight/smoke/all，默认all、RESUME=1、BACKEND=rocm、GPU=2。A0不训练，A1保持旧源码/合同/目录，从最近完整提交轮次恢复。新版A2/A3重新做小规模验收，然后各自独立训练。

新完整状态验收通过后，优先让A1以旧源码在卡2继续。GGCA_TEST_GPU默认1，仅当该卡空闲时用于新A2/A3小测试；没有额外空闲卡则顺序执行。A1_lane/train.log保存原A1的续跑日志，state.json与launch_*.json记录真实PID、启动时间、原命令和完整恢复点。控制器重启会核对并接管仍运行的同一A1，不会重复训练；新模块测试失败保留A1继续。

```sh
nohup env BACKEND=rocm GPU=2 MODE=all RESUME=1 \
  sh /home/cjm/OpenCOOD-main/cjmnet/gspr_ggca_geo_v2_r2/run_all.sh \
  >> /home/cjm/datasets/logs/gspr_ggca_geometry_v2_r2.launch.log 2>&1 < /dev/null &
```

仅在旧GGCA安全停止工具和migration_plan_v2.json已准备、当前统一入口未运行时恢复。该计划引用只读的初稿migration_plan.json哈希；统一入口等待停止记录验证通过，不会为更新源码向其他任务发信号。

新输出/home/cjm/datasets/logs/gspr-ggca-geometry-v2-20261028-r2。A1继续写原gspr-ggca-v3/A1，A0仍只读引用/data旧B0。新根stages/和run_identity绑定版本；migration计划显式引用初稿根migration/下的停止、原状态和源码快照；self_test/保存本版机制、A1状态恢复和完整模型验收；smoke/A2、smoke/A3为全新真实测试；reports/最终生成版本公平性、结果和案例。初稿等待调度器已主动停止，其身份与记录保留；该初稿没有启动正式A2/A3。

不更改旧合同或哈希。代码改变必须建立明确的新版本和未运行过的新输出，不能覆盖运行中源码。A1原源码已由PINS.json和迁移记录共同固定。METHOD_DESIGN.md为本版公式和PointDSC适配依据，design_history/保留旧设计，MIGRATION_AUDIT.md为衔接说明。实际测试是否完成以服务器文件为准。

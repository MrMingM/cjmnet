# 本地检查记录

日期：2026-10-20。仅检查静态代码及纯Python/NumPy合成输入；没有导入Torch/OpenCOOD，没有读取真实PCD、运行GPU、科研训练或推理。没有实验AP结果。

- 25个Python文件编译/AST解析通过；新增代码pyflakes无告警。
- 18项合成测试通过：当前天气密度差异、空图、半开边界/非有限点、密度截断、XY池化、三车变换方向/组合、固定场景划分、FOV保留原点索引、有效槽位/模糊来源标签剔除、缺失区域掩码参考、按密度打乱、无标签INCONCLUSIVE、预注册ABCD合成正例、场景级bootstrap、未舍入选模/最早并列、原子commit与不完整阶段、恢复身份/更新边界、pp差值/N/A报告。
- settings/protocol JSON与原UECP YAML解析检查通过；原版仅一处u_map.detach的AST约束通过；LF检查通过。
- `sh -n uecp_gspr_v2xr/run_all.sh`通过。Windows沙箱内Git sh无法启动，随后在沙箱外仅执行语法检查成功，没有执行脚本。
- 私有源码接口与依赖由PINS.json固定，新目录由source_snapshot.json固定；只读依赖逐项SHA256核对。旧目录未编辑；当前工作区main，未创建branch或worktree。

复查：

```sh
python -B uecp_gspr_v2xr/check_static.py
python -B -m pyflakes uecp_gspr_v2xr
python -B -m uecp_gspr_v2xr.tests_local
sh -n uecp_gspr_v2xr/run_all.sh
```

尚未执行，须在服务器确认：

1. 已接受8083帧/24088单车每天气的真实内容、32场景/92CAV组合、开发集划分、原生强度解码与冻结监督GSPR权重。
2. 相同实际Torch张量下E0原版UECP输出/检测/方向/单车/密度辅助损失等价；私有模块与原生runtime来源正确。
3. 实际warp坐标、GSPR有效槽位聚合、原版与受控detach、E2/E3及各消融梯度、冻结GSPR、无观测处权重及全空输出。
4. 单HIP卡空闲容量；九个模型的极小真实smoke、完整optimizer/scheduler/RNG重放、预测缓存故障注入恢复与GT一致性。
5. 正式20轮训练、train_dev选模、锁定后829帧三天气原生O/O*/F、可信标签足够时的E1判定、最终科研报告。

上述服务器检查已接入MODE=all；也可按README先分别后台执行preflight/self-test/smoke，然后RESUME=1 MODE=all。程序遇到不兼容/NaN/异常梯度/协议不一致会停止，保留失败与PARTIAL报告；科研负结果仍正常归档。

## 启动依赖范围修复复查

2026-10-20新增4项纯临时文件/静态回归检查，共22项：直接运行依赖清单排除GSPR/UECP文档；文档差异不拦截但运行代码差异仍逐项报告；更新安装保留历史README和原始失败日志且可重复执行；已有request.json或存在未识别代码修改时拒绝更新且不覆盖文件。

新增dependency_scope.py和install_update.py后共27个Python文件。修复只调整依赖范围、错误提示和包更新方式；模型、损失、数据适配、训练/评测及settings均与首次交付逐文件哈希比较。未在本地执行科研实验或服务器真实预检。

另在临时目录展开原始交付包，将GSPR README改为与交付版不同的合成文本，再应用实际修复ZIP：历史README和启动日志保留，348项运行依赖与34项新包文件哈希匹配，重复安装返回already_applied。此项仅验证静态文件更新流程，未导入Torch或执行模型代码。

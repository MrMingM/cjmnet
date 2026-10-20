# 本地检查记录

## 当前交付：2026-10-20 稳定计算 v4

用户选择修复计算并保留 v3 验收。新增 stable_crop、stable_checks、successor、tests_stable；
统一严格确定性设置，增加基础 Torch 双线性裁块与服务器等价/精确梯度检查，
新运行可校验继承旧数据缓存，不导入旧训练/评测状态。具体见 STABLE_V4.md。

既有 57 项合成检查通过，新版合成检查覆盖双线性数值、半像素零补齐、有限差分梯度、
重复窗口、后端不允许 warn-only、v3 规则不变、缓存 payload 与旧文件元数据不变、
新 journal 身份重建、拒绝破损和算法变化、初始化中断/恢复、继承输入路径及先恢复后评测。
没有导入 Torch/OpenCOOD、加载真实 checkpoint 或运行科研实验。
服务器等价性、严格后端支持、容量、四组新版 smoke 和正式实验均待运行。

最终检查：既有 57 项与新增 14 项合成检查均通过，共 71 项；25 个 Python 文件的
AST/LF/配置/源码快照检查及 326 项冻结依赖检查通过。
两个启动脚本 sh -n 通过（本机沙箱中的 Git sh 启动失败后，在沙箱外只做语法解析，无脚本执行）。
旧版备份 zip 的 31 项源码 SHA256 与父版本一致，v3 resume_check 函数 AST 与旧版完全一致。

以下为历史记录。

日期：2026-10-19。范围限于静态及纯 Python/NumPy 合成检查。没有读取真实PCD、导入Torch/OpenCOOD、使用GPU或运行科研训练/推理。没有实验AP结果。

- 16项合成检查通过：不同传感器原点；端点/空域/后方未知；多车冲突与相同x0；重复点不变；最近回波及角度环绕边界；体素边界；ego/仅邻车来源；同步几何变换；进入窗口的外部射线；全未知/空标签；类与来源分母；窗口随机流独立；掩码packbits往返；受控选模与最早并列；原生真实AST经适配后的零GT/空CAV路径（纯NumPy替身，无原生runtime导入）。
- 新增Python源码AST解析、JSON解析、配置约束、原生适配分支匹配、LF与源码SHA256快照校验通过。
- launch_v2xr_e0e3.sh、run_all.sh 均通过 `sh -n`。第一次Git sh在Windows沙箱内创建内核对象失败；随后在沙箱外仅执行语法检查成功，没有执行启动脚本。
- v2xr_train_attfuse、v2xr_protocol_audit、v2xr_official_repro 原source_snapshot逐项核对：缺失0，变化0。326个实际只读依赖源码/配置/文档文件记录在PINS.json。

复查命令：

```sh
python -B v2xr_raygen_e0e3/check_static.py
python -B -m v2xr_raygen_e0e3.tests_local
sh -n launch_v2xr_e0e3.sh
sh -n v2xr_raygen_e0e3/run_all.sh
```

尚待服务器：真实数据完整性/标签类别、原生输出等价、真实辅助梯度、单HIP显卡容量、真实四组smoke、Adam/scheduler/RNG恢复重放、评测中断恢复、完整训练与829帧三天气O/O*/F以及最终报告。自动检查已接入MODE=all。未声称这些服务器检查已经通过。

## 2026-10-20 自检U字段修复

用户服务器日志：7项官方AP合成检查通过，随后server_checks在全未知窗口调用auxiliary_loss时因缺少U失败。修复后的Torch/HIP检查尚未在服务器运行。

- 新增回归检查从smoke.py的checks函数提取并执行真实的纯NumPy样本构造语句，从losses.py读取实际统计字段。修复前单独运行该检查，复现同一个 `KeyError: 'U'`。
- 在测试样本中补齐 `U=~(O|E)` 后运行完整合成检查：17项通过。全未知窗口O/E/ego/peer计数0，U计数8192，E2/E3监督权重0。
- 本地仍未导入Torch/OpenCOOD、读取真实PCD或运行科研训练/推理。
- 修复版AST/LF/JSON/配置/源码快照检查通过；训练、损失、标签、配方、启动器和326项只读依赖的源码未改动。新快照随本目录交付，旧服务器输出契约不作修改。
- 原两个shell启动脚本没有变化，沿用前次已通过的sh -n结果。

本次涉及smoke.py、tests_local.py、README.md、LOCAL_VALIDATION.md及本目录source_snapshot.json。服务器请整目录同步后使用新OUTPUT_ROOT首次启动，详见README中的修复说明。

## 2026-10-20 原失败目录定向恢复（替代前述新目录建议）

为避免重新执行已完成的长时间smoke准备，新增resume_compat.py及U_FIX_COMPAT.json，并接入controller/worker/report。仅接受最初交付版本在server_checks的确切U缺失错误，且科学源码/配置/输入/选定microbatch及GPU不变、四组训练尚未开始。原契约/源码快照/缓存不作改写，实际新源码另存追加来源记录。

纯Python/NumPy检查累计25项通过，其中8项恢复检查使用临时合成文件状态，覆盖：保留所有原文件内容并记录实际新源码；拒绝科学源码变化；拒绝配置变化；拒绝损坏缓存；拒绝损坏显存检查提交；拒绝已开始训练的目录；拒绝其他错误；重复恢复保持同一修复来源并拒绝后续源码漂移。没有构建或加载真实模型，临时.pth仅为用于哈希检查的合成字节文件。

本次新模块和修改模块的AST/LF/JSON/配置/源码快照检查通过，326项只读依赖保持原样。两个shell入口未改动。真实服务器原目录恢复尚待执行。同步最新完整目录后，使用README中的原OUTPUT_ROOT及RESUME=1；不再采用前一小节的新目录建议来处理该特定错误。

## 2026-10-20 本次恢复改用物理GPU 2

用户明确指定本次使用2号卡。旧恢复器在检查原显卡号时退出，尚未创建修复来源记录。新版允许在这个相同的U字段错误边界登记目标显卡；保留已完成的CPU预检和标签缓存，换卡后在新卡上额外检查一次原microbatch的显存容量。

原execution.json和execution_commit.json不改写。目标卡号冻结在新增修复记录中；独立探测输出使用source_repair/gpu_probe/2/，通过后新增source_repair/gpu_capacity.json。runtime从经过核验的新记录选择显卡及硬件名称，training只新增探测路径参数及有效执行记录读取；训练配方、梯度累积、模型/损失/标签算法保持不变。报告同时记录原执行设置和实际新卡设置。2号卡显存不足时立即停止，不自动更改batch。

纯Python/NumPy合成检查累计34项通过，新增9项覆盖：换卡登记不修改原文件且要求新检查；限制目标卡和原microbatch；新卡提交可重复恢复且保留原设置；拒绝损坏探测产物；拒绝其他卡的探测；拒绝静默缩小batch；检测探测提交变化；控制器先探测再进入server_checks且重复恢复复用新探测；容量检查失败时停止后续步骤。控制器测试替换了子进程调用，没有运行Torch、OpenCOOD、真实数据或GPU实验。

本次修改resume_compat/controller/worker/runtime/training/report、测试及说明，并更新交付源码快照。AST/LF/JSON/配置/源码哈希检查通过，326项只读依赖不变。shell入口未改动。2号卡真实显存检查和完整恢复需在服务器执行；本地没有实验结果。

## 2026-10-20 E0恢复检查v2：精确加载与计算后数值差异分开验证

用户已回传原目录E0诊断输出：第一轮起点、显卡、第二轮采样/实际输入摘要、RNG、scheduler完全一致；119个模型浮点张量通过原容差，最大绝对差4.363246262073517e-06。Adam有8个张量共1446个位置超出原逐元素容差；这些张量的相对L2差4.7e-7至6.1e-5。另有4项浮点损失日志被要求完全相等。完整诊断保存在服务器diagnostics/resume_E0_1792466315475575112.json；本地仅读取用户回传的文本，没有加载真实checkpoint。

新增resume_compare.py和RESUME_CHECK_COMPAT.json；抽出训练真实共用的load_resume_state，要求刚加载后的模型/aux/Adam/scheduler/criterion/RNG全部逐项相等。smoke新增独立restore-check阶段，不执行训练更新；E0原两轮训练/验证/第二轮重放直接复用。模型权重沿用原逐元素容差，Adam仅对三个moment字段补充每个张量的L2误差界，浮点日志用原rtol/atol，整数计数/LR/step/RNG/配置仍精确。新规则作为v2明确记录，旧逐元素失败不删除、不改称通过。

原U修复来源不改写；仅在确切已诊断E0边界附加v2来源，固定上一版源码身份、诊断、E0完成记录、产物哈希及文件身份。首次升级重新计算产物哈希，后续worker核对完成记录哈希及产物元数据，恢复检查再次计算实际产物哈希；避免每个worker反复读取大checkpoint。已完成E0验证的故障注入阶段也直接跳过。最终报告附带完整v2规则和来源。原config/protocol/model/loss/data/geometry/cache/runtime/evaluation算法及326项依赖保持原样。

实验包共50项纯Python/NumPy合成检查通过，本轮新增16项，覆盖：浮点日志与整数计数分开；Adam张量L2及模型原容差；拒绝大幅moment漂移；LR/step/RNG精确；加载后微小权重差也失败；NaN/Inf失败；实际加载函数AST用NumPy替身执行并发现未加载优化器；升级保留每个原文件及旧来源；拒绝不匹配诊断/RNG、科学源码变化、大幅优化器差、损坏产物、后续组已开始；旧源码worker拒绝；控制器跳过E0训练/评测/容量并只增加恢复检查；损失次数变化与非有限数拒绝。

AST/LF/JSON/配置/源码快照和326项只读依赖检查通过；两个shell入口未修改。根目录只读诊断工具的8项合成检查此前已通过，工具保持原样。未在本地导入Torch/OpenCOOD、读取真实数据或运行训练/推理。真实精确加载检查及v2恢复验收仍待服务器执行。

## 2026-10-20 E1：RMS绝对预算v3修正

用户回传E1证据：564项状态精确回载成功，起点和数据相同；唯一失败为optimizer/state/17/exp_avg，L2差0.00017879605184775667、张量尺度1.0707758045136373、相对差0.0001669780462857662。此时E0已通过；E0/E1训练/验证/重放和精确加载均已完成。

v3保持原rtol/atol，整体Adam比较由L2差<=atol+rtol*尺度改为RMS差<=atol+rtol*RMS尺度，整块绝对预算相应乘sqrt(N)。旧v2规则、policy identity和既有失败不改写；新policy明确revision=3，报告记录元素数、RMS、L2与允许界限。精确加载证明继续使用未改动的v2加载函数，避免重复建立模型。模型原逐元素容差、LR/step/RNG/计数和NaN/Inf检查不变。

新增RESUME_RMS_COMPAT.json固定上一版30项交付文件，在确切E1边界检查产物哈希、旧来源、同一GPU及诊断/精确加载证明后附加第三段来源记录。E0/E1结果及旧报告不改写，后续重试只执行E1比较。新报告单独用v3路径，最终summary保留规则变更与来源。源码hash、训练/模型/损失/标签算法、数据、runtime和326项依赖保持冻结要求。

实验包57项合成检查通过，本轮新增7项：以E1尺度构造的有限NumPy张量验证按元素预算且模型检查仍严格；重复相同系数的结论不变；拒绝大幅漂移/NaN/Inf及LR变化；保留每个旧文件及全部旧来源；拒绝训练源码或精确加载证明变化；拒绝损坏checkpoint或已开始E2；控制器复用E0/E1训练与精确加载且只重试E1检查。所有.pth均为合成字节，未在本地加载Torch/OpenCOOD或运行科研实验。

AST/LF/JSON/配置/交付源码快照和只读依赖检查通过；shell入口未改动。服务器实际v3比较和后续实验仍待执行。

# 2026-09-19_11-41-48

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

> 最新实机反馈（2026-09-21）：用户报告行进右转时再次剧烈抖动，主机电源线脱落且无日志。暂停实机复现；下方历史导出说明不构成当前上机建议。

## 评估批次

- [native-assess-20260921-001](<evaluations/native-assess-20260921-001/report.md>)：4 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/native-assess-20260921-001/manifest.json>)。
- [turning-after-straight-20260921-001](<evaluations/turning-after-straight-20260921-001/report.md>)：6 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/turning-after-straight-20260921-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [events](<events>)
- [evidence/checkpoint_selection](<evidence/checkpoint_selection>)
- [evidence/export](<evidence/export>)
- [evidence/source](<evidence/source>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 本次对比分析

- [双髋惩罚对比指标、方法和证据哈希](evidence/analysis/hip-penalty-comparison-20260921-001.json)
- [髋目标、机身抖动与站立漂移对比图](evidence/analysis/hip-penalty-comparison-20260921-001.png)
- [训练 TensorBoard 全标量有限性检查与末两段 5000 迭代统计](evidence/training/tensorboard-review-20260921-001.json)
- [训练来源与当前评估源码逐文件核对](evidence/training/preflight-20260921-001.json)
- [训练日志摘要](evidence/training/summary-20260921-001.json)；[收敛判定限制](evidence/training/assessment-20260921-001.json)

四个固定 seed 42 场景完成，无跌倒或重置。新增双髋原始动作 L1 惩罚 -0.5 压低目标幅度，但 40–120 秒站立窗口的漂移和机身抖动退步；不能据此认定已解决实机启动振荡。下一步建议在既有实机观测上回放新旧策略，比较启动输出及髋反馈敏感度，再决定奖励调整；本次未修改参数或启动训练。

历史归档限制：未经实物验证，不代表 hardware-ready；当前已出现新的实机事故，应暂停实机复现。

## 新旧 ROA 同实机观测回放

- [启动髋输出、反馈敏感度、方法与输入校验](evidence/analysis/roa-hip-penalty-same-observation-20260921/metrics.json)
- [对比曲线](evidence/analysis/roa-hip-penalty-same-observation-20260921/comparison.png)
- [逐帧输出与 2×2 雅可比](evidence/analysis/roa-hip-penalty-same-observation-20260921/replay_and_sensitivity.csv)

固定使用旧策略实机事故的 29 帧观测与记录动作历史，CPU 离线推理；不是新版的实机闭环轨迹。新版早期髋目标有所降低，但历史填满后的 200–300 ms 位置敏感度接近旧版，整段目标仍被放大至约 3.41 rad；不能判定实机振荡已解决。

## 导出归档 20260921

- 用户选定本轮 model_49999.pt，已完成 JIT/ONNX 导出及8个时序样本（含reset）一致性校验。
- [选择记录](evidence/checkpoint_selection/selection-deploy-20260921-001.json)、[导出校验](evidence/export/deploy-20260921-001/receipt.json)、[归档记录](evidence/export/deploy-20260921-001/archive-receipt.json)。
- 归档目录：`/home/server/liufengrong/policy_storage/LW/wheel_loco/2026-09-19-11-41-48`，按训练开始时间命名。包含 policy.pt、policy.onnx、策略说明.txt、archive_manifest.json；说明保留站立漂移退步及实机回放限制。
- 历史归档限制：未经实物验证，不代表 hardware-ready；当前已出现新的实机事故，应暂停实机复现。

## 行进后转弯诊断 20260921

- [六项固定场景及原始证据](evaluations/turning-after-straight-20260921-001/report.md)
- [分阶段指标、配对校验与局限](evidence/analysis/turning-after-straight-20260921-001/metrics.json)
- [延迟、噪声与左右转对比图](evidence/analysis/turning-after-straight-20260921-001/phase-comparison.png)
- [实机事故反馈记录](2026-09-21T09-38-00-070415-00-00__physical-right-turn-oscillation-20260921-001.json)

seed 42、单环境、每项 90 秒：站立 10 秒 → 0.5 m/s 直行 30 秒 → 同速度叠加 ±0.6 rad/s 转弯 30 秒 → 零指令 20 秒。用户记不清事故指令；这是固定假设，不是事故重建。三组分别为固定执行器延迟 15 ms/有噪声、30 ms/有噪声、15 ms/关闭策略观测噪声。左右转前 40 秒全部匹配，六项完整且无重置。

42–70 秒窗口：30 ms 相对 15 ms 的机身横滚/俯仰角速度 RMS 增加约 10%–12%；关闭噪声使其下降约 15%，髋目标逐步变化 RMS 下降约 37%。实际机身偏航角速度右转 −0.259 至 −0.269、左转 +0.309 至 +0.318 rad/s，均低于 ±0.6 指令。未复现突然发散，估计误差未显示转弯后持续增长；不能排除实机端到端时延、机械/接触差异或状态分布外反馈。未据此修改训练参数，也未启动新训练。下一步优先只读核对实机部署时序和观测/动作契约，再提有限仿真方案。

## 传感器观测年龄诊断 20260921

- [两项有限仿真及不可变批次证据](evaluations/turning-sensor-age-20260921-001/report.md)
- [分阶段指标、基线配对及时间契约核验](evidence/analysis/turning-sensor-age-20260921-001/metrics.json)
- [一秒分组对比曲线](evidence/analysis/turning-sensor-age-20260921-001/phase-comparison.png)
- [授权范围与运行前核验](evidence/analysis/turning-sensor-age-20260921-001/preflight.json)

按用户批准，仅增加角速度、投影重力、关节位置/速度统一延迟一个策略周期（20 ms）；观测噪声/scale之后延迟，指令和前次动作保留原有时序，再逐周期推进10帧历史。保留15 ms执行器延迟、有观测噪声、seed42、单环境、原90秒命令时间表。20 ms是受控假设，不是事故实测。旧六项不重跑，直接引用其中左右delay15-noise基线。

两项均4500步完整，无done或timeout，逐帧来源和延迟校验通过；首帧用当前帧填充、年龄0，其后4499帧年龄20 ms。左右前40秒的状态、动作和观测逐元素完全一致；执行动作与student诊断动作误差0，teacher速度标签与同帧pre-action真值误差0。基线重算与旧分析最大差1.78e-15。

42–70秒：机身横滚/俯仰角速度RMS，右转0.2868→0.3424 rad/s（+19.4%），左转0.3071→0.3565（+16.1%）；实际髋角速度RMS分别+14.4%/+12.6%，但髋目标逐步变化RMS分别−3.7%/−5.6%。右转机身角速度模长采样峰值0.785→1.492 rad/s；实际平均yaw右转−0.261、左转+0.311 rad/s，仍低于±0.6指令。72–90秒停后净位移右转0.373→0.526 m，左转0.326→0.305 m。

此受控试验支持观测陈旧会加重机身和关节运动，作用并非右转独有；未复现持续发散/重置，不能确定本次实机故障根因。统一固定延迟不覆盖不同传感器的异步、抖动、丢帧和滤波；20 ms采样不排除物理子步瞬时饱和。继续暂停实机复现，不调整训练参数。未提交推送，部署模型与外部仓库未写入。

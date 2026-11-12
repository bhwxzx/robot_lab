# 评估批次 wheel-to-leg-no-leg-spacing-20261112-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-11_19-54-20`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`078d1aac4c78970d392517395761a2ddf195265f5d29086fff7bd4f7b48d2906`；runner：`OnPolicyRunner`。
- [场景来源 scenario-8455fdf533a8c2a44ddae4665cc16e1920d6a2c6975a4f376d471e63bb3f0e1d.json](<../../provenance/scenario-8455fdf533a8c2a44ddae4665cc16e1920d6a2c6975a4f376d471e63bb3f0e1d.json>)
- [场景来源 scenario-916142d2b142d585aaec20a3fcad1450b4727a5207be55e5aa13fd418b34818c.json](<../../provenance/scenario-916142d2b142d585aaec20a3fcad1450b4727a5207be55e5aa13fd418b34818c.json>)
- [训练上下文 context-813f98c69247044b1e25ee41e0fa0d4c8d9a26ccee47987b75cbea006478b156.json](<../../provenance/context-813f98c69247044b1e25ee41e0fa0d4c8d9a26ccee47987b75cbea006478b156.json>)
- [训练有效配置 config-0d1eb6af4e96f8535d40767e5e9b94ee17c530e6a2324e3d9d11b9eafd54ba69.json](<../../provenance/config-0d1eb6af4e96f8535d40767e5e9b94ee17c530e6a2324e3d9d11b9eafd54ba69.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 结论：本机2026-11-11_19-54-20/model_49999.pt通过本次标称及随机化有界动作测试，但存在随机轮距收窄、双髋偏角增大及落地高度/俯仰退步；不能判定整体优于旧版，也不能认定实机抖动已修复。
- 身份：checkpoint SHA-256=078d1aac4c78970d392517395761a2ddf195265f5d29086fff7bd4f7b48d2906，actor590→10、critic146；训练HEAD=bca8144并含已保存本地修改，评估HEAD=ff2ade8。已核对当前flat_env_cfg.py字节与训练期间保留版本一致，其余相关源码自训练HEAD无改动。准确train.py argv来自本会话此前对PID639016的现场观测，非由YAML反推。
- 训练有效配置：4096环境、seed42、50000迭代，数据增强和镜像损失开启（系数0.1）；leg_symmetry=0、feet_distance_penalize=0、joint_symmetry=-0.5、hip_pos_zero_l2=-1、wheel_contact_continuous=0.5。不是youngHit的2026-11-11_19-51-59模型。
- 训练达到配置终点：TensorBoard共50000个记录、最后step49999，checkpoint iter49999；末1000步平均reward7.0044、episode length76.989、motion_finished0.99518、镜像损失Loss/symmetry0.00016083。Loss/symmetry作用于策略镜像输出，并非只作用于髋部；训练随机帧重置，不能将motion_finished比例等同完整动作成功率。无批准收敛标准，不作收敛结论。
- 测试预算：两个Native场景，分别标称/随机化，seed42，每场1环境、1000控制步20s；每场1000个20ms控制样本、4000个5ms物理样本。每场5个170帧完整动作成功、0失败，另1截断尾段；尾段不算成功/失败，极值统计包含尾段。两场完整遥测和bundle校验通过，无需重跑旧版。
- 标称关闭观测噪声、指定质量/材料/COM/PD等随机化及推扰、重置偏差和执行器延迟；随机化保留训练设置。两场均固定由参考frame0启动。具体覆盖已绑定contract.json和scenario证据。初始质量、COM、材料及PD参数与复用的Nov10/Nov09同场景记录逐项一致。
- 主要基线为本机2026-11-10_13-58-00/model_49999.pt；另复用11月9日记录供历史参照。训练环境数8192→4096、7项奖励权重及轮约束时间窗同时变化，观察差异不能归因于单独关闭leg_symmetry或脚间距项；跨不同奖励配置的总reward不可直接用于优劣排序。
- 位置跟踪：全部控制样本anchor position RMSE标称6.408→2.958cm（下降53.8%），随机18.638→17.142cm（下降8.0%）；姿态RMSE标称0.06789→0.03352rad，随机0.14482→0.12601rad。随机body position RMSE0.03266→0.03497m，说明各项指标并非都改善。
- 落地后[2.0,3.4]s两轮5ms速度合并RMS的周期均值：标称1.361→0.158rad/s（下降88.4%），随机1.253→0.202rad/s（下降83.9%）。但完整周期全程轮速RMS标称2.274→2.672、随机3.410→3.930rad/s，全部物理样本轮峰速标称7.542→11.398、随机13.239→15.316rad/s。落地后静止改善不等于全程轮速更低。
- 轮距采用双轮轮心世界XY欧氏距离，原奖励窗口参考frame0..62（0–1.25s）。标称均值51.646cm，均在原[49.6,53.6]cm区间；随机均值49.601→47.050cm、最小43.585→40.638cm、原区间外样本比例43.49%→98.49%。原区间仅作为旧配置参照，不是新增验收阈值。
- 双髋最大实际偏角（完整周期，右/左）：标称2.714/2.210→5.512/5.506°，随机8.276/5.126→11.239/11.061°。起跳前0..54帧目标角峰峰值的周期均值，标称10.391/10.786→13.361/13.323°，随机17.605/20.095→13.877/17.076°；随机有所下降，标称有所上升。目标变化由原始action×0.25计算，常量default offset不影响峰峰值，但不是执行器延迟后的实际目标。
- 镜像动作输出正则开启不等于轮距受约束；本次随机双髋偏角与轮距收窄同时存在。没有实机闭环或匹配状态敏感度新验证，不能由仿真周期成功、较低髋力矩或部分目标变化减小推断实机抖动已经消失。
- 起跳/落地：标称主腾空段开始1.320→1.345s、结束1.570→1.610s，持续250→265ms；随机主腾空均值251→249ms。判据为监测法向力正Fz总和<10N且持续至少15ms、窗口1.0–2.3s中的最长段，描述性阈值而非精确实物离地。起跳窗口基座峰值世界vz标称1.779→2.024m/s，随机均值1.822→1.869m/s；这是base COM速度，不是整机COM速度。
- 高度：跳跃窗口[1.10,1.70]s标称顶点0.6973→0.7011m，随机均值0.6996→0.6966m。落地窗口[1.45,1.90]s高度RMSE标称2.162→1.421cm，随机2.829→4.062cm（变差）；随机五周期落地最低高度范围0.4178–0.5216m。落地俯仰绝对峰值标称7.412→11.220°、随机10.571→13.372°。
- 前后偏差：标称首次足触地X误差-6.72→-4.33cm、终点-5.00→-3.92cm。随机五周期首次足触地X误差[-26.67,-17.56,+17.18,+16.78,-0.03]cm，终点[-30.19,-18.21,+11.95,+24.44,-1.63]cm；正号表示前于参考，负号表示后于参考。使用同一记录时刻的参考/实际误差，不能用有符号均值的正负抵消来判断改善。
- 接触载荷：足部最大法向力模长标称1081.22→884.04N，随机2051.31→1457.04N；轮部峰值标称543.31→748.05N、随机927.18→1078.68N。监测大腿/小腿对地法向力为0。仿真5ms法向力不包含摩擦力，也不是实物传感器测力。
- 关节约束：最新标称无超过配置速度参考值的物理样本；随机仅右足1个5ms样本达到10.246rad/s（参考10），相较旧版右足12.710rad/s及左足10.096rad/s尖峰改善。全部关节未见达到99%配置力矩上限的样本；随机左小腿峰矩112.620N·m，占120N·m限值93.85%，高于旧版105.767N·m。配置速度参考值不等于已经验证的硬件安全边界。
- 统计口径：关节速度/力矩峰值使用全部5ms物理样本，可能大于result.json中20ms控制采样指标；joint-loads.csv同时保留完整周期/全部观察两个范围及极值发生时刻、周期编号。
- 保全：2850个已有证据、模型、配置和源码文件哈希不变，HEAD及3个stash不变，未修改项目代码、启停训练、导出、安装、删除或提交。仅新增本run评估证据。历史查询为query_context_incomplete，direct_parameter_change_supported=false；本报告为显式同场景描述性比较。
- 局限：单seed，每场5完整周期；标称重复轨迹相同，不能视为5次独立随机试验。未录制视频、未进行实物验证。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 全部观察范围的10关节峰值（速度rad/s，力矩N·m；包含截断尾段）：

| 关节 | 标称峰速 | 标称峰矩 | 随机峰速 | 随机峰矩 |
| --- | ---: | ---: | ---: | ---: |
| right_hip_joint | 1.246 | 38.128 | 3.191 | 31.233 |
| left_hip_joint | 1.267 | 38.838 | 2.545 | 41.162 |
| right_thigh_joint | 13.420 | 26.733 | 15.102 | 47.560 |
| left_thigh_joint | 13.689 | 34.845 | 13.992 | 38.491 |
| right_shank_joint | 14.803 | 87.975 | 17.629 | 90.497 |
| left_shank_joint | 15.348 | 79.879 | 16.761 | 112.620 |
| right_foot_joint | 8.421 | 16.583 | 10.246 | 20.639 |
| left_foot_joint | 8.086 | 17.032 | 9.355 | 21.251 |
| right_wheel_joint | 11.306 | 3.360 | 13.212 | 6.289 |
| left_wheel_joint | 11.398 | 2.701 | 15.316 | 6.855 |
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis-method.json](<derived/analysis-method.json>)
- [analysis.json](<derived/analysis.json>)
- [cases-ready.json](<derived/cases-ready.json>)
- [cycle-metrics.csv](<derived/cycle-metrics.csv>)
- [height-contact-audit.json](<derived/height-contact-audit.json>)
- [height-contact-method.json](<derived/height-contact-method.json>)
- [hip-spacing-comparison.png](<derived/hip-spacing-comparison.png>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [phase-comparison.png](<derived/phase-comparison.png>)
- [stance-load-method.json](<derived/stance-load-method.json>)
- [stance-load-metrics.json](<derived/stance-load-metrics.json>)
- [batch-driver-method-20261112-001.json](<../../evidence/training/batch-driver-method-20261112-001.json>)
- [effective-config-diff-20261112-001.json](<../../evidence/training/effective-config-diff-20261112-001.json>)
- [evaluation-request-20261112-001.json](<../../evidence/training/evaluation-request-20261112-001.json>)
- [history-query-20261112-001.json](<../../evidence/training/history-query-20261112-001.json>)
- [protection-after-20261112-001.json](<../../evidence/training/protection-after-20261112-001.json>)
- [protection-before-20261112-001.json](<../../evidence/training/protection-before-20261112-001.json>)
- [tensorboard-windows-20261112-001.json](<../../evidence/training/tensorboard-windows-20261112-001.json>)
- [training-origin-20261112-001.json](<../../evidence/training/training-origin-20261112-001.json>)
- [report-input.json](<derived/report-input.json>)

## 建议与待授权事项

- 保留旧版基线；本版具有标称跟踪与落地后轮速优势，但随机轮距、髋偏角、落地姿态仍需处理，不建议仅凭5/5成功就判定可替代旧版。
- 下一步优先对新旧模型做相同多seed有界测试，核查轮距收窄、髋偏角和落地高度/俯仰；若调整奖励，应先提交单项改动方案并获得批准，避免同时改变多项后无法归因。
- 实机抖动是否缓解需要匹配状态敏感度分析及新的受监督实机闭环证据。本次不自动训练、改参数、导出或部署。

# 评估批次 wheel-to-leg-local-20261026-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-25_11-55-09`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8512cbffcace36bc1512682d6e9c1b5cd9eaf158b4dd394f924733bb85ad607c`；runner：`OnPolicyRunner`。
- [场景来源 scenario-66fde5ce5364b1718821e9744f0e5b4ae1443dd02b74f62ffb6ac46b0120e2d8.json](<../../provenance/scenario-66fde5ce5364b1718821e9744f0e5b4ae1443dd02b74f62ffb6ac46b0120e2d8.json>)
- [场景来源 scenario-d34b846d2c34c1c61eb98bcf75d956bd881dee11c2a9c06efabeafdb449b28b3.json](<../../provenance/scenario-d34b846d2c34c1c61eb98bcf75d956bd881dee11c2a9c06efabeafdb449b28b3.json>)
- [训练上下文 context-e3ab53e9cb1fc4f9bd11b805863ce793569977ec5c6978110de661388bc36b42.json](<../../provenance/context-e3ab53e9cb1fc4f9bd11b805863ce793569977ec5c6978110de661388bc36b42.json>)
- [训练有效配置 config-5c021199b9f352c34bfa9f640935d99d73d1fc1d84da691cd519c3ffe7c19fcf.json](<../../provenance/config-5c021199b9f352c34bfa9f640935d99d73d1fc1d84da691cd519c3ffe7c19fcf.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 本机 younghit 场次 2026-10-25_11-55-09/model_49999.pt，SHA-256 8512cbffcace36bc1512682d6e9c1b5cd9eaf158b4dd394f924733bb85ad607c。用户确认 bash scripts/start_beyondmimic.sh --type wheel，无额外参数或覆盖。
- 训练及评估 HEAD 08f2208a491b06131615b43fad228bd9e83f69aa，训练时 Git 工作区干净。保存 YAML 确认位置/连杆位置/连杆线速度权重 2/2/3，undesired_contacts=-1.0；action_rate=-0.2、smoothness=-0.075；anchor_pos 高度阈值 0.15 m，左右 foot_link 的 ee_body_pos 高度阈值 0.25 m；推扰关闭。
- 参考仍为 10/23 NPZ，SHA-256 8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03。PPO / OnPolicyRunner / ActorCritic、seed42、10 帧分项展开历史、590 维策略输入、10 维动作，腿/轮/足执行延迟 0..3 物理步。运行时 reset 历史一致性断言通过。
- 两组 Native、单环境、seed42，各 1000 控制步 / 4000 物理步、无视频。遥测 complete，必需信号无缺失；独立接触视图交叉校验误差 0 N。固定组禁用噪声和重置/质量/材质/gain 随机化，保留延迟；随机组保留训练随机化并随机起始相位。
- play-ground-a01：从 frame0 完成 5 个完整周期，0 个非 motion_finished 终止，另有1个未完成尾段；完整周期大腿触地 0/5。全1000步观察到的大腿/小腿对地法向力均为0，不能扩展为 base/hip 也无接触。
- 固定5周期：1.10–1.36s 起跳向上峰速 1.283–1.365 m/s、均值1.326（参考1.967）；1.30s 向上速度均值1.325（参考1.827）；1.20–1.30s 速度RMSE0.381 m/s，1.20–1.70s 高度RMSE3.01cm。与上一轮相比，峰速1.168→1.326，高度RMSE5.91→3.01cm。
- 固定5周期：八个记录连杆全部低于1N的最长连续区间为235–250ms、均值241ms，参考计划飞行300ms；10N阈值结果相同。失载开始于参考1.345–1.355s，仍晚于计划1.30s。时长262→241ms的缩短不等于运动退步，应结合落地与跟踪误差。
- 固定准备阶段0.02–1.30s 双轮同时≥1N比例均值99.45%（上一轮79.77%）；下蹲最低基座高度均值0.528m（参考0.490，上一轮0.539）。准备俯仰峰值均值8.87→6.67°，髋镜像残差峰值均值13.03→10.42°，但横滚峰值均值3.59→6.30°，姿态未全面改善。
- 固定5周期飞行最高基座高度均值0.707m（参考0.714）；落地1.60–2.00s最低基座高度0.437–0.473m、均值0.457，上一轮均值0.379。大腿/小腿对地接触时长从每周期均值46ms降为0，正法向冲量均值35.79→0 N·s。
- 固定全程记录Fz合力峰值1.567kN（上一轮5.658kN）；没有驱动力矩达到配置限幅，足关节未超过10rad/s名义上限。左右小腿峰值91.50/106.99Nm，左右足峰速6.27/9.78rad/s。驱动统计不包含碰撞被动关节反力矩。
- training-random-phase-a01：12个已完成随机相位片段，0个非 motion_finished 终止，另有1个未完成尾段；没有 frame0 完整周期。4/12已完成片段大腿触地，尾段另有1段触地；不能当作12次完整跳跃的失败率。
- 随机组仅2个已完成片段覆盖1.10–1.80s完整观察窗口：起跳峰速1.273–1.440m/s、均值1.357；高度RMSE均值4.65cm；八连杆失载270–280ms。它们的准备阶段起始部分未观测，相关全准备窗口指标为null。
- 上述2个随机片段的大腿/小腿接触时长均值65→27.5ms，正法向冲量均值54.16→24.67 N·s，但接触并未消失；落地最低基座高度均值0.395→0.376m，仍有明显下沉。
- 随机全程单大腿/小腿对地峰值4.429kN（left_thigh_link，参考1.640s），记录Fz合力峰值4.833kN（上一轮7.362kN）；左小腿120Nm限幅17个样本、累计85ms，左足27Nm限幅5个样本、累计25ms，不能理解成一次连续持续时长。左足峰速10.11rad/s，1样本超过10rad/s名义上限。
- 随机2个覆盖片段的一阶动作差分平方和均值0.577→1.075，二阶0.250→0.456；固定5周期分别0.444→0.489、0.1605→0.1685。差分按每片段内部动作计算并排除reset跨界，片段初始化影响仍存在。
- ee_body_pos只检查左右foot_link的Z误差：固定组右足最大7.68cm（参考1.30s、尾段）、左足5.92cm（1.46s）；随机组右足8.50cm（1.56s）、左足16.78cm（1.58s）。均未超过25cm阈值，两组均未触发该终止。
- 同机同seed、相同场景字典与时长对照上一轮2026-10-24_13-38-14；有效配置除日志目录外改变4个奖励权重，agent及其他训练参数相同。能描述组合配置的结果差异，不能归因于其中一项权重；也不能把固定组与随机组差异单独归因于域随机化，因为起始相位同时改变。
- 训练console确认至iteration49999，无非有限指标。最后两个各1000轮窗口：mean_reward 14.2318→14.1957；最新motion_finished均值99.642%，主采样bin归一化索引0.25、对应4bins中的索引1，概率约52.7%。权重改变后总奖励不宜直接与上一轮比较。
- 接触力为5ms物理步平均法向力；四个大腿/小腿信号通过net减robot自接触区分对地力，但未覆盖base/hips。八连杆失载不等于全身腾空证明。源参考规划README记录超时与严格节点可行性验收未通过，该限制不能单独解释本策略表现。
- 检查点、保存YAML、既有导出、参考轨迹、配置源码及Isaac Sim持久用户设置的字节和mtime均未改变。评估显式 --/app/settings/persistent=false。本轮未修改代码、训练、重新导出、部署、安装删除或提交推送。
- 本批次单seed、单环境、无视频；固定完整周期表现明显改善，随机初相位与训练随机化叠加下仍有落地接触和驱动限幅。缺少批准的收敛判据，收敛结论indeterminate，不构成实物部署成功证明。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [jump-contact-play-ground-a01.png](<derived/jump-contact-play-ground-a01.png>)
- [jump-contact-training-random-phase-a01.png](<derived/jump-contact-training-random-phase-a01.png>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [protection-after-20261026-001.json](<../../evidence/training/protection-after-20261026-001.json>)
- [training-origin-20261026-001.json](<../../evidence/training/training-origin-20261026-001.json>)
- [training-tail-20261026-001.json](<../../evidence/training/training-tail-20261026-001.json>)
- [contact-duration-and-landing.json](<derived/contact-duration-and-landing.json>)
- [manifest.json](<../../../2026-10-24_13-38-14/evaluations/wheel-to-leg-local-20261025-001/manifest.json>)

## 建议与待授权事项

- 保留本轮作为固定起始回放的候选基线；下一步优先验证固定起始相位且保留训练随机化的有限测试，以分离域随机化和中途相位初始化影响。
- 重点检查随机场景左腿落地、躯干横滚和恢复阶段力矩限幅；不要仅凭固定组无大腿触地宣称整体成功。本轮不自动调整权重或启动训练。

# 评估批次 torque-margin-history10-20261008-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-07_15-57-49`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| legacy-randomized-no-push-a01 | completed | legacy-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/legacy-randomized-no-push-a01/result.json>) / [日志](<raw/legacy-randomized-no-push-a01/console.log>) |
| current-randomized-no-push-a01 | completed | current-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/current-randomized-no-push-a01/result.json>) / [日志](<raw/current-randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`d1ae8cb7ca4496442f47db7b2e6ce9cf28520347dc1f746960ce3929a54be393`；runner：`OnPolicyRunner`。
- [场景来源 scenario-10ffca0f375f9324f4c342df9d0f817ff04da79f3575791fb09497e3fe8acaee.json](<../../provenance/scenario-10ffca0f375f9324f4c342df9d0f817ff04da79f3575791fb09497e3fe8acaee.json>)
- [场景来源 scenario-3b2c1b8671a4531b33f8fba66fa0c40b60f6d0fdd05079460f264459404c4c5b.json](<../../provenance/scenario-3b2c1b8671a4531b33f8fba66fa0c40b60f6d0fdd05079460f264459404c4c5b.json>)
- [场景来源 scenario-4004754fd3eefaee482f3b7a5b25d4d9c474567eefc9499fef3815ad012aacb3.json](<../../provenance/scenario-4004754fd3eefaee482f3b7a5b25d4d9c474567eefc9499fef3815ad012aacb3.json>)
- [场景来源 scenario-45d22e0162f66f14312d5f7e96294efac21ea8861c875f727a8dc1208264bdce.json](<../../provenance/scenario-45d22e0162f66f14312d5f7e96294efac21ea8861c875f727a8dc1208264bdce.json>)
- [场景来源 scenario-78692595d5d5a367c1fba518fd07cc14f8a24b6076bd8ed8e0d93b6d24d9466e.json](<../../provenance/scenario-78692595d5d5a367c1fba518fd07cc14f8a24b6076bd8ed8e0d93b6d24d9466e.json>)
- [训练上下文 context-f8a058ca641feaede7fa721e4c9aea0d4ec65fde9d4f54e641b2cb392afc18cf.json](<../../provenance/context-f8a058ca641feaede7fa721e4c9aea0d4ec65fde9d4f54e641b2cb392afc18cf.json>)
- [训练有效配置 config-6d9f4d5d424450f302b365a529f1d8ce5e210db1b254a0460565f9d004fceba7.json](<../../provenance/config-6d9f4d5d424450f302b365a529f1d8ce5e210db1b254a0460565f9d004fceba7.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `legacy-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `legacy-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_others": null}`。
- `current-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `current-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.push_robot": null}`。

## 观察与限制

- 获批准的固定五组预算：seed42、env1、1000控制步、600秒超时，无视频；与基线逐组匹配。
- 新增足90%和小腿95%力矩裕度；实际、请求力矩及20/5ms采样差异联合审计。
- 保全核验：356个既有证据与43个源码/参考文件。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [evaluation-preflight-20261008-001.json](<../../evidence/training/evaluation-preflight-20261008-001.json>)
- [manifest.json](<../../../2026-10-07_00-51-40/evaluations/torque-power-history10-20261007-001/manifest.json>)
- [analysis-audit.json](<plots/analysis-audit.json>)
- [case_metrics_comparison.csv](<plots/case_metrics_comparison.csv>)
- [contact_and_joint_peak_audit.csv](<plots/contact_and_joint_peak_audit.csv>)
- [cycle_metrics_comparison.csv](<plots/cycle_metrics_comparison.csv>)
- [effort_motion_comparison.png](<plots/effort_motion_comparison.png>)
- [effort_motion_comparison.svg](<plots/effort_motion_comparison.svg>)
- [joint_effort_comparison.csv](<plots/joint_effort_comparison.csv>)
- [phase_metrics_comparison.csv](<plots/phase_metrics_comparison.csv>)
- [randomized_phase_comparison.png](<plots/randomized_phase_comparison.png>)
- [randomized_phase_comparison.svg](<plots/randomized_phase_comparison.svg>)
- [reward_sampling_comparison.csv](<plots/reward_sampling_comparison.csv>)
- [torque_margin_comparison.png](<plots/torque_margin_comparison.png>)
- [torque_margin_comparison.svg](<plots/torque_margin_comparison.svg>)

## 建议与待授权事项

- 根据力矩RMS/峰值、未裁剪请求、短时超速、接触冲击及姿态共同确定下一步；修改或新增测试前提交具体方案并获批准。
- 单seed有限评估与训练完成不能证明收敛；现有证据未作硬件就绪认定。

## 新旧策略的详细比较

新策略：2026-10-07_15-57-49/model_49999.pt；基线：2026-10-07_00-51-40/model_49999.pt。基线五组证据通过完整校验并直接复用，未重跑。五种场景的覆盖、seed42、环境数1、控制步1000、参考运动、延迟及随机化开关逐组匹配。有效配置除日志目录外，只新增足与小腿力矩裕度两项；agent.yaml 完全相同。独立训练的随机性仍需考虑，不能分别归因于某一个新增项。

足关节阈值24.3Nm、权重-0.05；小腿阈值114Nm、权重-0.025。硬力矩限幅仍为27/120Nm。关节出力采用全部5ms样本，包括末尾删失周期；姿态、功率、错轮和落地指标采用完整成功周期。下表为“基线 → 新策略”。

| 条件 | 完整成功/失败/删失 | 双足τ RMS/峰值 Nm | 双小腿τ RMS/峰值 Nm | 足速度峰值 rad/s | 足超速最长 ms | 全10关节绝对机械功率均值 W |
| --- | --- | --- | --- | --- | --- | --- |
| Play | 5/0/1 | 2.22 → 2.73 / 15.76 → 20.44 | 23.66 → 23.67 / 115.44 → 116.83 | 7.31 → 10.00 | 0 → 0 | 112.89 → 119.29 |
| Zero delay | 5/0/1 | 1.94 → 2.67 / 13.78 → 20.42 | 23.30 → 23.34 / 115.53 → 114.85 | 4.47 → 7.18 | 0 → 0 | 109.47 → 116.23 |
| 15ms delay | 5/0/1 | 2.38 → 2.74 / 15.54 → 20.04 | 24.19 → 24.17 / 114.68 → 117.15 | 5.77 → 8.42 | 0 → 0 | 119.07 → 127.66 |
| Legacy DR | 5/0/1 | 5.13 → 4.93 / 27.00 → 27.00 | 28.60 → 27.93 / 120.00 → 120.00 | 15.66 → 10.26 | 10 → 5 | 148.20 → 152.95 |
| Current DR | 5/0/1 | 5.60 → 5.54 / 27.00 → 27.00 | 25.85 → 27.15 / 116.57 → 120.00 | 13.30 → 10.35 | 5 → 10 | 135.85 → 155.86 |

| 条件 | 足请求τ峰值 Nm | 小腿请求τ峰值 Nm | 足/小腿超过软阈值时长 joint ms | 足/小腿请求裁剪时长 joint ms |
| --- | --- | --- | --- | --- |
| Play | 15.8 → 20.4 | 115.4 → 116.8 | 0 → 0 / 25 → 35 | 0 → 0 / 0 → 0 |
| Zero delay | 13.8 → 20.4 | 115.5 → 114.8 | 0 → 0 / 60 → 60 | 0 → 0 / 0 → 0 |
| 15ms delay | 15.5 → 20.0 | 114.7 → 117.2 | 0 → 0 / 30 → 30 | 0 → 0 / 0 → 0 |
| Legacy DR | 42.6 → 30.6 | 140.1 → 137.9 | 340 → 65 / 310 → 225 | 225 → 40 / 220 → 140 |
| Current DR | 34.7 → 37.0 | 116.6 → 128.1 | 225 → 260 / 10 → 75 | 155 → 140 / 0 → 30 |

| 条件 | 空中附近后仰峰值 deg (60–79帧) | 双轮前后差峰值 cm (80–166帧) | 落地双轮合力峰值 kN (75–99帧) | 双轮支撑比例 % (80–166帧) | 起跳峰值高度 m |
| --- | --- | --- | --- | --- | --- |
| Play | 10.764 → 13.077 | 3.343 → 1.908 | 2.730 → 4.590 | 100.000 → 100.000 | 0.748–0.762 |
| Zero delay | 7.726 → 10.390 | 2.460 → 0.181 | 1.963 → 1.889 | 100.000 → 100.000 | 0.746–0.746 |
| 15ms delay | 11.581 → 13.861 | 3.610 → 2.201 | 2.327 → 2.740 | 100.000 → 100.000 | 0.768–0.769 |
| Legacy DR | 10.135 → 14.987 | 3.526 → 3.599 | 2.177 → 2.494 | 99.828 → 99.885 | 0.717–0.771 |
| Current DR | 13.983 → 15.230 | 6.782 → 3.529 | 3.062 → 3.112 | 99.483 → 99.598 | 0.739–0.805 |

当前随机化的关键结果：
- foot：实际峰值 27.000→27.000Nm；未裁剪请求峰值 34.740→37.022Nm；软阈值以上累计 225→260joint ms；速度峰值 13.296→10.348rad/s。
- shank：实际峰值 116.567→120.000Nm；未裁剪请求峰值 116.567→128.103Nm；软阈值以上累计 10→75joint ms；速度峰值 16.075→17.854rad/s。
- 全关节绝对机械功率 135.846→155.857W；落地双轮合力峰值 3.062→3.112kN；这两项作为动作轻柔程度的共同反指标。

## 峰值时刻与计量范围

参考帧阶段：准备0–54、蹬伸55–64、空中65–74、落地缓冲75–99、轮态80–166、静态尾段157–166。落地和轮态有重叠；60–79帧的“空中附近”包含蹬伸末尾和早期落地，不应全解释为腾空。实际触地时刻与左右轮时间差见逐周期CSV。
- play-equivalent-a01：foot/torque峰值在episode 3、第64帧、1.285s，right_foot_joint；实际τ=-20.444Nm、请求τ=-20.444Nm、速度=1.545rad/s。
- play-equivalent-a01：foot/velocity峰值在episode 0、第65帧、1.315s，right_foot_joint；实际τ=-1.171Nm、请求τ=-1.171Nm、速度=-10.000rad/s。
- play-equivalent-a01：foot/requested_torque峰值在episode 3、第64帧、1.285s，right_foot_joint；实际τ=-20.444Nm、请求τ=-20.444Nm、速度=1.545rad/s。
- play-equivalent-a01：shank/torque峰值在episode 5、第61帧、1.240s，right_shank_joint；实际τ=-116.833Nm、请求τ=-116.833Nm、速度=-7.374rad/s。
- play-equivalent-a01：shank/velocity峰值在episode 5、第70帧、1.410s，right_shank_joint；实际τ=21.771Nm、请求τ=21.771Nm、速度=18.026rad/s。
- play-equivalent-a01：shank/requested_torque峰值在episode 5、第61帧、1.240s，right_shank_joint；实际τ=-116.833Nm、请求τ=-116.833Nm、速度=-7.374rad/s。
- zero-delay-a01：foot/torque峰值在episode 0、第64帧、1.285s，right_foot_joint；实际τ=-20.422Nm、请求τ=-20.422Nm、速度=-0.092rad/s。
- zero-delay-a01：foot/velocity峰值在episode 0、第65帧、1.310s，right_foot_joint；实际τ=-2.094Nm、请求τ=-2.094Nm、速度=-7.180rad/s。
- zero-delay-a01：foot/requested_torque峰值在episode 0、第64帧、1.285s，right_foot_joint；实际τ=-20.422Nm、请求τ=-20.422Nm、速度=-0.092rad/s。
- zero-delay-a01：shank/torque峰值在episode 0、第61帧、1.225s，right_shank_joint；实际τ=-114.848Nm、请求τ=-114.848Nm、速度=-6.866rad/s。
- zero-delay-a01：shank/velocity峰值在episode 0、第70帧、1.410s，right_shank_joint；实际τ=1.299Nm、请求τ=1.299Nm、速度=15.325rad/s。
- zero-delay-a01：shank/requested_torque峰值在episode 0、第61帧、1.225s，right_shank_joint；实际τ=-114.848Nm、请求τ=-114.848Nm、速度=-6.866rad/s。
- fixed-15ms-a01：foot/torque峰值在episode 1、第64帧、1.300s，right_foot_joint；实际τ=-20.037Nm、请求τ=-20.037Nm、速度=-1.756rad/s。
- fixed-15ms-a01：foot/velocity峰值在episode 1、第66帧、1.325s，right_foot_joint；实际τ=-0.076Nm、请求τ=-0.076Nm、速度=-8.420rad/s。
- fixed-15ms-a01：foot/requested_torque峰值在episode 1、第64帧、1.300s，right_foot_joint；实际τ=-20.037Nm、请求τ=-20.037Nm、速度=-1.756rad/s。
- fixed-15ms-a01：shank/torque峰值在episode 1、第61帧、1.240s，right_shank_joint；实际τ=-117.152Nm、请求τ=-117.152Nm、速度=-7.348rad/s。
- fixed-15ms-a01：shank/velocity峰值在episode 1、第70帧、1.410s，right_shank_joint；实际τ=21.673Nm、请求τ=21.673Nm、速度=18.004rad/s。
- fixed-15ms-a01：shank/requested_torque峰值在episode 1、第61帧、1.240s，right_shank_joint；实际τ=-117.152Nm、请求τ=-117.152Nm、速度=-7.348rad/s。
- legacy-randomized-no-push-a01：foot/torque峰值在episode 3、第5帧、0.120s，right_foot_joint；实际τ=-27.000Nm、请求τ=-29.589Nm、速度=1.583rad/s。
- legacy-randomized-no-push-a01：foot/velocity峰值在episode 5、第15帧、0.305s，left_foot_joint；实际τ=0.437Nm、请求τ=0.437Nm、速度=-10.260rad/s。
- legacy-randomized-no-push-a01：foot/requested_torque峰值在episode 3、第6帧、0.135s，right_foot_joint；实际τ=-27.000Nm、请求τ=-30.595Nm、速度=1.778rad/s。
- legacy-randomized-no-push-a01：shank/torque峰值在episode 1、第61帧、1.235s，right_shank_joint；实际τ=-120.000Nm、请求τ=-125.985Nm、速度=-7.191rad/s。
- legacy-randomized-no-push-a01：shank/velocity峰值在episode 2、第70帧、1.405s，right_shank_joint；实际τ=26.005Nm、请求τ=26.005Nm、速度=19.188rad/s。
- legacy-randomized-no-push-a01：shank/requested_torque峰值在episode 2、第61帧、1.240s，right_shank_joint；实际τ=-120.000Nm、请求τ=-137.864Nm、速度=-6.849rad/s。
- current-randomized-no-push-a01：foot/torque峰值在episode 0、第15帧、0.315s，left_foot_joint；实际τ=27.000Nm、请求τ=32.149Nm、速度=-5.321rad/s。
- current-randomized-no-push-a01：foot/velocity峰值在episode 5、第10帧、0.215s，left_foot_joint；实际τ=3.765Nm、请求τ=3.765Nm、速度=10.348rad/s。
- current-randomized-no-push-a01：foot/requested_torque峰值在episode 0、第20帧、0.420s，left_foot_joint；实际τ=27.000Nm、请求τ=37.022Nm、速度=-0.923rad/s。
- current-randomized-no-push-a01：shank/torque峰值在episode 0、第60帧、1.220s，right_shank_joint；实际τ=-120.000Nm、请求τ=-123.990Nm、速度=-6.648rad/s。
- current-randomized-no-push-a01：shank/velocity峰值在episode 0、第70帧、1.410s，right_shank_joint；实际τ=23.932Nm、请求τ=23.932Nm、速度=17.854rad/s。
- current-randomized-no-push-a01：shank/requested_torque峰值在episode 3、第58帧、1.165s，right_shank_joint；实际τ=-120.000Nm、请求τ=-128.103Nm、速度=-3.648rad/s。

## 训练记录与解释限制

末1000轮按训练日志汇总（日志只有四位小数，微小裕度代价可能显示为零）：
- wandb/run-20261007_005152-m07zsr06/files/output.log：mean reward=6.74311，身体位置误差=2.1335cm，身体姿态误差=0.11145rad，motion_finished均值=0.997656。
- wandb/run-20261007_155800-f6nvnu5z/files/output.log：mean reward=6.38280，身体位置误差=2.2326cm，身体姿态误差=0.11327rad，motion_finished均值=0.998220。

训练正常完成50000轮、4915200000步。未提供获批准的收敛判据，收敛结论为indeterminate；训练随机起始相位的motion_finished不是从第0帧完成整套动作的成功率。仅seed42与单环境的有限回合不能建立普遍成功率。

力矩是PD执行器驱动力矩，不能代表被动碰撞载荷。功率Σ|τ·qdot|是绝对机械功率，不等于电池消耗。实际力矩限幅与速度上限分别解释；请求力矩受裁剪不等于真实输出超过硬限幅。新奖励读取20ms控制步末帧，5ms统计用于审计短瞬态；采样差异见reward_sampling_comparison.csv。相位峰值、支持间断、轮态几何和动作差分应共同检查，避免只压低RMS而增加落地冲击。

## 图表与完整数据

- [analysis-audit.json](<plots/analysis-audit.json>)
- [case_metrics_comparison.csv](<plots/case_metrics_comparison.csv>)
- [contact_and_joint_peak_audit.csv](<plots/contact_and_joint_peak_audit.csv>)
- [cycle_metrics_comparison.csv](<plots/cycle_metrics_comparison.csv>)
- [effort_motion_comparison.png](<plots/effort_motion_comparison.png>)
- [effort_motion_comparison.svg](<plots/effort_motion_comparison.svg>)
- [joint_effort_comparison.csv](<plots/joint_effort_comparison.csv>)
- [phase_metrics_comparison.csv](<plots/phase_metrics_comparison.csv>)
- [randomized_phase_comparison.png](<plots/randomized_phase_comparison.png>)
- [randomized_phase_comparison.svg](<plots/randomized_phase_comparison.svg>)
- [reward_sampling_comparison.csv](<plots/reward_sampling_comparison.csv>)
- [torque_margin_comparison.png](<plots/torque_margin_comparison.png>)
- [torque_margin_comparison.svg](<plots/torque_margin_comparison.svg>)

仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

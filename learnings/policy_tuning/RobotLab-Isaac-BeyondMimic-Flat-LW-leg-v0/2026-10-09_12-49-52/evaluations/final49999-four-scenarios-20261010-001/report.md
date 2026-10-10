# 评估批次 final49999-four-scenarios-20261010-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-09_12-49-52`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| current-randomized-no-push-a01 | completed | current-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/current-randomized-no-push-a01/result.json>) / [日志](<raw/current-randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8909d6a6c2db883fc3dd37843053fdb887f90b92ead76cff036f8341b5cdd2b1`；runner：`OnPolicyRunner`。
- [场景来源 scenario-0d3bdfb2c02169497536a450875e435837fa02fd62582efd059904ce84fb17c9.json](<../../provenance/scenario-0d3bdfb2c02169497536a450875e435837fa02fd62582efd059904ce84fb17c9.json>)
- [场景来源 scenario-a0c5dfecaad274796358a2b23a853ba8a48ab82d1abb60c6750e1f59da41e9ff.json](<../../provenance/scenario-a0c5dfecaad274796358a2b23a853ba8a48ab82d1abb60c6750e1f59da41e9ff.json>)
- [场景来源 scenario-b3d2310bb30bdf4472fee367d88652c1a06193cb2e6beee2c1c2a9dae7fd43ed.json](<../../provenance/scenario-b3d2310bb30bdf4472fee367d88652c1a06193cb2e6beee2c1c2a9dae7fd43ed.json>)
- [场景来源 scenario-df14ddb4e694d429a099c362e2fd2e488ece9175326234e31f6ee18c39040fd7.json](<../../provenance/scenario-df14ddb4e694d429a099c362e2fd2e488ece9175326234e31f6ee18c39040fd7.json>)
- [训练上下文 context-719efc869d9200b0c248799a18fe8215e297c0727602acf3cc3cb4f908885cb5.json](<../../provenance/context-719efc869d9200b0c248799a18fe8215e297c0727602acf3cc3cb4f908885cb5.json>)
- [训练有效配置 config-ed83bdbc225d9af2a6db3b732873a090767a80ec6db51bff78a6a5895739bfad.json](<../../provenance/config-ed83bdbc225d9af2a6db3b732873a090767a80ec6db51bff78a6a5895739bfad.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-08/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.65, "rewards.motion_anchor_roll_horizontal": null, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.65, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.65, "rewards.same_foot_x_position.params.start_time_s": 1.65, "rewards.wheel_contact_continuous.params.start_time_s": 1.65}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-08/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.65, "rewards.motion_anchor_roll_horizontal": null, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.65, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.65, "rewards.same_foot_x_position.params.start_time_s": 1.65, "rewards.wheel_contact_continuous.params.start_time_s": 1.65, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-08/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.65, "rewards.motion_anchor_roll_horizontal": null, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.65, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.65, "rewards.same_foot_x_position.params.start_time_s": 1.65, "rewards.wheel_contact_continuous.params.start_time_s": 1.65, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `current-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `current-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-08/leg_to_wheel_transform_50hz.npz", "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.push_robot": null, "rewards.leg_symmetry.params.start_time_s": 1.65, "rewards.motion_anchor_roll_horizontal": null, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.65, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.65, "rewards.same_foot_x_position.params.start_time_s": 1.65, "rewards.wheel_contact_continuous.params.start_time_s": 1.65}`。

## 观察与限制

- 已批准四场各1env/1000控制步/seed42，无视频，共4000控制步、16000个5ms物理样本；仅4次启动、无重试。旧10/6、10/7、10/8评估只读复用。本批次 completed 表示证据完成；策略成功由完整周期终止与覆盖单独验证。
- 目标训练原源码0eb1b882加保存的timing diff；本次评估源码2b2bb9e。仅运行时恢复10/8原参考（170帧50Hz、参考腾空1.30–1.60s）、五个1.65s奖励边界和无新增roll奖励。保留当前训练、当前10/10参考及其他刚体质量随机化排除基座配置。
- 完整周期必须从frame0覆盖0..169全部170控制样本/680物理样本，motion_finished且无任何失败/timeout。下列主表和阶段表只使用成功完整周期；failed partial与censored tail保存在机器分析中，整场1000步指标另列。每场单seed，多次同起点周期不是多个独立seed。
- 阶段按每个样本的物理结束时间映射参考时钟：physics=frame×.02+(substep+1)×.005，control=frame×.02+.02；prep[0,1)、push[1,1.3)、reference flight[1.3,1.6)、landing[1.6,1.8)、wheel[1.8,3.4]。末端3.4s纳入wheel。Roll仅20ms采样；与此前按command frame归组的quick表边界口径不同，本批次两策略均用此统一口径重新读取分析，旧证据保持不变。
- 10/9成功完整周期主指标；周期列为成功/失败/截尾。接触力峰值为同一时刻双轮world-Z法向力之和。

| 场景 | 周期 | anchor位置RMSE m | anchor姿态RMSE rad | |roll|max ° | roll RMS ° | 足PD峰值 Nm | 胫PD峰值 Nm | 双轮法向Z和峰值 N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| play-equivalent | 5/0/1 | 0.042 | 0.045 | 3.418 | 1.135 | 10.461 | 98.358 | 2468.7 |
| zero-delay | 5/0/1 | 0.045 | 0.039 | 3.172 | 1.061 | 10.028 | 97.953 | 2461.3 |
| fixed-15ms | 5/0/1 | 0.036 | 0.055 | 3.591 | 1.549 | 10.050 | 98.783 | 3148.5 |
| current-randomized-no-push | 5/0/1 | 0.143 | 0.118 | 7.995 | 1.992 | 27.000 | 118.427 | 2684.5 |
- 同场景历史对比（10/8 → 10/9），均为成功完整周期。奖励窗口1.60/1.65s不同，不以reward跨策略排序；训练配置完整语义差异为log_dir及五个奖励边界，agent无差异。独立训练与策略权重变化仍不支持单项因果归因。

| 场景 | 位置RMSE m | 轮速RMSE rad/s | |roll|max ° | roll RMS ° | 足PD峰值 Nm | 双轮法向Z和峰值 N |
| --- | --- | --- | --- | --- | --- | --- |
| play-equivalent | 0.038 → 0.042 | 3.180 → 3.055 | 4.162 → 3.418 | 1.248 → 1.135 | 18.658 → 10.461 | 4533.2 → 2468.7 |
| zero-delay | 0.038 → 0.045 | 3.322 → 3.137 | 3.561 → 3.172 | 1.262 → 1.061 | 17.695 → 10.028 | 2484.6 → 2461.3 |
| fixed-15ms | 0.041 → 0.036 | 2.854 → 2.952 | 3.984 → 3.591 | 1.134 → 1.549 | 17.251 → 10.050 | 4531.0 → 3148.5 |
| current-randomized-no-push | 0.150 → 0.143 | 3.673 → 3.233 | 7.883 → 7.995 | 2.042 → 1.992 | 27.000 → 27.000 | 3030.7 → 2684.5 |
- 分阶段同场景反指标（10/8 → 10/9）：聚合峰值只在各阶段样本内取，接触阶段与参考阶段可能不同。阶段跟踪、computed/applied足力矩、足超限fraction、wheel接触率均见phase-analysis。

| 场景 | 参考阶段 | 足PD Nm | 胫PD Nm | roll峰 ° | 双轮法向Z和 N | 位置RMSE m |
| --- | --- | --- | --- | --- | --- | --- |
| play-equivalent | preparation | 4.616 → 5.986 | 29.054 → 32.939 | 1.198 → 1.895 | 0.0 → 0.0 | 0.021 → 0.012 |
| play-equivalent | push_off | 17.986 → 8.784 | 99.835 → 98.358 | 1.692 → 2.516 | 0.0 → 0.0 | 0.066 → 0.047 |
| play-equivalent | reference_flight | 18.658 → 10.461 | 82.222 → 86.951 | 4.162 → 3.418 | 4533.2 → 2468.7 | 0.071 → 0.068 |
| play-equivalent | landing | 2.241 → 3.889 | 36.809 → 38.102 | 2.594 → 2.822 | 673.5 → 723.2 | 0.045 → 0.047 |
| play-equivalent | wheel | 0.602 → 0.824 | 23.198 → 17.481 | 1.471 → 2.176 | 454.2 → 469.0 | 0.029 → 0.045 |
| zero-delay | preparation | 4.252 → 5.543 | 26.976 → 32.045 | 1.187 → 1.829 | 0.0 → 0.0 | 0.019 → 0.010 |
| zero-delay | push_off | 17.695 → 8.858 | 99.897 → 97.953 | 1.815 → 2.617 | 0.0 → 0.0 | 0.066 → 0.049 |
| zero-delay | reference_flight | 17.325 → 10.028 | 72.524 → 75.302 | 3.561 → 3.172 | 2484.6 → 2461.3 | 0.073 → 0.070 |
| zero-delay | landing | 1.163 → 3.172 | 30.043 → 34.218 | 2.073 → 0.782 | 592.5 → 754.8 | 0.051 → 0.058 |
| zero-delay | wheel | 0.505 → 0.840 | 17.175 → 13.315 | 1.312 → 0.777 | 441.4 → 414.9 | 0.026 → 0.048 |
| fixed-15ms | preparation | 4.370 → 5.933 | 32.080 → 33.002 | 1.265 → 1.802 | 0.0 → 0.0 | 0.022 → 0.013 |
| fixed-15ms | push_off | 13.380 → 5.538 | 98.468 → 98.783 | 1.749 → 2.176 | 0.0 → 0.0 | 0.065 → 0.044 |
| fixed-15ms | reference_flight | 17.251 → 10.050 | 87.053 → 92.279 | 3.984 → 3.591 | 4531.0 → 3148.5 | 0.069 → 0.066 |
| fixed-15ms | landing | 1.184 → 2.137 | 26.216 → 31.546 | 0.203 → 3.379 | 482.5 → 896.3 | 0.038 → 0.028 |
| fixed-15ms | wheel | 0.563 → 0.812 | 24.451 → 17.998 | 1.151 → 3.346 | 342.6 → 439.2 | 0.038 → 0.037 |
| current-randomized-no-push | preparation | 27.000 → 27.000 | 95.667 → 100.725 | 4.687 → 6.160 | 0.0 → 0.0 | 0.128 → 0.115 |
| current-randomized-no-push | push_off | 20.460 → 13.593 | 108.334 → 118.427 | 3.682 → 5.442 | 0.0 → 0.0 | 0.159 → 0.144 |
| current-randomized-no-push | reference_flight | 18.231 → 13.282 | 92.423 → 94.934 | 7.883 → 7.995 | 3030.7 → 1967.5 | 0.142 → 0.127 |
| current-randomized-no-push | landing | 4.867 → 2.764 | 35.026 → 54.620 | 4.464 → 5.771 | 2250.3 → 2684.5 | 0.163 → 0.151 |
| current-randomized-no-push | wheel | 2.142 → 1.661 | 22.638 → 19.270 | 2.675 → 2.523 | 744.7 → 630.7 | 0.160 → 0.159 |
- 实际腾空/落地逐完整周期（1N阈值）：在frame50..100内取四个足/轮法向力norm均低于阈值的最长连续段，duration=样本数×5ms。阈值穿越精度5ms，第一air样本只界定穿越区间，未触及窗口边缘才视为完整air段。左右轮触地时刻在该air段起点之后分别寻找；峰值窗口是实际首次接触后的100ms。

| 场景 | 周期 | 首air s | 腾空 s | 左轮触地 s | 右轮触地 s | 右减左 s | 触地100ms双轮峰 N | 峰时 s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| play-equivalent | 0 | 1.325 | 0.255 | 1.580 | 1.595 | 0.015 | 2379.8 | 1.595 |
| play-equivalent | 1 | 1.325 | 0.260 | 1.585 | 1.595 | 0.010 | 2448.6 | 1.595 |
| play-equivalent | 2 | 1.335 | 0.250 | 1.585 | 1.590 | 0.005 | 1992.9 | 1.595 |
| play-equivalent | 3 | 1.330 | 0.260 | 1.590 | 1.595 | 0.005 | 2422.4 | 1.595 |
| play-equivalent | 4 | 1.330 | 0.255 | 1.585 | 1.595 | 0.010 | 2468.7 | 1.595 |
| zero-delay | 0 | 1.325 | 0.260 | 1.585 | 1.595 | 0.010 | 2461.3 | 1.595 |
| zero-delay | 1 | 1.325 | 0.260 | 1.585 | 1.595 | 0.010 | 2461.3 | 1.595 |
| zero-delay | 2 | 1.325 | 0.260 | 1.585 | 1.595 | 0.010 | 2461.3 | 1.595 |
| zero-delay | 3 | 1.325 | 0.260 | 1.585 | 1.595 | 0.010 | 2461.3 | 1.595 |
| zero-delay | 4 | 1.325 | 0.260 | 1.585 | 1.595 | 0.010 | 2461.3 | 1.595 |
| fixed-15ms | 0 | 1.335 | 0.255 | 1.590 | 1.590 | 0.000 | 3148.5 | 1.590 |
| fixed-15ms | 1 | 1.335 | 0.255 | 1.590 | 1.590 | 0.000 | 2960.5 | 1.590 |
| fixed-15ms | 2 | 1.335 | 0.255 | 1.590 | 1.590 | 0.000 | 2960.5 | 1.590 |
| fixed-15ms | 3 | 1.335 | 0.255 | 1.590 | 1.590 | 0.000 | 2960.5 | 1.590 |
| fixed-15ms | 4 | 1.335 | 0.255 | 1.590 | 1.590 | 0.000 | 2960.5 | 1.590 |
| current-randomized-no-push | 0 | 1.325 | 0.260 | 1.605 | 1.585 | -0.020 | 1337.1 | 1.605 |
| current-randomized-no-push | 1 | 1.325 | 0.255 | 1.580 | 1.585 | 0.005 | 1967.5 | 1.585 |
| current-randomized-no-push | 2 | 1.330 | 0.260 | 1.605 | 1.590 | -0.015 | 2315.7 | 1.605 |
| current-randomized-no-push | 3 | 1.330 | 0.275 | 1.615 | 1.605 | -0.010 | 2684.5 | 1.615 |
| current-randomized-no-push | 4 | 1.330 | 0.265 | 1.605 | 1.595 | -0.010 | 2370.6 | 1.605 |
- 腾空历史范围比较（10/8 → 10/9）；计划腾空0.300s，当前首air与首次接触列均为10/9。

| 场景 | 腾空时长 s | 首air范围 s | 首次接触范围 s | 右减左触地差 s |
| --- | --- | --- | --- | --- |
| play-equivalent | 0.245–0.255 → 0.250–0.260 | 1.325–1.335 | 1.580–1.590 | 0.000–0.005 → 0.005–0.015 |
| zero-delay | 0.255–0.255 → 0.260–0.260 | 1.325–1.325 | 1.585–1.585 | 0.005–0.005 → 0.010–0.010 |
| fixed-15ms | 0.245–0.245 → 0.255–0.255 | 1.335–1.335 | 1.590–1.590 | 0.000–0.000 → 0.000–0.000 |
| current-randomized-no-push | 0.235–0.250 → 0.255–0.275 | 1.325–1.330 | 1.580–1.605 | -0.025–0.005 → -0.020–0.005 |
- 1N/10N阈值敏感性：[{"scenario": "play-equivalent", "episode_id": 2, "duration_1n_s": 0.25, "duration_10n_s": 0.255, "first_contact_1n_s": 1.585, "first_contact_10n_s": 1.585, "window_censored": false}, {"scenario": "play-equivalent", "episode_id": 3, "duration_1n_s": 0.26, "duration_10n_s": 0.265, "first_contact_1n_s": 1.59, "first_contact_10n_s": 1.59, "window_censored": false}] 两个阈值并非硬件接触判据。
- play等效：整体roll峰与足PD峰、双轮法向力和峰较10/8低，但位置RMSE较高、准备及wheel阶段roll峰更高；零延迟：整体roll与足PD峰较低，位置RMSE较高，左右触地差由5ms增至10ms。固定15ms：位置和anchor姿态误差、足PD峰及双轮力和峰较低，但roll RMS更高、参考landing/wheel阶段roll峰更高，轮速RMSE略高；这些阶段反指标阻止把低整体峰解释为全程更稳。
- PD限值计数（tolerance1e-5 Nm；foot为两关节，分母=3400物理样本×2；all applied分母=3400×10）及参考wheel段双轮支撑率。computed超限与applied受限分别解释；PD驱动力矩不等于结构/被动冲击载荷。

| 场景 | 策略 | 足computed>27Nm计数 | 足computed峰 Nm | 全部applied超限计数 | 双轮接触比例≥10N |
| --- | --- | --- | --- | --- | --- |
| play-equivalent | 10/8 | 0 | 18.658 | 0 | 1.0000 |
| play-equivalent | 10/9 | 0 | 10.461 | 0 | 1.0000 |
| zero-delay | 10/8 | 0 | 17.695 | 0 | 1.0000 |
| zero-delay | 10/9 | 0 | 10.028 | 0 | 1.0000 |
| fixed-15ms | 10/8 | 0 | 17.251 | 0 | 1.0000 |
| fixed-15ms | 10/9 | 0 | 10.050 | 0 | 1.0000 |
| current-randomized-no-push | 10/8 | 4 | 27.872 | 0 | 1.0000 |
| current-randomized-no-push | 10/9 | 23 | 39.945 | 0 | 1.0000 |
- 法向接触力是5ms物理步内平均的PhysX仿真normal force，不含摩擦；不是实物真实冲击峰值。双轮Z和峰在同一物理样本计算，不能把左右独立峰相加。逐周期触地速度（轮轴中心world-Z）、100ms法向冲量、触地不对称和最近20ms roll在机器分析中；并存不证明roll导致力峰。
- 实际抽样的body mass、逐shape材料及COM已保存在各场initial_runtime；初始PhysX joint_stiffness/damping不等于软件PD增益，当前评估器未捕获实际软件增益/随机延迟，保持unavailable，未扩大启动预算补测。历史初始运行时已捕获数组是否完全一致及差值见comparison；若一致仍不证明未捕获随机项相同。
- 训练并行资源：4次启动均exit0；GPU最小free 8548MiB，高于3072MiB保留量。训练从preflight iteration 41125推进至分析时41392；基线最近64次median FPS 117119.5，监测点最低即时FPS 46855.0（基线的40.0%）。分析时即时FPS 118322.0、最近64次median 116617.5。吞吐损失已授权，仅报告有限监测点，不能作为精确积分资源成本。原训练未收到任何信号。
- 保护核验：1259个既有文件SHA256均保持不变，tracked diff为空。标准runner将在全部分析完成后一次封存report/manifest，并追加一个evaluation_batch事件；只重建目标run的mutable index，其他旧证据保持原位。
- 当前随机化无push是质量/COM/材料/软件增益/延迟/观测噪声/起始扰动的联合条件；与前三场的差异不能归因为某一随机项。roll全程或足PD力矩单项降低都不能证明策略全面最优。单seed、少量完整周期不构成收敛或硬件验证；仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 随机化无push：完整周期anchor位置RMSE 0.1503→0.1428m、轮速RMSE 3.6726→3.2328rad/s、双轮法向力Z和峰3030.7→2684.5N；但roll峰7.883→7.995°，胫PD峰108.33→118.43Nm，足computed峰27.87→39.95Nm。足computed>27Nm由4→23个关节样本（4/6800=0.0588%、23/6800=0.3382%），applied保持27Nm上限且全关节applied超限为0。主要超限发生在准备阶段，不能由较低落地力峰掩盖。腾空0.255–0.275s仍短于计划0.300s，左右轮接触差在−20至+5ms。
- 同一10/9策略内部的zero-delay与fixed15ms合同，仅actuator delay边界0与3个物理步不同，可作为当前Native/seed42条件下的延迟配置对照：fixed15ms位置RMSE更小（0.04456→0.03611m），roll RMS更大（1.061→1.549°），双轮力和峰更高（2461.3→3148.5N）。play保留0..3步随机delay，实际抽样值未记录；不把此有限仿真对照推广为实物延迟因果，更不将跨独立训练策略差异归因为奖励边界。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [full-four-preflight-20261010-001.json](<../../evidence/training/full-four-preflight-20261010-001.json>)
- [resource-observations.json](<resource-observations.json>)
- [phase-analysis.json](<phase-analysis.json>)
- [manifest.json](<../../../2026-10-08_21-48-24/evaluations/history10-newref-20261009-001/manifest.json>)

## 建议与待授权事项

- 依据分阶段反指标讨论候选方案；当前证据不足以直接支持单项参数更改。若需验证1.65s边界的因果或扩大seed/场景、补充软件actuator诊断，先提交具体计划并取得新授权。
- 本批次不启动新训练、不改代码/配置/参考、不导出、不提交或推送。

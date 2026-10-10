# 评估批次 latest49999-reference1009-retry-20261010-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-10_01-25-39`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a02 | completed | play-equivalent，2000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a02/result.json>) / [日志](<raw/play-equivalent-a02/console.log>) |
| fixed-15ms-a02 | completed | fixed-15ms，2000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a02/result.json>) / [日志](<raw/fixed-15ms-a02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`b46799d67a3d0ad5e686257fe3832a0c41c5af29dc9a8e3cb1cef41ffc1c3d12`；runner：`OnPolicyRunner`。
- [场景来源 scenario-9555e3b5fdedd5186f0c120b3e2d2209218f92539009c542cf0cdf359da3df70.json](<../../provenance/scenario-9555e3b5fdedd5186f0c120b3e2d2209218f92539009c542cf0cdf359da3df70.json>)
- [场景来源 scenario-fc714f4520ddd77d83c2bbfd61318eec8a2c6db92647109dbe7720f10b096bb1.json](<../../provenance/scenario-fc714f4520ddd77d83c2bbfd61318eec8a2c6db92647109dbe7720f10b096bb1.json>)
- [训练上下文 context-76e09091681988239986b9e1ed36574d9892693c8042bcb3297115b7e1778442.json](<../../provenance/context-76e09091681988239986b9e1ed36574d9892693c8042bcb3297115b7e1778442.json>)
- [训练有效配置 config-af14ff75f436ff28e8c227255ee5ea4a03ef2c5f8c4f2c6e9d820b11cc19b5a8.json](<../../provenance/config-af14ff75f436ff28e8c227255ee5ea4a03ef2c5f8c4f2c6e9d820b11cc19b5a8.json>)

- `play-equivalent-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `fixed-15ms-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。

## 观察与限制

- 本批次仅重试首批未启动的两个用例；删除原场景中不存在的 events.randomize_apply_external_force_torque 覆盖，不修改项目源码。每个场景 Native 单环境 seed=42 2000 步，从第 0 帧开始。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- play-equivalent-a02 motion metrics: {"action_rate_rms": 0.2090735099655864, "anchor_orientation_rmse_rad": 0.06024292813116456, "anchor_position_rmse_m": 0.04520765031497954, "body_orientation_rmse_rad": 0.08910772831953373, "body_position_rmse_m": 0.027088938386118727, "censored_episodes": 1, "completed_episodes": 11, "failed_episodes": 0, "frame_zero_completed_episodes": 11, "frame_zero_success_fraction": 1.0, "frame_zero_successful_episodes": 11, "max_abs_action": 17.314804077148438, "max_abs_applied_torque": 94.93730926513672, "max_joint_effort_utilization": 0.7911442438761394, "mean_reward": 0.11032054928224534, "motion_success_fraction": 1.0, "position_joint_rmse_rad": 0.11612625967546365, "real_time_factor": 0.8502997688136639, "successful_motion_episodes": 11, "termination_count_anchor_ori": 0, "termination_count_anchor_pos": 0, "termination_count_ee_body_pos": 0, "termination_count_motion_finished": 11, "termination_count_time_out": 0, "torque_saturation_sample_fraction": 0.0, "wheel_velocity_rmse_rad_s": 2.3198073408723148}
- fixed-15ms-a02 motion metrics: {"action_rate_rms": 0.20160418314243897, "anchor_orientation_rmse_rad": 0.04809581324097261, "anchor_position_rmse_m": 0.042313949081978976, "body_orientation_rmse_rad": 0.08785036759510767, "body_position_rmse_m": 0.026021796336880128, "censored_episodes": 1, "completed_episodes": 11, "failed_episodes": 0, "frame_zero_completed_episodes": 11, "frame_zero_success_fraction": 1.0, "frame_zero_successful_episodes": 11, "max_abs_action": 9.396068572998047, "max_abs_applied_torque": 94.4569320678711, "max_joint_effort_utilization": 0.7871411005655925, "mean_reward": 0.11095314791426063, "motion_success_fraction": 1.0, "position_joint_rmse_rad": 0.10968550815768396, "real_time_factor": 0.8500239192962795, "successful_motion_episodes": 11, "termination_count_anchor_ori": 0, "termination_count_anchor_pos": 0, "termination_count_ee_body_pos": 0, "termination_count_motion_finished": 11, "termination_count_time_out": 0, "torque_saturation_sample_fraction": 0.0, "wheel_velocity_rmse_rad_s": 2.106762417235352}
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [summary-latest49999-20261010-001.json](<../../evidence/training/summary-latest49999-20261010-001.json>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

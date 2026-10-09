# 评估批次 checkpoint-40000-20261009-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-08_21-48-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| legacy-randomized-no-push-a01 | completed | legacy-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/legacy-randomized-no-push-a01/result.json>) / [日志](<raw/legacy-randomized-no-push-a01/console.log>) |
| current-randomized-no-push-a01 | completed | current-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/current-randomized-no-push-a01/result.json>) / [日志](<raw/current-randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ab2f18f4145743bafa23a42404309f818654d99cd32058c01affef50883b3fff`；runner：`OnPolicyRunner`。
- [场景来源 scenario-3d775b5d4a9909897ec4971d0a62761686a3f83407761df44eb0a15b1c2e087e.json](<../../provenance/scenario-3d775b5d4a9909897ec4971d0a62761686a3f83407761df44eb0a15b1c2e087e.json>)
- [场景来源 scenario-495d39b06fe3a8e2ff4ef1fec6b2714a7d5d4effb1505a4b8fc6125fd63ef790.json](<../../provenance/scenario-495d39b06fe3a8e2ff4ef1fec6b2714a7d5d4effb1505a4b8fc6125fd63ef790.json>)
- [场景来源 scenario-57a3c48446c729f73918305d37431719e29e9e8bd30708755e9536d8c4e50327.json](<../../provenance/scenario-57a3c48446c729f73918305d37431719e29e9e8bd30708755e9536d8c4e50327.json>)
- [场景来源 scenario-99d2b4806d6ac2790ca4bca83bae0ff6719ef79d1daeded1848c042aefacbcbb.json](<../../provenance/scenario-99d2b4806d6ac2790ca4bca83bae0ff6719ef79d1daeded1848c042aefacbcbb.json>)
- [场景来源 scenario-d9100fc79c7a263d5e3f565d29f3a97d671b13468911d250f0e326ee19c4acce.json](<../../provenance/scenario-d9100fc79c7a263d5e3f565d29f3a97d671b13468911d250f0e326ee19c4acce.json>)
- [训练上下文 context-8ae1eb140f3dc2c4f3772360965167adf951394a69bc85f58eea4a788ea1b571.json](<../../provenance/context-8ae1eb140f3dc2c4f3772360965167adf951394a69bc85f58eea4a788ea1b571.json>)
- [训练有效配置 config-6bc3906349a38b20aca8bdde3f6e31cf111ec38e6277c0db84c1d1ec95f9b329.json](<../../provenance/config-6bc3906349a38b20aca8bdde3f6e31cf111ec38e6277c0db84c1d1ec95f9b329.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `legacy-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `legacy-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_others": null, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6}`。
- `current-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `current-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.push_robot": null, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

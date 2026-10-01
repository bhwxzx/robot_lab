# 评估批次 startup-reset-20261001-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-09-30_19-50-46`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| zero-delay-a01 | completed | zero-delay，200 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，200 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| randomized-no-push-a01 | completed | randomized-no-push，200 / 1 / 42 | complete | [结果](<raw/randomized-no-push-a01/result.json>) / [日志](<raw/randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`782f9cedbe27a68fcfda54b9a47801250d585a17ca42cb08d509cf9c3c02db85`；runner：`OnPolicyRunner`。
- [场景来源 scenario-5254098f9caf65c9d711ccb49ebf19ceb3ccd50f66b2dec888fa19144339b013.json](<../../provenance/scenario-5254098f9caf65c9d711ccb49ebf19ceb3ccd50f66b2dec888fa19144339b013.json>)
- [场景来源 scenario-5b13d422dd0c324d6dddb9e23af6a3fa0a0d82cf0d6747baa7d14119c1ef2443.json](<../../provenance/scenario-5b13d422dd0c324d6dddb9e23af6a3fa0a0d82cf0d6747baa7d14119c1ef2443.json>)
- [场景来源 scenario-5f720dd4ace137fe93c6cb48b48099fc34bf27960594e8a20167ba4a7322e0c1.json](<../../provenance/scenario-5f720dd4ace137fe93c6cb48b48099fc34bf27960594e8a20167ba4a7322e0c1.json>)
- [训练上下文 context-c3e0507e7160afc30ff33f026069e73d99b027641d7c2fbe774151ec23e1da04.json](<../../provenance/context-c3e0507e7160afc30ff33f026069e73d99b027641d7c2fbe774151ec23e1da04.json>)
- [训练有效配置 config-64f55c031b41134030731554a055f85ce0e20f55596632a891a7d39dbda13f5b.json](<../../provenance/config-64f55c031b41134030731554a055f85ce0e20f55596632a891a7d39dbda13f5b.json>)

- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.initialize_reset_targets": true, "commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.initialize_reset_targets": true, "commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.initialize_reset_targets": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 141], "evaluation.motion_sha256": "48dd4ee594e132f447ee86cfda1c7b3d4fe2d87eb20e2c37a1ca84c24de31bda", "events.push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

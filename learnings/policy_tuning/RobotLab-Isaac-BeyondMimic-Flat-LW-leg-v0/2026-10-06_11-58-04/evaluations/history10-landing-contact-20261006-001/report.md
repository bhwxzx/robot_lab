# 评估批次 history10-landing-contact-20261006-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-06_11-58-04`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| randomized-no-push-a01 | completed | randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/randomized-no-push-a01/result.json>) / [日志](<raw/randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`72c5150a1bcf2f8b76ac33959113e4325e231e290fcd352772ccf4def156f0f1`；runner：`OnPolicyRunner`。
- [场景来源 scenario-21c3512921a7df61598cc3e69b76cc1c5ccd1263ba511d5001e4a0b808c693cb.json](<../../provenance/scenario-21c3512921a7df61598cc3e69b76cc1c5ccd1263ba511d5001e4a0b808c693cb.json>)
- [场景来源 scenario-4d5f4ce206804c84b7cb5f8c4c357d510cd78d467a54f996d75dc409f90f3164.json](<../../provenance/scenario-4d5f4ce206804c84b7cb5f8c4c357d510cd78d467a54f996d75dc409f90f3164.json>)
- [场景来源 scenario-5f79b5744d5dc6e9752607cd838c77b05afd5173897c406e87c445e0861fa686.json](<../../provenance/scenario-5f79b5744d5dc6e9752607cd838c77b05afd5173897c406e87c445e0861fa686.json>)
- [场景来源 scenario-81fbfbf636f4a5109d307f7c0dbf5e3a76b4747794415723af65398835a37ad9.json](<../../provenance/scenario-81fbfbf636f4a5109d307f7c0dbf5e3a76b4747794415723af65398835a37ad9.json>)
- [训练上下文 context-182098389054f462a09a94663a153e589072039a1a5f01e6d430bbde1a665094.json](<../../provenance/context-182098389054f462a09a94663a153e589072039a1a5f01e6d430bbde1a665094.json>)
- [训练有效配置 config-fb0b3fe2a99b6c805c2da0be7a45d26246d948c3cd902f48127068584a78f787.json](<../../provenance/config-fb0b3fe2a99b6c805c2da0be7a45d26246d948c3cd902f48127068584a78f787.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0.0, 0.0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

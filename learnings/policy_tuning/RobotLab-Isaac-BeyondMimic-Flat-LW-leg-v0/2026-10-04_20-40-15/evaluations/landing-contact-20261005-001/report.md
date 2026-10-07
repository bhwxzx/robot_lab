# 评估批次 landing-contact-20261005-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-04_20-40-15`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| randomized-no-push-a01 | completed | randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/randomized-no-push-a01/result.json>) / [日志](<raw/randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`7e3304d1650000da82aaa4c051839afdb1ce4b4a7e1891c8b2bf6294452241d1`；runner：`OnPolicyRunner`。
- [场景来源 scenario-14f02fb15365e243215b83a153889ad44d232a1be5c1160071beb657043dc7c3.json](<../../provenance/scenario-14f02fb15365e243215b83a153889ad44d232a1be5c1160071beb657043dc7c3.json>)
- [场景来源 scenario-27cf3d821b0e5a12126db7340b83e6bf30f600c510d4eca50fa83e2312d422fa.json](<../../provenance/scenario-27cf3d821b0e5a12126db7340b83e6bf30f600c510d4eca50fa83e2312d422fa.json>)
- [场景来源 scenario-74ed0958548035874e0b72351917472baef03689b2dd3da3d506657a4a8b8f0c.json](<../../provenance/scenario-74ed0958548035874e0b72351917472baef03689b2dd3da3d506657a4a8b8f0c.json>)
- [场景来源 scenario-fc636e11632f28ca4c139cb6fabb06d17def7a6b94ff7cd227e384e9652a0242.json](<../../provenance/scenario-fc636e11632f28ca4c139cb6fabb06d17def7a6b94ff7cd227e384e9652a0242.json>)
- [训练上下文 context-b2d9cee7d8fc1a19640204a87a4df449f5d23718fc614f4b293b32289ce5630b.json](<../../provenance/context-b2d9cee7d8fc1a19640204a87a4df449f5d23718fc614f4b293b32289ce5630b.json>)
- [训练有效配置 config-52c3dd9e382d6dac60c9c3c3696855be94dec1c68c77211bb72c84ddb8de0bc0.json](<../../provenance/config-52c3dd9e382d6dac60c9c3c3696855be94dec1c68c77211bb72c84ddb8de0bc0.json>)

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

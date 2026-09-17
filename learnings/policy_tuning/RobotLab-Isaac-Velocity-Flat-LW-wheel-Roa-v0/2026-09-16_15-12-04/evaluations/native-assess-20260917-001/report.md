# 评估批次 native-assess-20260917-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-09-16_15-12-04`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| baseline-no-push-a01 | completed | baseline-no-push，2100 / 4 / 42 | complete | [结果](<raw/baseline-no-push-a01/result.json>) / [日志](<raw/baseline-no-push-a01/console.log>) |
| current-push-a01 | completed | current-push，2100 / 4 / 42 | complete | [结果](<raw/current-push-a01/result.json>) / [日志](<raw/current-push-a01/console.log>) |
| nominal-no-noise-a01 | completed | nominal-no-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-no-noise-a01/result.json>) / [日志](<raw/nominal-no-noise-a01/console.log>) |
| nominal-training-noise-a01 | completed | nominal-training-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-training-noise-a01/result.json>) / [日志](<raw/nominal-training-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`6574370ca9df2dcc49a9ec0caa74172b94af2eb7d2bb66487eccc1a442fab5d8`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-3ca94d4b15aa81d1a91ad0e67839c90491f537fbb94e82f203b4f3947c110523.json](<../../provenance/scenario-3ca94d4b15aa81d1a91ad0e67839c90491f537fbb94e82f203b4f3947c110523.json>)
- [场景来源 scenario-ad9428c0d09b91ae866ffbeb27f08b45e75d485f7712a66efcb5fd377c4fa858.json](<../../provenance/scenario-ad9428c0d09b91ae866ffbeb27f08b45e75d485f7712a66efcb5fd377c4fa858.json>)
- [场景来源 scenario-d0a0ba97f0fa907dabd319710eda1b288f42cdc15ba805f3cee06f64b2c677d0.json](<../../provenance/scenario-d0a0ba97f0fa907dabd319710eda1b288f42cdc15ba805f3cee06f64b2c677d0.json>)
- [场景来源 scenario-f3861ace694269528e4bf178479ba3ebc53137ad03da577758b0d488173a4e25.json](<../../provenance/scenario-f3861ace694269528e4bf178479ba3ebc53137ad03da577758b0d488173a4e25.json>)
- [训练上下文 context-3fb9bef30d90c134f0d910ec5e5307b9c4420a1fa747e09d23169e80e86218ba.json](<../../provenance/context-3fb9bef30d90c134f0d910ec5e5307b9c4420a1fa747e09d23169e80e86218ba.json>)
- [训练有效配置 config-570cf09c96e2605c623c68f98e4fccc754dedce464b72621160d3a59dd670d6b.json](<../../provenance/config-570cf09c96e2605c623c68f98e4fccc754dedce464b72621160d3a59dd670d6b.json>)

- `baseline-no-push-a01` 指标：`{"max_joint_velocity_utilization": 0.6020063342470111, "max_tilt": 0.3558371663093567, "termination_rate": 0.0, "tracking_xy_rmse": 0.14945728622706902, "tracking_yaw_rmse": 0.22420607202442266}`
- `baseline-no-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot": null}`。
- `current-push-a01` 指标：`{"max_joint_velocity_utilization": 0.8459125287605055, "max_tilt": 0.26239126920700073, "termination_rate": 0.0, "tracking_xy_rmse": 0.1935542135370593, "tracking_yaw_rmse": 0.24904715364005592}`
- `current-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}}`。
- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.1884184241294861, "max_tilt": 0.07518812268972397, "termination_rate": 0.0, "tracking_xy_rmse": 0.012788101734400918, "tracking_yaw_rmse": 0.016762017744745776}`
- `nominal-no-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.2446896524140329, "max_tilt": 0.11632424592971802, "termination_rate": 0.0, "tracking_xy_rmse": 0.04109008866204322, "tracking_yaw_rmse": 0.036512384033734704}`
- `nominal-training-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

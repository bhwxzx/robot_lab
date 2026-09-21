# 评估批次 turning-sensor-age-20260921-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-09-19_11-41-48`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| right-sensor20-delay15-noise-a01 | completed | right-sensor20-delay15-noise，4500 / 1 / 42 | complete | [结果](<raw/right-sensor20-delay15-noise-a01/result.json>) / [日志](<raw/right-sensor20-delay15-noise-a01/console.log>) |
| left-sensor20-delay15-noise-a01 | completed | left-sensor20-delay15-noise，4500 / 1 / 42 | complete | [结果](<raw/left-sensor20-delay15-noise-a01/result.json>) / [日志](<raw/left-sensor20-delay15-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`fed8b88dc1bf7f0805cb039595233b1d49128739bc8888a434da70785740b053`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-e44519be743b5e498e685e0f97406e82c83470c1b040f71b3acba53904d91225.json](<../../provenance/scenario-e44519be743b5e498e685e0f97406e82c83470c1b040f71b3acba53904d91225.json>)
- [场景来源 scenario-f48ff8cae04666d0d5008fa27cfd766dfea638fea311dc1926845c1f7dc9f432.json](<../../provenance/scenario-f48ff8cae04666d0d5008fa27cfd766dfea638fea311dc1926845c1f7dc9f432.json>)
- [训练上下文 context-a52af23db0bd7b27e5fe221bc4855bc6c9c68290beeba6d310b79f4c45a1b9a2.json](<../../provenance/context-a52af23db0bd7b27e5fe221bc4855bc6c9c68290beeba6d310b79f4c45a1b9a2.json>)
- [训练有效配置 config-6ce9efb3c41a46b97f81db9ebc4cd58002ecd7700f583e35d36f192d6a5b10c7.json](<../../provenance/config-6ce9efb3c41a46b97f81db9ebc4cd58002ecd7700f583e35d36f192d6a5b10c7.json>)

- `right-sensor20-delay15-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.541914448593602, "max_tilt": 0.14710883796215057, "termination_rate": 0.0, "tracking_xy_rmse": 0.07676237916162684, "tracking_yaw_rmse": 0.26118024116489275}`
- `right-sensor20-delay15-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, -0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "evaluation.roa_sensor_delay_steps": 1, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `left-sensor20-delay15-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.38358179728190106, "max_tilt": 0.1318427473306656, "termination_rate": 0.0, "tracking_xy_rmse": 0.08651964373264481, "tracking_yaw_rmse": 0.2372026685750378}`
- `left-sensor20-delay15-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, 0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "evaluation.roa_sensor_delay_steps": 1, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

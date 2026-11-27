# 评估批次 commandamp-50000-stand-population-20261127-005

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-nominal-population-a05 | failed | stand-nominal-nopush，700 / 16 / 42 | — | — / [日志](<raw/stand-nominal-population-a05/console.log>) |
| stand-randomized-population-a05 | failed | stand-randomized-nopush，700 / 16 / 42 | — | — / [日志](<raw/stand-randomized-population-a05/console.log>) |

## 策略与场景


- `stand-nominal-population-a05` 命令调度：`[{"command": [0, 0, 0], "end_step": 699, "start_step": 0}]`；训练配置覆盖：`{"commands.base_velocity.ranges.ang_vel_z": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_x": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_y": [0.0, 0.0], "events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `stand-nominal-population-a05` 未完成原因：evaluation exit=-11; see console log
- `stand-randomized-population-a05` 命令调度：`[{"command": [0, 0, 0], "end_step": 699, "start_step": 0}]`；训练配置覆盖：`{"commands.base_velocity.ranges.ang_vel_z": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_x": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_y": [0.0, 0.0], "events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": true}`。
- `stand-randomized-population-a05` 未完成原因：evaluation exit=-11; see console log

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

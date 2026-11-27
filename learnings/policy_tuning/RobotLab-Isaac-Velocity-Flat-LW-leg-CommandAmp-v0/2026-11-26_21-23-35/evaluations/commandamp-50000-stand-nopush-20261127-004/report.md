# 评估批次 commandamp-50000-stand-nopush-20261127-004

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-nominal-a04 | completed | stand-nominal-nopush，700 / 16 / 42 | complete | [结果](<raw/stand-nominal-a04/result.json>) / [日志](<raw/stand-nominal-a04/console.log>) |
| stand-randomized-a04 | completed | stand-randomized-nopush，700 / 16 / 42 | complete | [结果](<raw/stand-randomized-a04/result.json>) / [日志](<raw/stand-randomized-a04/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-723ae2981ab01f62485c74b3477ed2826294b1036b9e09af2a31b64b7b963f6f.json](<../../provenance/scenario-723ae2981ab01f62485c74b3477ed2826294b1036b9e09af2a31b64b7b963f6f.json>)
- [场景来源 scenario-c8a69c51ba2a3878c558f93e58a32a71d620411308f028a139a95b41e0203b08.json](<../../provenance/scenario-c8a69c51ba2a3878c558f93e58a32a71d620411308f028a139a95b41e0203b08.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `stand-nominal-a04` 指标：`{"max_joint_velocity_utilization": 0.8444225311279296, "max_tilt": 0.0872391015291214, "termination_rate": 0.0, "tracking_xy_rmse": 0.04948371320194636, "tracking_yaw_rmse": 0.02724548469481055}`
- `stand-nominal-a04` 命令调度：`[{"start_step": 0, "end_step": 699, "command": [0, 0, 0]}]`；训练配置覆盖：`{"commands.base_velocity.ranges.ang_vel_z": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_x": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_y": [0.0, 0.0], "events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `stand-randomized-a04` 指标：`{"max_joint_velocity_utilization": 1.041325569152832, "max_tilt": 0.18906837701797485, "termination_rate": 0.0, "tracking_xy_rmse": 0.059825637697563945, "tracking_yaw_rmse": 0.06815145795893615}`
- `stand-randomized-a04` 命令调度：`[{"start_step": 0, "end_step": 699, "command": [0, 0, 0]}]`；训练配置覆盖：`{"commands.base_velocity.ranges.ang_vel_z": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_x": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_y": [0.0, 0.0], "events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": true}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

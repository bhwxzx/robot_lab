# 评估批次 zero-standing-plane-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Dwaq-v0`；训练：`2026-09-12_22-05-27`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-no-noise-a01 | completed | nominal-no-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-no-noise-a01/result.json>) / [日志](<raw/nominal-no-noise-a01/console.log>) |
| nominal-training-noise-a01 | completed | nominal-training-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-training-noise-a01/result.json>) / [日志](<raw/nominal-training-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`4007440e1dac0e48396aa87cc2454202ddb66de730c25790170b6d4c4bcf5652`；runner：`OnPolicyRunnerDwaq`。
- [场景来源 scenario-08f82215cdf6a5efa0c2a4344f9dad6f394d9e8c0561ef630b1930f40c936575.json](<../../provenance/scenario-08f82215cdf6a5efa0c2a4344f9dad6f394d9e8c0561ef630b1930f40c936575.json>)
- [场景来源 scenario-bda0dea2a89fe9842bd3a03d3656f851e44d71b09e54e59a7424550694ec4a92.json](<../../provenance/scenario-bda0dea2a89fe9842bd3a03d3656f851e44d71b09e54e59a7424550694ec4a92.json>)
- [训练上下文 context-54eedfcce866adfff0f4ca26b6663ebb25b3f777b62c6149451d7e772a92f74a.json](<../../provenance/context-54eedfcce866adfff0f4ca26b6663ebb25b3f777b62c6149451d7e772a92f74a.json>)
- [训练有效配置 config-f46d43603f25a340da12ac319f6fd9a594315e59c6bc510c30a5574a910f700f.json](<../../provenance/config-f46d43603f25a340da12ac319f6fd9a594315e59c6bc510c30a5574a910f700f.json>)

- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.22922367038148822, "max_tilt": 0.0628490149974823, "termination_rate": 0.0, "tracking_xy_rmse": 0.03368841097443354, "tracking_yaw_rmse": 0.0334046496150897}`
- `nominal-no-noise-a01` 命令调度：`[{"start_step": 0, "end_step": 5999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.26446041916355945, "max_tilt": 0.08337529748678207, "termination_rate": 0.0, "tracking_xy_rmse": 0.05492311940096451, "tracking_yaw_rmse": 0.05456229637762223}`
- `nominal-training-noise-a01` 命令调度：`[{"start_step": 0, "end_step": 5999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

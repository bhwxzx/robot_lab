# 评估批次 zero-standing-plane-20260912-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Dwaq-v0`；训练：`2026-09-11_17-04-10`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-no-noise-a01 | completed | nominal-no-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-no-noise-a01/result.json>) / [日志](<raw/nominal-no-noise-a01/console.log>) |
| nominal-training-noise-a01 | completed | nominal-training-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-training-noise-a01/result.json>) / [日志](<raw/nominal-training-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8798c9d1f744f5795ff7c9234147b7cab033295005056c18857fe83fbdbc22b5`；runner：`OnPolicyRunnerDwaq`。
- [场景来源 scenario-411da8f1cf27addab0e3527578da31307fcf3712fb105738b73718c63f4ab1b4.json](<../../provenance/scenario-411da8f1cf27addab0e3527578da31307fcf3712fb105738b73718c63f4ab1b4.json>)
- [场景来源 scenario-990e601d275e27a111394de9398e5f566273924b5ede9aa2d6c9da8d983879ca.json](<../../provenance/scenario-990e601d275e27a111394de9398e5f566273924b5ede9aa2d6c9da8d983879ca.json>)
- [训练上下文 context-2686b80cb207919f0fac07b73d48c355ae739c87b36f2a77b365851c006123ef.json](<../../provenance/context-2686b80cb207919f0fac07b73d48c355ae739c87b36f2a77b365851c006123ef.json>)
- [训练有效配置 config-5b31a80d98c92183b81556df703062b454a62b7df4091becebb2ffcb97a93730.json](<../../provenance/config-5b31a80d98c92183b81556df703062b454a62b7df4091becebb2ffcb97a93730.json>)

- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.20903938466852362, "max_tilt": 0.08290130645036697, "termination_rate": 0.0, "tracking_xy_rmse": 0.024299436307258553, "tracking_yaw_rmse": 0.024506947302419875}`
- `nominal-no-noise-a01` 命令调度：`[{"start_step": 0, "end_step": 5999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.2809575687755238, "max_tilt": 0.11738574504852295, "termination_rate": 0.0, "tracking_xy_rmse": 0.06245670623338201, "tracking_yaw_rmse": 0.05441757630403769}`
- `nominal-training-noise-a01` 命令调度：`[{"start_step": 0, "end_step": 5999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

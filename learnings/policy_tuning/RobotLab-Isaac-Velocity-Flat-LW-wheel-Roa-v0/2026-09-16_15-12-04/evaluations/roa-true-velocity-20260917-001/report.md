# 评估批次 roa-true-velocity-20260917-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-09-16_15-12-04`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| all-noise-student-true-velocity-a01 | completed | all-noise-student-true-velocity，6000 / 1 / 42 | complete | [结果](<raw/all-noise-student-true-velocity-a01/result.json>) / [日志](<raw/all-noise-student-true-velocity-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`6574370ca9df2dcc49a9ec0caa74172b94af2eb7d2bb66487eccc1a442fab5d8`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-52c1307c6ae0516fb029ff7bfd771e0584e6bd9d86a461b26f86b7d603a5674f.json](<../../provenance/scenario-52c1307c6ae0516fb029ff7bfd771e0584e6bd9d86a461b26f86b7d603a5674f.json>)
- [训练上下文 context-3fb9bef30d90c134f0d910ec5e5307b9c4420a1fa747e09d23169e80e86218ba.json](<../../provenance/context-3fb9bef30d90c134f0d910ec5e5307b9c4420a1fa747e09d23169e80e86218ba.json>)
- [训练有效配置 config-570cf09c96e2605c623c68f98e4fccc754dedce464b72621160d3a59dd670d6b.json](<../../provenance/config-570cf09c96e2605c623c68f98e4fccc754dedce464b72621160d3a59dd670d6b.json>)

- `all-noise-student-true-velocity-a01` 指标：`{"max_joint_velocity_utilization": 0.14862421787146365, "max_tilt": 0.08104713261127472, "termination_rate": 0.0, "tracking_xy_rmse": 0.019488788731624317, "tracking_yaw_rmse": 0.019807522390192272}`
- `all-noise-student-true-velocity-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "evaluation.roa_mode": "student_true_velocity", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

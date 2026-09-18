# 评估批次 native-assess-20260918-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-09-17_23-06-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| baseline-no-push-a01 | completed | baseline-no-push，2100 / 4 / 42 | complete | [结果](<raw/baseline-no-push-a01/result.json>) / [日志](<raw/baseline-no-push-a01/console.log>) |
| current-push-a01 | completed | current-push，2100 / 4 / 42 | complete | [结果](<raw/current-push-a01/result.json>) / [日志](<raw/current-push-a01/console.log>) |
| nominal-no-noise-a01 | completed | nominal-no-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-no-noise-a01/result.json>) / [日志](<raw/nominal-no-noise-a01/console.log>) |
| nominal-training-noise-a01 | completed | nominal-training-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-training-noise-a01/result.json>) / [日志](<raw/nominal-training-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`dd5b07a33de6f5e92524e68711d43357edbd8d4bf67db4dd43538ec009fed589`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-0199f423881cdf6929dcaec63be0e456a818dfddb3019e2f8befdb6540d08fdd.json](<../../provenance/scenario-0199f423881cdf6929dcaec63be0e456a818dfddb3019e2f8befdb6540d08fdd.json>)
- [场景来源 scenario-0da8008934391819dbed6a71482be5f5e9a60b9c8a174668fe8f959fd2d53844.json](<../../provenance/scenario-0da8008934391819dbed6a71482be5f5e9a60b9c8a174668fe8f959fd2d53844.json>)
- [场景来源 scenario-276ef08216d1269735843c1a4b99500723747da7b6db1a565446cd4c2c7f683b.json](<../../provenance/scenario-276ef08216d1269735843c1a4b99500723747da7b6db1a565446cd4c2c7f683b.json>)
- [场景来源 scenario-b48504f9fc11cae67023b36d2784f9f1e02530da70b659ad63758bcf09529088.json](<../../provenance/scenario-b48504f9fc11cae67023b36d2784f9f1e02530da70b659ad63758bcf09529088.json>)
- [训练上下文 context-7fb06d54b203e5e6500bd03793d4bab558d67bc27f2726e6f7a1a1e2feebc179.json](<../../provenance/context-7fb06d54b203e5e6500bd03793d4bab558d67bc27f2726e6f7a1a1e2feebc179.json>)
- [训练有效配置 config-676e89f6e4913fa7af9a3c47d26414808b40751865765f7d411a6c51b2922988.json](<../../provenance/config-676e89f6e4913fa7af9a3c47d26414808b40751865765f7d411a6c51b2922988.json>)

- `baseline-no-push-a01` 指标：`{"max_joint_velocity_utilization": 0.5317657933090673, "max_tilt": 0.31558966636657715, "termination_rate": 0.0, "tracking_xy_rmse": 0.15744036563980277, "tracking_yaw_rmse": 0.21855261564569337}`
- `baseline-no-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot": null}`。
- `current-push-a01` 指标：`{"max_joint_velocity_utilization": 0.8767577662612452, "max_tilt": 0.5179103016853333, "termination_rate": 0.0, "tracking_xy_rmse": 0.20217754366691365, "tracking_yaw_rmse": 0.25193773709544875}`
- `current-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}}`。
- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.19640685319900514, "max_tilt": 0.07049044221639633, "termination_rate": 0.0, "tracking_xy_rmse": 0.007542647211301955, "tracking_yaw_rmse": 0.013254351381331034}`
- `nominal-no-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.22269510500358813, "max_tilt": 0.12222040444612503, "termination_rate": 0.0, "tracking_xy_rmse": 0.0528764210568451, "tracking_yaw_rmse": 0.04574187821866278}`
- `nominal-training-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

# 评估批次 turn-nominal-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-left-nominal-a01 | completed | turn-left-nominal，1000 / 1 / 42 | complete | [结果](<raw/turn-left-nominal-a01/result.json>) / [视频](<raw/turn-left-nominal-a01/video.mp4>) / [日志](<raw/turn-left-nominal-a01/console.log>) |
| turn-right-nominal-a01 | completed | turn-right-nominal，1000 / 1 / 42 | complete | [结果](<raw/turn-right-nominal-a01/result.json>) / [视频](<raw/turn-right-nominal-a01/video.mp4>) / [日志](<raw/turn-right-nominal-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-13bed3269eda0ffab0d1e0a51c8008beadd12f8a9d8434ef9134905201b0c36a.json](<../../provenance/scenario-13bed3269eda0ffab0d1e0a51c8008beadd12f8a9d8434ef9134905201b0c36a.json>)
- [场景来源 scenario-ae6e8354d8285830cf3f6f98073feae589de06ce0c69f21b3ee30f76d71d444d.json](<../../provenance/scenario-ae6e8354d8285830cf3f6f98073feae589de06ce0c69f21b3ee30f76d71d444d.json>)
- [训练上下文 context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json](<../../provenance/context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `turn-left-nominal-a01` 指标：`{"max_joint_velocity_utilization": 0.6704298496246338, "max_tilt": 0.19683153927326202, "termination_rate": 0.0, "tracking_xy_rmse": 0.06282192370556616, "tracking_yaw_rmse": 0.2657559422182422}`
- `turn-left-nominal-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.add_joint_default_pos.params.pos_distribution_params": [0.0, 0.0], "events.randomize_actuator_gains.params.damping_distribution_params": [1.0, 1.0], "events.randomize_actuator_gains.params.stiffness_distribution_params": [1.0, 1.0], "events.randomize_com_positions.params.com_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_push_robot": null, "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [0.0, 0.0], "events.randomize_rigid_body_mass_others.params.mass_distribution_params": [1.0, 1.0], "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `turn-right-nominal-a01` 指标：`{"max_joint_velocity_utilization": 0.7685716152191162, "max_tilt": 0.13182272017002106, "termination_rate": 0.0, "tracking_xy_rmse": 0.0753973267206643, "tracking_yaw_rmse": 0.2396580493423438}`
- `turn-right-nominal-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.add_joint_default_pos.params.pos_distribution_params": [0.0, 0.0], "events.randomize_actuator_gains.params.damping_distribution_params": [1.0, 1.0], "events.randomize_actuator_gains.params.stiffness_distribution_params": [1.0, 1.0], "events.randomize_com_positions.params.com_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_push_robot": null, "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [0.0, 0.0], "events.randomize_rigid_body_mass_others.params.mass_distribution_params": [1.0, 1.0], "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

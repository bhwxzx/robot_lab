# 评估批次 unified-base1p5-pd0p9-model_49999-20261010-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-10_01-19-26`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| unified-base1p5-pd0p9-fixed15ms-a01 | completed | unified-base1p5-pd0p9-fixed15ms，1000 / 1 / 42 | complete | [结果](<raw/unified-base1p5-pd0p9-fixed15ms-a01/result.json>) / [日志](<raw/unified-base1p5-pd0p9-fixed15ms-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8968226ca5ccbdbacba308b3aaefe6cad0bc42599d0ca2da3cd15b22326a8d89`；runner：`OnPolicyRunner`。
- [场景来源 scenario-b12ba2d9b64b50bfcfd3e7eb9f848613dea68c7441e6af4accc833184ae0edb5.json](<../../provenance/scenario-b12ba2d9b64b50bfcfd3e7eb9f848613dea68c7441e6af4accc833184ae0edb5.json>)
- [场景来源 scenario-c11e3b837d6f87b9512f5a8ea42d8659168b07f4bded04ef25ddc370dccbec7b.json](<../../provenance/scenario-c11e3b837d6f87b9512f5a8ea42d8659168b07f4bded04ef25ddc370dccbec7b.json>)
- [场景来源 scenario-d95845d4d20609d01b4108e58ebfac2c1f5517e995a449afd3071c8a96c72bb0.json](<../../provenance/scenario-d95845d4d20609d01b4108e58ebfac2c1f5517e995a449afd3071c8a96c72bb0.json>)
- [训练上下文 context-00554a2ddb7239c20aa8a2a961ccd88c75700330fec55b29f95c473c260c05fe.json](<../../provenance/context-00554a2ddb7239c20aa8a2a961ccd88c75700330fec55b29f95c473c260c05fe.json>)
- [训练有效配置 config-4ba6cb50fa854f3ff81ed5d949e3459c1d1b37c4add0b376671987bb28d3b7e8.json](<../../provenance/config-4ba6cb50fa854f3ff81ed5d949e3459c1d1b37c4add0b376671987bb28d3b7e8.json>)

- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-10/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "7127fbcfc16c964f835e3c98bb8714975edde04e414367c86d72ceb1eb19ef48", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "observations.policy.history_length": 10, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.start_time_s": 1.0, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.start_time_s": 1.0, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6, "scene.robot.actuators.foots.damping": 1.4, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.foots.stiffness": 28.0, "scene.robot.actuators.legs.damping": 3.0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.legs.stiffness": 90.0, "scene.robot.actuators.wheels.damping": 0.5, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.robot.actuators.wheels.stiffness": 0.0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-10/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "7127fbcfc16c964f835e3c98bb8714975edde04e414367c86d72ceb1eb19ef48", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "observations.policy.history_length": 10, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.start_time_s": 1.0, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.start_time_s": 1.0, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6, "scene.robot.actuators.foots.damping": 1.4, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.foots.stiffness": 28.0, "scene.robot.actuators.legs.damping": 3.0, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.legs.stiffness": 90.0, "scene.robot.actuators.wheels.damping": 0.5, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.robot.actuators.wheels.stiffness": 0.0}`。
- `unified-base1p5-pd0p9-fixed15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `unified-base1p5-pd0p9-fixed15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-10/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "7127fbcfc16c964f835e3c98bb8714975edde04e414367c86d72ceb1eb19ef48", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [1.5, 1.5], "events.randomize_rigid_body_mass_base.params.operation": "add", "events.randomize_rigid_body_mass_base.params.recompute_inertia": true, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "observations.policy.history_length": 10, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.start_time_s": 1.0, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.start_time_s": 1.0, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6, "scene.robot.actuators.foots.damping": 1.26, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.foots.stiffness": 25.2, "scene.robot.actuators.legs.damping": 2.7, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.legs.stiffness": 81.0, "scene.robot.actuators.wheels.damping": 0.45, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.robot.actuators.wheels.stiffness": 0.0}`。

## 观察与限制

- zero-delay: 5 full successful cycles, 0 failure/incomplete ended, 1 censored tails; physics5ms/control20ms.
- Full-success population: foot request peak 22.960Nm; shank drive 101.349Nm; roll 1.963deg; wheel net normal-Z sum 2154.724N averaged5ms.
- fixed-15ms: 5 full successful cycles, 0 failure/incomplete ended, 1 censored tails; physics5ms/control20ms.
- Full-success population: foot request peak 22.697Nm; shank drive 106.751Nm; roll 2.248deg; wheel net normal-Z sum 2276.028N averaged5ms.
- unified-base1p5-pd0p9-fixed15ms: 5 full successful cycles, 0 failure/incomplete ended, 1 censored tails; physics5ms/control20ms.
- Full-success population: foot request peak 20.053Nm; shank drive 110.823Nm; roll 1.832deg; wheel net normal-Z sum 1646.777N averaged5ms.
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [cycle-phase-analysis.json](<cycle-phase-analysis.json>)
- [plan.json](<../../../comparisons/unified-base1p5-pd0p9-20261010-001/plan.json>)

## 建议与待授权事项

- 按完整遥测与统一指标比较；此批次不自动重试。

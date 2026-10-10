# 评估批次 unified-base1p5-pd0p9-model_49999-20261010-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-07_15-57-49`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| unified-base1p5-pd0p9-fixed15ms-a01 | completed | unified-base1p5-pd0p9-fixed15ms，1000 / 1 / 42 | complete | [结果](<raw/unified-base1p5-pd0p9-fixed15ms-a01/result.json>) / [日志](<raw/unified-base1p5-pd0p9-fixed15ms-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`d1ae8cb7ca4496442f47db7b2e6ce9cf28520347dc1f746960ce3929a54be393`；runner：`OnPolicyRunner`。
- [场景来源 scenario-e8bcda1b72c13195964ae1c16c8809061456ab8052a74e54f585130399507141.json](<../../provenance/scenario-e8bcda1b72c13195964ae1c16c8809061456ab8052a74e54f585130399507141.json>)
- [训练上下文 context-99e40fe9376d014631661027482fb000013618a580720d6fcb23f383998b6689.json](<../../provenance/context-99e40fe9376d014631661027482fb000013618a580720d6fcb23f383998b6689.json>)
- [训练有效配置 config-6d9f4d5d424450f302b365a529f1d8ce5e210db1b254a0460565f9d004fceba7.json](<../../provenance/config-6d9f4d5d424450f302b365a529f1d8ce5e210db1b254a0460565f9d004fceba7.json>)

- `unified-base1p5-pd0p9-fixed15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `unified-base1p5-pd0p9-fixed15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-01/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [1.5, 1.5], "events.randomize_rigid_body_mass_base.params.operation": "add", "events.randomize_rigid_body_mass_base.params.recompute_inertia": true, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "observations.policy.history_length": 10, "rewards.leg_symmetry.params.start_time_s": 1.6, "rewards.motion_anchor_roll_horizontal": null, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.6, "rewards.motion_takeoff_anchor_ori.params.start_time_s": 1.0, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.6, "rewards.motion_takeoff_pitch_ang_vel.params.start_time_s": 1.0, "rewards.same_foot_x_position.params.start_time_s": 1.6, "rewards.wheel_contact_continuous.params.start_time_s": 1.6, "scene.robot.actuators.foots.damping": 1.26, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.foots.stiffness": 25.2, "scene.robot.actuators.legs.damping": 2.7, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.legs.stiffness": 81.0, "scene.robot.actuators.wheels.damping": 0.45, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.robot.actuators.wheels.stiffness": 0.0}`。

## 观察与限制

- unified-base1p5-pd0p9-fixed15ms: 5 full successful cycles, 0 failure/incomplete ended, 1 censored tails; physics5ms/control20ms.
- Full-success population: foot request peak 20.289Nm; shank drive 118.721Nm; roll 6.829deg; wheel net normal-Z sum 2152.292N averaged5ms.
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [cycle-phase-analysis.json](<cycle-phase-analysis.json>)
- [plan.json](<../../../comparisons/unified-base1p5-pd0p9-20261010-001/plan.json>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

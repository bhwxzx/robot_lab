# 评估批次 latest49999-reference1009-20261010-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-10_01-25-39`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| zero-delay-a01 | completed | zero-delay，2000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| play-equivalent-a01 | failed | play-equivalent，2000 / 1 / 42 | — | — / [日志](<raw/play-equivalent-a01/console.log>) |
| fixed-15ms-a01 | failed | fixed-15ms，2000 / 1 / 42 | — | — / [日志](<raw/fixed-15ms-a01/console.log>) |
| training-randomized-a01 | completed | training-randomized，2000 / 1 / 42 | complete | [结果](<raw/training-randomized-a01/result.json>) / [日志](<raw/training-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`b46799d67a3d0ad5e686257fe3832a0c41c5af29dc9a8e3cb1cef41ffc1c3d12`；runner：`OnPolicyRunner`。
- [场景来源 scenario-aa6be8ecd36bc39815d81966fe24db3c65d63811200047e3efbb348b431de05c.json](<../../provenance/scenario-aa6be8ecd36bc39815d81966fe24db3c65d63811200047e3efbb348b431de05c.json>)
- [场景来源 scenario-c6ebce903fb79668afa7dc6a7e2b9a84f1bba201cd2fd3a7b448939417eea6a6.json](<../../provenance/scenario-c6ebce903fb79668afa7dc6a7e2b9a84f1bba201cd2fd3a7b448939417eea6a6.json>)
- [训练上下文 context-76e09091681988239986b9e1ed36574d9892693c8042bcb3297115b7e1778442.json](<../../provenance/context-76e09091681988239986b9e1ed36574d9892693c8042bcb3297115b7e1778442.json>)
- [训练有效配置 config-af14ff75f436ff28e8c227255ee5ea4a03ef2c5f8c4f2c6e9d820b11cc19b5a8.json](<../../provenance/config-af14ff75f436ff28e8c227255ee5ea4a03ef2c5f8c4f2c6e9d820b11cc19b5a8.json>)

- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293"}`。
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `play-equivalent-a01` 未完成原因：evaluation exit=1; see console log
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `fixed-15ms-a01` 未完成原因：evaluation exit=1; see console log
- `training-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-09/leg_to_wheel_transform_50hz.npz", "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "47805c271159df64c6ebc0a40e283bfa2361333ee7de6d0698e33f371bd02293"}`。

## 观察与限制

- 本批次为 Native 单环境 seed=42 的有限仿真评估；收集完成不等于策略通过。完整轨迹成功仅计 episode_start_frame=0 的已结束回合，尾段单独记为 censored。
- 训练启动命令经用户确认：bash scripts/start_beyondmimic.sh --type leg；W&B 原训练源码提交 2b2bb9e7e37b54652ca60164e948e33a9bf4b3d2，与本次采集提交的相关源码一致。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- zero-delay-a01 motion metrics: {"action_rate_rms": 0.199421337553991, "anchor_orientation_rmse_rad": 0.05663326328950954, "anchor_position_rmse_m": 0.03460926155572356, "body_orientation_rmse_rad": 0.0829020510371894, "body_position_rmse_m": 0.02757297450772265, "censored_episodes": 1, "completed_episodes": 11, "failed_episodes": 0, "frame_zero_completed_episodes": 11, "frame_zero_success_fraction": 1.0, "frame_zero_successful_episodes": 11, "max_abs_action": 10.064905166625977, "max_abs_applied_torque": 81.3468017578125, "max_joint_effort_utilization": 0.6778900146484375, "mean_reward": 0.11102844988182187, "motion_success_fraction": 1.0, "position_joint_rmse_rad": 0.11760485761747431, "real_time_factor": 0.8550828691482367, "successful_motion_episodes": 11, "termination_count_anchor_ori": 0, "termination_count_anchor_pos": 0, "termination_count_ee_body_pos": 0, "termination_count_motion_finished": 11, "termination_count_time_out": 0, "torque_saturation_sample_fraction": 0.0, "wheel_velocity_rmse_rad_s": 2.223063656908726}
- training-randomized-a01 motion metrics: {"action_rate_rms": 0.2660821691452312, "anchor_orientation_rmse_rad": 0.12135233064040471, "anchor_position_rmse_m": 0.15427564985648187, "body_orientation_rmse_rad": 0.13913547289488143, "body_position_rmse_m": 0.04018039909770252, "censored_episodes": 1, "completed_episodes": 11, "failed_episodes": 0, "frame_zero_completed_episodes": 11, "frame_zero_success_fraction": 1.0, "frame_zero_successful_episodes": 11, "max_abs_action": 13.00682258605957, "max_abs_applied_torque": 120.0, "max_joint_effort_utilization": 1.0, "mean_reward": 0.10300549233763014, "motion_success_fraction": 1.0, "position_joint_rmse_rad": 0.14202275073604564, "real_time_factor": 0.8464133141943813, "successful_motion_episodes": 11, "termination_count_anchor_ori": 0, "termination_count_anchor_pos": 0, "termination_count_ee_body_pos": 0, "termination_count_motion_finished": 11, "termination_count_time_out": 0, "torque_saturation_sample_fraction": 0.0004, "wheel_velocity_rmse_rad_s": 2.5160520165505593}
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

# 评估批次 final49999-quick-play-20261010-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-09_12-49-52`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，500 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`8909d6a6c2db883fc3dd37843053fdb887f90b92ead76cff036f8341b5cdd2b1`；runner：`OnPolicyRunner`。
- [场景来源 scenario-b8e1752cdb33ebf637872c42a5ebf05493f4c2d80df36f7730f072802d2d4e89.json](<../../provenance/scenario-b8e1752cdb33ebf637872c42a5ebf05493f4c2d80df36f7730f072802d2d4e89.json>)
- [训练上下文 context-719efc869d9200b0c248799a18fe8215e297c0727602acf3cc3cb4f908885cb5.json](<../../provenance/context-719efc869d9200b0c248799a18fe8215e297c0727602acf3cc3cb4f908885cb5.json>)
- [训练有效配置 config-ed83bdbc225d9af2a6db3b732873a090767a80ec6db51bff78a6a5895739bfad.json](<../../provenance/config-ed83bdbc225d9af2a6db3b732873a090767a80ec6db51bff78a6a5895739bfad.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-08/leg_to_wheel_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.leg_symmetry.params.start_time_s": 1.65, "rewards.motion_anchor_roll_horizontal": null, "rewards.motion_takeoff_anchor_ori.params.end_time_s": 1.65, "rewards.motion_takeoff_pitch_ang_vel.params.end_time_s": 1.65, "rewards.same_foot_x_position.params.start_time_s": 1.65, "rewards.wheel_contact_continuous.params.start_time_s": 1.65}`。

## 观察与限制

- 这是 final49999 的单场景、单 seed、500步有限评估；不等同五场景正式比较或收敛判断。
- 原训练 HEAD 为 0eb1b88 加保存的 flat 配置 diff；本次评估 HEAD 为 2b2bb9e。运行时恢复10/8参考、五个1.65秒边界，并禁用后来新增的roll奖励；源码保持当前配置。
- Play等效关闭观测噪声、起始扰动和事件随机化，保留原actuator delay；不等同零延迟组。
- 完整周期指标：{"action_rate_rms": 0.2047144288410587, "anchor_orientation_rmse_rad": 0.039723822824310986, "anchor_position_rmse_m": 0.04178712033297485, "body_orientation_rmse_rad": 0.08134083585164616, "body_position_rmse_m": 0.02836623194157289, "censored_episodes": 1, "completed_episodes": 2, "failed_episodes": 0, "frame_zero_completed_episodes": 2, "frame_zero_success_fraction": 1.0, "frame_zero_successful_episodes": 2, "max_abs_action": 14.469882011413574, "max_abs_applied_torque": 93.82344055175781, "max_joint_effort_utilization": 0.7818620045979817, "mean_reward": 0.10087057812511921, "motion_success_fraction": 1.0, "position_joint_rmse_rad": 0.11177788998950529, "real_time_factor": 0.3672154951051251, "successful_motion_episodes": 2, "termination_count_anchor_ori": 0, "termination_count_anchor_pos": 0, "termination_count_ee_body_pos": 0, "termination_count_motion_finished": 2, "termination_count_time_out": 0, "torque_saturation_sample_fraction": 0.0, "wheel_velocity_rmse_rad_s": 3.116299200204891}
- 分阶段遥测：{"phase": "preparation", "reference_time_start_s": 0.0, "reference_time_end_s": 1.0, "physics_samples": 400, "control_samples": 100, "foot_max_abs_applied_nm": 5.96064567565918, "foot_max_abs_computed_nm": 5.96064567565918, "foot_computed_over_limit_fraction": 0.0, "shank_max_abs_applied_nm": 33.638572692871094, "max_single_foot_normal_force_n": 167.20516967773588, "max_summed_wheel_normal_force_z_n": 0.0, "max_abs_roll_deg": 2.0295021367518165, "roll_rms_deg": 0.7016896745376103}
- 分阶段遥测：{"phase": "push_off", "reference_time_start_s": 1.0, "reference_time_end_s": 1.3, "physics_samples": 120, "control_samples": 30, "foot_max_abs_applied_nm": 9.078542709350586, "foot_max_abs_computed_nm": 9.078542709350586, "foot_computed_over_limit_fraction": 0.0, "shank_max_abs_applied_nm": 98.3580093383789, "max_single_foot_normal_force_n": 345.6741943359406, "max_summed_wheel_normal_force_z_n": 0.0, "max_abs_roll_deg": 2.8985611316102182, "roll_rms_deg": 1.757732435324136}
- 分阶段遥测：{"phase": "reference_flight", "reference_time_start_s": 1.3, "reference_time_end_s": 1.6, "physics_samples": 120, "control_samples": 30, "foot_max_abs_applied_nm": 9.710338592529297, "foot_max_abs_computed_nm": 9.710338592529297, "foot_computed_over_limit_fraction": 0.0, "shank_max_abs_applied_nm": 75.13914489746094, "max_single_foot_normal_force_n": 106.26186370849705, "max_summed_wheel_normal_force_z_n": 2448.6363220214844, "max_abs_roll_deg": 3.1879412026746405, "roll_rms_deg": 2.4589629800439456}
- 分阶段遥测：{"phase": "landing", "reference_time_start_s": 1.6, "reference_time_end_s": 1.8, "physics_samples": 80, "control_samples": 20, "foot_max_abs_applied_nm": 2.9407260417938232, "foot_max_abs_computed_nm": 2.9407260417938232, "foot_computed_over_limit_fraction": 0.0, "shank_max_abs_applied_nm": 38.10191345214844, "max_single_foot_normal_force_n": 0.0, "max_summed_wheel_normal_force_z_n": 617.7987670898438, "max_abs_roll_deg": 0.8661328821137091, "roll_rms_deg": 0.4729294444027822}
- 分阶段遥测：{"phase": "wheel", "reference_time_start_s": 1.8, "reference_time_end_s": null, "physics_samples": 640, "control_samples": 160, "foot_max_abs_applied_nm": 0.7792357206344604, "foot_max_abs_computed_nm": 0.7792357206344604, "foot_computed_over_limit_fraction": 0.0, "shank_max_abs_applied_nm": 16.104005813598633, "max_single_foot_normal_force_n": 0.0, "max_summed_wheel_normal_force_z_n": 468.98988342285156, "max_abs_roll_deg": 0.9855763518792697, "roll_rms_deg": 0.41539918004633614}
- 腾空操作定义与阈值敏感性：[{"episode_id": 0, "threshold_n": 1.0, "duration_s": 0.255, "first_air_sample_reference_time_s": 1.325}, {"episode_id": 0, "threshold_n": 10.0, "duration_s": 0.255, "first_air_sample_reference_time_s": 1.325}, {"episode_id": 1, "threshold_n": 1.0, "duration_s": 0.26, "first_air_sample_reference_time_s": 1.325}, {"episode_id": 1, "threshold_n": 10.0, "duration_s": 0.26, "first_air_sample_reference_time_s": 1.325}]
- 5 ms PhysX net normal contact forces; averages over the physics step, excluding friction. Not physical hardware impact peaks. PD applied/computed drive torque, excluding passive impact/internal loads.
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [quick-play-preflight-20261010-001.json](<../../evidence/training/quick-play-preflight-20261010-001.json>)
- [resource-observations.json](<resource-observations.json>)
- [phase-analysis.json](<phase-analysis.json>)

## 建议与待授权事项

- 根据本次完整周期和分阶段数据讨论参数候选；扩大到五场景或其他checkpoint需要另定范围。不能把训练完成或本次完成率视作全面最优。

# 评估批次 wheel-to-leg-knee-ground-20261022-005

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-21_10-48-07`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`4a8cfbaf37057ef942be44c3f4bea31ac37c16c14b243952e4ccadc4577ab957`；runner：`OnPolicyRunner`。
- [场景来源 scenario-bbec9197ab92335c030ce02533195dd579ffe9f047d1405fb70f7b10d760df8d.json](<../../provenance/scenario-bbec9197ab92335c030ce02533195dd579ffe9f047d1405fb70f7b10d760df8d.json>)
- [训练上下文 context-1e5b275b1b62abfcdf4ee21f97148945bac7b7931355f9da4eddd47fd2636965.json](<../../provenance/context-1e5b275b1b62abfcdf4ee21f97148945bac7b7931355f9da4eddd47fd2636965.json>)
- [训练有效配置 config-d005b9662d06d2aa82e0f81500f425c8fd1bf8abe6a279280963a126f6a27dc4.json](<../../provenance/config-d005b9662d06d2aa82e0f81500f425c8fd1bf8abe6a279280963a126f6a27dc4.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.0}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

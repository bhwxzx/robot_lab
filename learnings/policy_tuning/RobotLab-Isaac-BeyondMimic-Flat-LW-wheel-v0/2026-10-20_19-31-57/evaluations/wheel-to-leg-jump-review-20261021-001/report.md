# 评估批次 wheel-to-leg-jump-review-20261021-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-20_19-31-57`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-zero-delay-a01 | completed | nominal-zero-delay，1000 / 1 / 42 | complete | [结果](<raw/nominal-zero-delay-a01/result.json>) / [日志](<raw/nominal-zero-delay-a01/console.log>) |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| training-random-start-a01 | completed | training-random-start，1000 / 1 / 42 | complete | [结果](<raw/training-random-start-a01/result.json>) / [日志](<raw/training-random-start-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`668eddb7697bca0cb70a73d5121744f28b39453adacc572e32205dd3c2b3f82e`；runner：`OnPolicyRunner`。
- [场景来源 scenario-382839fb159d9d5373af5d54fb43139d1a7ecef4429436ba578c1f1d19541af8.json](<../../provenance/scenario-382839fb159d9d5373af5d54fb43139d1a7ecef4429436ba578c1f1d19541af8.json>)
- [场景来源 scenario-4f04812cdb79cc8a97c5fac13f6f140d88826b424decfddc671dc79aec0d936c.json](<../../provenance/scenario-4f04812cdb79cc8a97c5fac13f6f140d88826b424decfddc671dc79aec0d936c.json>)
- [场景来源 scenario-5695c2e67de334b1ec38e2f1c3d205e197b8f66971a2f2bae858623a5c09cb5a.json](<../../provenance/scenario-5695c2e67de334b1ec38e2f1c3d205e197b8f66971a2f2bae858623a5c09cb5a.json>)
- [场景来源 scenario-b79cb713a4d44b86fd00ab3c8b13741e13d71287ce92d0b6c76cd2e9403d6374.json](<../../provenance/scenario-b79cb713a4d44b86fd00ab3c8b13741e13d71287ce92d0b6c76cd2e9403d6374.json>)
- [训练上下文 context-cf0117fb529d39a29b4fa614020fd68bd8b38dc64b7b596f03bd58e86dd27792.json](<../../provenance/context-cf0117fb529d39a29b4fa614020fd68bd8b38dc64b7b596f03bd58e86dd27792.json>)
- [训练有效配置 config-1a79a3874a4544359290606a895a632f6994c0554f0caafebe2137d9f848d811.json](<../../provenance/config-1a79a3874a4544359290606a895a632f6994c0554f0caafebe2137d9f848d811.json>)

- `nominal-zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62"}`。
- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `training-random-start-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-start-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62"}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

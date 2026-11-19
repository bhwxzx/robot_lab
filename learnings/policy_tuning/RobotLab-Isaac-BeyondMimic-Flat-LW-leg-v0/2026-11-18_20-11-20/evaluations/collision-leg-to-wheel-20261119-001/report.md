# 评估批次 collision-leg-to-wheel-20261119-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-11-18_20-11-20`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-start-20261119-a01 | completed | nominal-start，2000 / 1 / 42 | complete | [结果](<raw/nominal-start-20261119-a01/result.json>) / [日志](<raw/nominal-start-20261119-a01/console.log>) |
| randomized-start-20261119-a01 | completed | randomized-start，2000 / 1 / 42 | complete | [结果](<raw/randomized-start-20261119-a01/result.json>) / [日志](<raw/randomized-start-20261119-a01/console.log>) |
| randomized-phase-20261119-a01 | completed | randomized-phase，2000 / 1 / 42 | complete | [结果](<raw/randomized-phase-20261119-a01/result.json>) / [日志](<raw/randomized-phase-20261119-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`a8701a796f5f28593ae457e9116ad4ce208a43eb249e25ae6a96346a748b91c3`；runner：`OnPolicyRunner`。
- [场景来源 scenario-65cc446512c6100cdd52532941ddebd650b8af1a276a1340782d5fa2216a6455.json](<../../provenance/scenario-65cc446512c6100cdd52532941ddebd650b8af1a276a1340782d5fa2216a6455.json>)
- [场景来源 scenario-73cff631e92fba34704d41a421a518eac5b9a651f1efbeacbf8704f38f3c1027.json](<../../provenance/scenario-73cff631e92fba34704d41a421a518eac5b9a651f1efbeacbf8704f38f3c1027.json>)
- [场景来源 scenario-7e1660b587cbdb6840f540ba44a0acde3cf1f20906ea99844ed71d147b84b452.json](<../../provenance/scenario-7e1660b587cbdb6840f540ba44a0acde3cf1f20906ea99844ed71d147b84b452.json>)
- [训练上下文 context-f4313d3b185c07766f1feaf0ad46a2091acb0354b384c4e763698de0f22f7b02.json](<../../provenance/context-f4313d3b185c07766f1feaf0ad46a2091acb0354b384c4e763698de0f22f7b02.json>)
- [训练有效配置 config-e9c231e606c10c8e026c61127e578c2bab45fae6cbcb88c0f44ae9ed71c4c7e7.json](<../../provenance/config-e9c231e606c10c8e026c61127e578c2bab45fae6cbcb88c0f44ae9ed71c4c7e7.json>)

- `nominal-start-20261119-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-start-20261119-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。
- `randomized-start-20261119-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-start-20261119-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。
- `randomized-phase-20261119-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-phase-20261119-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

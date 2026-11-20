# 评估批次 symmetry-off-leg-to-wheel-20261120-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-11-19_21-30-58`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-start-20261120-a01 | completed | nominal-start，2000 / 1 / 42 | complete | [结果](<raw/nominal-start-20261120-a01/result.json>) / [日志](<raw/nominal-start-20261120-a01/console.log>) |
| randomized-start-20261120-a01 | completed | randomized-start，2000 / 1 / 42 | complete | [结果](<raw/randomized-start-20261120-a01/result.json>) / [日志](<raw/randomized-start-20261120-a01/console.log>) |
| randomized-phase-20261120-a01 | completed | randomized-phase，2000 / 1 / 42 | complete | [结果](<raw/randomized-phase-20261120-a01/result.json>) / [日志](<raw/randomized-phase-20261120-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`64b704206de696421244719f441605c472a3e0b592715531c2383dc0c1e1dac1`；runner：`OnPolicyRunner`。
- [场景来源 scenario-28288c3ce3b0d084f636941ddba8aa8f84aa1fad7fa4579297b20c62b1ca5444.json](<../../provenance/scenario-28288c3ce3b0d084f636941ddba8aa8f84aa1fad7fa4579297b20c62b1ca5444.json>)
- [场景来源 scenario-83c0acd2024ed5ba695b998ca183f14b74c9e81e1bd4569f982dee0d36623316.json](<../../provenance/scenario-83c0acd2024ed5ba695b998ca183f14b74c9e81e1bd4569f982dee0d36623316.json>)
- [场景来源 scenario-ae4af9628f2a41465985d5f4bf0a144692db4947cf99c5d359e782def4f8ab81.json](<../../provenance/scenario-ae4af9628f2a41465985d5f4bf0a144692db4947cf99c5d359e782def4f8ab81.json>)
- [训练上下文 context-79dc588472bdd3e65631be1322bdb6cb11f21b07efdb7bfe42ce89f4d3871bdf.json](<../../provenance/context-79dc588472bdd3e65631be1322bdb6cb11f21b07efdb7bfe42ce89f4d3871bdf.json>)
- [训练有效配置 config-95b19b87a5c326f1fabeff1fcf5913f4d6e1b63513030b28cd9e14af25d782ef.json](<../../provenance/config-95b19b87a5c326f1fabeff1fcf5913f4d6e1b63513030b28cd9e14af25d782ef.json>)

- `nominal-start-20261120-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-start-20261120-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。
- `randomized-start-20261120-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-start-20261120-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。
- `randomized-phase-20261120-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-phase-20261120-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

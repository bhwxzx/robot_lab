# 评估批次 wheel-to-leg-fixed-start-20261027-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-26_14-18-23`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a02 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a02/result.json>) / [日志](<raw/fixed-start-nominal-a02/console.log>) |
| fixed-start-randomized-a02 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a02/result.json>) / [日志](<raw/fixed-start-randomized-a02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ce31934b0a22a6877e3118776e5a174dcf5622033f1f50bfe805076138caf60e`；runner：`OnPolicyRunner`。
- [场景来源 scenario-0cc8468b21c46dadd1292c92d0f522cbc9915ac6d2c1685087e34fba00cc0734.json](<../../provenance/scenario-0cc8468b21c46dadd1292c92d0f522cbc9915ac6d2c1685087e34fba00cc0734.json>)
- [场景来源 scenario-6192a25cfd920bbf4bdbf277c44cfc470e5937aa9315475dc14e5be893645161.json](<../../provenance/scenario-6192a25cfd920bbf4bdbf277c44cfc470e5937aa9315475dc14e5be893645161.json>)
- [训练上下文 context-db4e2cc35b6f398152ac032c9ede1af26cbf3a124284f17967f3c93b7a5c789c.json](<../../provenance/context-db4e2cc35b6f398152ac032c9ede1af26cbf3a124284f17967f3c93b7a5c789c.json>)
- [训练有效配置 config-a812c5426d34ab3672148b295270f60dc468e1ce7b0162a880525ba15c8a7e54.json](<../../provenance/config-a812c5426d34ab3672148b295270f60dc468e1ce7b0162a880525ba15c8a7e54.json>)

- `fixed-start-nominal-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a02` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a02` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

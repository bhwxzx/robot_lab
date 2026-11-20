# 评估批次 wheel-to-leg-symmetry-off-20261120-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-19_21-40-19`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`fb2737bf0dd59c45373ebadceb19d55bfa2cb7ab0e796d1d827df9335354b284`；runner：`OnPolicyRunner`。
- [场景来源 scenario-774ce499e38b2ba7d538f481565affb021a53d30d57f4bb9e85f0d6c31bd2796.json](<../../provenance/scenario-774ce499e38b2ba7d538f481565affb021a53d30d57f4bb9e85f0d6c31bd2796.json>)
- [场景来源 scenario-8a81636419874cf81cd1a4e9c93f0c7df55fca80828d8f22c8a287078c473a68.json](<../../provenance/scenario-8a81636419874cf81cd1a4e9c93f0c7df55fca80828d8f22c8a287078c473a68.json>)
- [训练上下文 context-00d8c23bf5e0270b434888e3a26d79a4bee7c57d10c3ecc6371f465a651b2c4e.json](<../../provenance/context-00d8c23bf5e0270b434888e3a26d79a4bee7c57d10c3ecc6371f465a651b2c4e.json>)
- [训练有效配置 config-172f16131739b06b24adae01da673d25b56059cbb383dd1dc9992a176e487a13.json](<../../provenance/config-172f16131739b06b24adae01da673d25b56059cbb383dd1dc9992a176e487a13.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

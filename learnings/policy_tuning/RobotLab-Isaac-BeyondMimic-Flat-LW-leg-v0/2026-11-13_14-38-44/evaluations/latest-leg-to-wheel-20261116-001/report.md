# 评估批次 latest-leg-to-wheel-20261116-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-11-13_14-38-44`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| nominal-start-a01 | completed | nominal-start，2000 / 1 / 42 | complete | [结果](<raw/nominal-start-a01/result.json>) / [日志](<raw/nominal-start-a01/console.log>) |
| randomized-start-a01 | completed | randomized-start，2000 / 1 / 42 | complete | [结果](<raw/randomized-start-a01/result.json>) / [日志](<raw/randomized-start-a01/console.log>) |
| randomized-phase-a01 | completed | randomized-phase，2000 / 1 / 42 | complete | [结果](<raw/randomized-phase-a01/result.json>) / [日志](<raw/randomized-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ebcf2534b8bcf45a4344c6a37868f083bd5fb3c24759c46774154974990a615e`；runner：`OnPolicyRunner`。
- [场景来源 scenario-bad2148a6254c4aeb2228efc78d2929f7026be696f893bab80e9058caaff0965.json](<../../provenance/scenario-bad2148a6254c4aeb2228efc78d2929f7026be696f893bab80e9058caaff0965.json>)
- [场景来源 scenario-f0f669b74f21c130b7a480a9097b756bab088e3c9284da1d1e9b734d277ea8b8.json](<../../provenance/scenario-f0f669b74f21c130b7a480a9097b756bab088e3c9284da1d1e9b734d277ea8b8.json>)
- [场景来源 scenario-f99c815d65296bdee96b712458dc01692e458f02a6838dd03c6744d08a645cf5.json](<../../provenance/scenario-f99c815d65296bdee96b712458dc01692e458f02a6838dd03c6744d08a645cf5.json>)
- [训练上下文 context-5eaa5eb6b13c67d0069c67d4f744654a3959fb4edf8ed08c0fab2391dd34c73c.json](<../../provenance/context-5eaa5eb6b13c67d0069c67d4f744654a3959fb4edf8ed08c0fab2391dd34c73c.json>)
- [训练有效配置 config-4e4884c54b98bee0e5420c6b0ff4fa8a21d08bda1043b5ab791a9e4c304ca7ea.json](<../../provenance/config-4e4884c54b98bee0e5420c6b0ff4fa8a21d08bda1043b5ab791a9e4c304ca7ea.json>)

- `nominal-start-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `nominal-start-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。
- `randomized-start-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-start-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。
- `randomized-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `randomized-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

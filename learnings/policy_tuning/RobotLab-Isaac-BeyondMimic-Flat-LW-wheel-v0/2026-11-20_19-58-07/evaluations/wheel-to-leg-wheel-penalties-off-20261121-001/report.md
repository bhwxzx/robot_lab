# 评估批次 wheel-to-leg-wheel-penalties-off-20261121-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-20_19-58-07`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`784c5a7acb2045c014edeb3e8370de1b5c3ac35acf979a9f28f8e52810bce249`；runner：`OnPolicyRunner`。
- [场景来源 scenario-6b40d2a8e5b92953bd188da171ad0930288001c462cefd72cfbc9f41120c1f07.json](<../../provenance/scenario-6b40d2a8e5b92953bd188da171ad0930288001c462cefd72cfbc9f41120c1f07.json>)
- [场景来源 scenario-77182903ab1ec7e4a789401a5ac386c07fe9e1d382312c2449366a1680ab77f3.json](<../../provenance/scenario-77182903ab1ec7e4a789401a5ac386c07fe9e1d382312c2449366a1680ab77f3.json>)
- [训练上下文 context-cefcd41a53e052e02c1e196f54568f90df6aea79374b0affb2788d31b3ef9b9d.json](<../../provenance/context-cefcd41a53e052e02c1e196f54568f90df6aea79374b0affb2788d31b3ef9b9d.json>)
- [训练有效配置 config-a033444d414e2021c13462c8e83d936292ab3e64b6aaee89907b051982fec620.json](<../../provenance/config-a033444d414e2021c13462c8e83d936292ab3e64b6aaee89907b051982fec620.json>)

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

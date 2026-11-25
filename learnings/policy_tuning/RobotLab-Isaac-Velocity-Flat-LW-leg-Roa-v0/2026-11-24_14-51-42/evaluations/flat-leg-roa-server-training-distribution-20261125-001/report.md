# 评估批次 flat-leg-roa-server-training-distribution-20261125-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-24_14-51-42`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| trained-randomization-32-a01 | completed | training-terrain-commands-pushes，2000 / 32 / 42 | complete | [结果](<raw/trained-randomization-32-a01/result.json>) / [日志](<raw/trained-randomization-32-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`dd24cbd6c1baf05e56645535f7b8ac4c2e0fffb8b3d5d3797383faaf34b61650`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-574477e0fc04304c34c9dd12b6aab190ced8fb4d03f37780a6dc1a230bf60b70.json](<../../provenance/scenario-574477e0fc04304c34c9dd12b6aab190ced8fb4d03f37780a6dc1a230bf60b70.json>)
- [训练上下文 context-46f1813cdaad954a30cec57e0d1e3379e01013506ca3f2a3d3ba9395819cf52d.json](<../../provenance/context-46f1813cdaad954a30cec57e0d1e3379e01013506ca3f2a3d3ba9395819cf52d.json>)
- [训练有效配置 config-4bae6b94de31c2d9886f91e98b1469651f9909fbfe916912c30e3dd34c6fbc40.json](<../../provenance/config-4bae6b94de31c2d9886f91e98b1469651f9909fbfe916912c30e3dd34c6fbc40.json>)

- `trained-randomization-32-a01` 指标：`{"max_joint_velocity_utilization": 3.3439483642578125, "max_tilt": 0.9990237355232239, "termination_rate": 3.125e-05, "tracking_xy_rmse": 0.2773345862548676, "tracking_yaw_rmse": 0.3429358134470485}`
- `trained-randomization-32-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.roa_mode": "student"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

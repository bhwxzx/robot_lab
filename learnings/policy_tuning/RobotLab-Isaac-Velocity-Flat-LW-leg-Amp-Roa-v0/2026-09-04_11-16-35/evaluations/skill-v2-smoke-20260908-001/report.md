# 评估批次 skill-v2-smoke-20260908-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-04_11-16-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| native-smoke-a01 | completed | skill-v2-forward-smoke，250 / 1 / 42 | complete | [结果](<raw/native-smoke-a01/result.json>) / [视频](<raw/native-smoke-a01/video.mp4>) / [日志](<raw/native-smoke-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`59ab5245acb17de8ea1b79e924c2f0189765be3c51c85f437b22b8e2423b0e19`；runner：`OnPolicyRunnerAmpROA`。

- `native-smoke-a01` 指标：`{"max_joint_velocity_utilization": 0.8183629989624024, "max_tilt": 0.18907295167446136, "termination_rate": 0.0, "tracking_xy_rmse": 0.2648798006241373, "tracking_yaw_rmse": 0.07551598304379306}`
- `native-smoke-a01` 命令调度：`[{"start_step": 0, "end_step": 249, "command": [0.4, 0.0, 0.0]}]`；训练配置覆盖：`{"events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

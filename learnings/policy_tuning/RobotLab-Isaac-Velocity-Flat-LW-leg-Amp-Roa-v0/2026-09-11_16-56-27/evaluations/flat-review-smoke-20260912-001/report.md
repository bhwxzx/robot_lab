# 评估批次 flat-review-smoke-20260912-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-11_16-56-27`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-camera-a01 | completed | stand-camera，150 / 1 / 42 | complete | [结果](<raw/stand-camera-a01/result.json>) / [视频](<raw/stand-camera-a01/video.mp4>) / [日志](<raw/stand-camera-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`cf0ae2f701c0b2ec15f5a43bfadb61425b879f68e425667a41d67a0b673d4d93`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-635d23b8cc15513dd863470f5b21285b69309685478d239b350012e5d69501dd.json](<../../provenance/scenario-635d23b8cc15513dd863470f5b21285b69309685478d239b350012e5d69501dd.json>)
- [训练上下文 context-81acb5359f1d1d0c02f1c56d9ffecd2ca49944775fde3d764257e5d17c073b3e.json](<../../provenance/context-81acb5359f1d1d0c02f1c56d9ffecd2ca49944775fde3d764257e5d17c073b3e.json>)
- [训练有效配置 config-7db9a64a6a06a89fe1ebe97abae3bf4d468aa6f8277752ade4caf4d198232053.json](<../../provenance/config-7db9a64a6a06a89fe1ebe97abae3bf4d468aa6f8277752ade4caf4d198232053.json>)

- `stand-camera-a01` 指标：`{"max_joint_velocity_utilization": 1.0761926651000977, "max_tilt": 0.1716337949037552, "termination_rate": 0.0, "tracking_xy_rmse": 0.14524370924430047, "tracking_yaw_rmse": 0.2691687298297107}`
- `stand-camera-a01` 命令调度：`[{"start_step": 0, "end_step": 149, "command": [0, 0, 0]}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

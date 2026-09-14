# 评估批次 turn-com-y-ablation-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-left-com-y-zero-offset-a01 | completed | turn-left-com-y-zero-offset，1000 / 1 / 42 | complete | [结果](<raw/turn-left-com-y-zero-offset-a01/result.json>) / [视频](<raw/turn-left-com-y-zero-offset-a01/video.mp4>) / [日志](<raw/turn-left-com-y-zero-offset-a01/console.log>) |
| turn-right-com-y-zero-offset-a01 | completed | turn-right-com-y-zero-offset，1000 / 1 / 42 | complete | [结果](<raw/turn-right-com-y-zero-offset-a01/result.json>) / [视频](<raw/turn-right-com-y-zero-offset-a01/video.mp4>) / [日志](<raw/turn-right-com-y-zero-offset-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-5054c50b8d8675da1a337e3d8da4e1c087333b58f2026df6757a87b45f7c557c.json](<../../provenance/scenario-5054c50b8d8675da1a337e3d8da4e1c087333b58f2026df6757a87b45f7c557c.json>)
- [场景来源 scenario-660dc3fd880e84b7f3c2227296cf15f8e04a91c546b552e478d0dcf903560f01.json](<../../provenance/scenario-660dc3fd880e84b7f3c2227296cf15f8e04a91c546b552e478d0dcf903560f01.json>)
- [训练上下文 context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json](<../../provenance/context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `turn-left-com-y-zero-offset-a01` 指标：`{"max_joint_velocity_utilization": 0.9976666450500489, "max_tilt": 0.11497603356838226, "termination_rate": 0.0, "tracking_xy_rmse": 0.07053329228871578, "tracking_yaw_rmse": 0.23625183558710042}`
- `turn-left-com-y-zero-offset-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [-0.075, 0.075], "y": [0.0, 0.0], "z": [-0.075, 0.075]}, "events.randomize_push_robot": null}`。
- `turn-right-com-y-zero-offset-a01` 指标：`{"max_joint_velocity_utilization": 0.9976666450500489, "max_tilt": 0.12349623441696167, "termination_rate": 0.0, "tracking_xy_rmse": 0.08049816592268094, "tracking_yaw_rmse": 0.20879224665220045}`
- `turn-right-com-y-zero-offset-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [-0.075, 0.075], "y": [0.0, 0.0], "z": [-0.075, 0.075]}, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

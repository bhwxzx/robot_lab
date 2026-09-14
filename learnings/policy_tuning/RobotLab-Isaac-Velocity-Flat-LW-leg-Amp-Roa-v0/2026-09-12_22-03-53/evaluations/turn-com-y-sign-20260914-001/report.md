# 评估批次 turn-com-y-sign-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-left-com-y-minus025-a01 | completed | turn-left-com-y-minus025，1000 / 1 / 42 | complete | [结果](<raw/turn-left-com-y-minus025-a01/result.json>) / [视频](<raw/turn-left-com-y-minus025-a01/video.mp4>) / [日志](<raw/turn-left-com-y-minus025-a01/console.log>) |
| turn-right-com-y-minus025-a01 | completed | turn-right-com-y-minus025，1000 / 1 / 42 | complete | [结果](<raw/turn-right-com-y-minus025-a01/result.json>) / [视频](<raw/turn-right-com-y-minus025-a01/video.mp4>) / [日志](<raw/turn-right-com-y-minus025-a01/console.log>) |
| turn-left-com-y-plus025-a01 | completed | turn-left-com-y-plus025，1000 / 1 / 42 | complete | [结果](<raw/turn-left-com-y-plus025-a01/result.json>) / [视频](<raw/turn-left-com-y-plus025-a01/video.mp4>) / [日志](<raw/turn-left-com-y-plus025-a01/console.log>) |
| turn-right-com-y-plus025-a01 | completed | turn-right-com-y-plus025，1000 / 1 / 42 | complete | [结果](<raw/turn-right-com-y-plus025-a01/result.json>) / [视频](<raw/turn-right-com-y-plus025-a01/video.mp4>) / [日志](<raw/turn-right-com-y-plus025-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-1db2a918d608447b4f474f46d6482712d05baae333913881adc8d6eafc4b7cb8.json](<../../provenance/scenario-1db2a918d608447b4f474f46d6482712d05baae333913881adc8d6eafc4b7cb8.json>)
- [场景来源 scenario-5af62d9f6f010bb8a88709d35355d51e85a50cb5598b0c0acd635c5f804abd57.json](<../../provenance/scenario-5af62d9f6f010bb8a88709d35355d51e85a50cb5598b0c0acd635c5f804abd57.json>)
- [场景来源 scenario-7b03bfef85c0174d24fd9c54de60b4a0c22b650691ab4b63e7af78d5701e306e.json](<../../provenance/scenario-7b03bfef85c0174d24fd9c54de60b4a0c22b650691ab4b63e7af78d5701e306e.json>)
- [场景来源 scenario-a32a82aad12f010cc2851580f0e3405d23427fc18a6fb25be75f7a6757561f91.json](<../../provenance/scenario-a32a82aad12f010cc2851580f0e3405d23427fc18a6fb25be75f7a6757561f91.json>)
- [训练上下文 context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json](<../../provenance/context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `turn-left-com-y-minus025-a01` 指标：`{"max_joint_velocity_utilization": 0.9961872100830078, "max_tilt": 0.11939031630754471, "termination_rate": 0.0, "tracking_xy_rmse": 0.0709332091584028, "tracking_yaw_rmse": 0.2436823991048782}`
- `turn-left-com-y-minus025-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [-0.075, 0.075], "y": [-0.025, -0.025], "z": [-0.075, 0.075]}, "events.randomize_push_robot": null}`。
- `turn-right-com-y-minus025-a01` 指标：`{"max_joint_velocity_utilization": 0.9961872100830078, "max_tilt": 0.11837133020162582, "termination_rate": 0.0, "tracking_xy_rmse": 0.0747241132906195, "tracking_yaw_rmse": 0.20004819084463293}`
- `turn-right-com-y-minus025-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [-0.075, 0.075], "y": [-0.025, -0.025], "z": [-0.075, 0.075]}, "events.randomize_push_robot": null}`。
- `turn-left-com-y-plus025-a01` 指标：`{"max_joint_velocity_utilization": 1.0074148178100586, "max_tilt": 0.10042887181043625, "termination_rate": 0.0, "tracking_xy_rmse": 0.07037364351431397, "tracking_yaw_rmse": 0.24195950619225892}`
- `turn-left-com-y-plus025-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [-0.075, 0.075], "y": [0.025, 0.025], "z": [-0.075, 0.075]}, "events.randomize_push_robot": null}`。
- `turn-right-com-y-plus025-a01` 指标：`{"max_joint_velocity_utilization": 1.0131382942199707, "max_tilt": 0.14344891905784607, "termination_rate": 0.0, "tracking_xy_rmse": 0.07693033726704146, "tracking_yaw_rmse": 0.20321699611521102}`
- `turn-right-com-y-plus025-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_com_positions.params.com_range": {"x": [-0.075, 0.075], "y": [0.025, 0.025], "z": [-0.075, 0.075]}, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

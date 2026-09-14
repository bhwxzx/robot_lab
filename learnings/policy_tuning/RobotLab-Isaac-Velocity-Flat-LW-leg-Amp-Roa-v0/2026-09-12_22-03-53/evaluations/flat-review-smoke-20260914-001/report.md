# 评估批次 flat-review-smoke-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-camera-a01 | completed | stand-camera，150 / 1 / 42 | complete | [结果](<raw/stand-camera-a01/result.json>) / [视频](<raw/stand-camera-a01/video.mp4>) / [日志](<raw/stand-camera-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-68b4d42e87ab5d090673cd08d0300d27095f1b4993b8e00ce0d4ddea6fb1d698.json](<../../provenance/scenario-68b4d42e87ab5d090673cd08d0300d27095f1b4993b8e00ce0d4ddea6fb1d698.json>)
- [训练上下文 context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json](<../../provenance/context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `stand-camera-a01` 指标：`{"max_joint_velocity_utilization": 1.0100210189819336, "max_tilt": 0.08422399312257767, "termination_rate": 0.0, "tracking_xy_rmse": 0.1037588395611784, "tracking_yaw_rmse": 0.05982531417846563}`
- `stand-camera-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 149, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

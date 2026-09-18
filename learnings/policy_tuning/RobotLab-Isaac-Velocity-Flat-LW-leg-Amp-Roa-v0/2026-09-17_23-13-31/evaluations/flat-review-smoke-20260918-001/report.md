# 评估批次 flat-review-smoke-20260918-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-17_23-13-31`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-camera-a01 | completed | stand-camera，150 / 1 / 42 | complete | [结果](<raw/stand-camera-a01/result.json>) / [视频](<raw/stand-camera-a01/video.mp4>) / [日志](<raw/stand-camera-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ac3ade5cb64737ceb3daf8f7253476b533fa3309d86164ebcf9f4caeb6e331ab`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-651e0f74ede52fe3040d39bddda618d92c1509441db3ec7410b561c3bc8b3215.json](<../../provenance/scenario-651e0f74ede52fe3040d39bddda618d92c1509441db3ec7410b561c3bc8b3215.json>)
- [训练上下文 context-3e2b1131a0506080679af2bf9a632fbcbcd31d87f9630dc424066d02446720a2.json](<../../provenance/context-3e2b1131a0506080679af2bf9a632fbcbcd31d87f9630dc424066d02446720a2.json>)
- [训练有效配置 config-ae9fa62a220fa36ed27740376d3bb85527a5563bc8c5a940a66b15ce667b9fda.json](<../../provenance/config-ae9fa62a220fa36ed27740376d3bb85527a5563bc8c5a940a66b15ce667b9fda.json>)

- `stand-camera-a01` 指标：`{"max_joint_velocity_utilization": 0.49009881019592283, "max_tilt": 0.10077513009309769, "termination_rate": 0.0, "tracking_xy_rmse": 0.1348652797296661, "tracking_yaw_rmse": 0.09849512065826393}`
- `stand-camera-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 149, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

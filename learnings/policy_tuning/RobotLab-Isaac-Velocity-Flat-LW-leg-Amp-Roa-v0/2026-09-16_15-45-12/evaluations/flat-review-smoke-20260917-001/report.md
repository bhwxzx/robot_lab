# 评估批次 flat-review-smoke-20260917-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-16_15-45-12`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-camera-a01 | completed | stand-camera，150 / 1 / 42 | complete | [结果](<raw/stand-camera-a01/result.json>) / [视频](<raw/stand-camera-a01/video.mp4>) / [日志](<raw/stand-camera-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`f8956bb1f9b43bdc638a52dbaf0a3301b022a4f967b581f2ed40663eb797d16a`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-871a833d8c720f7483cc0db544a7e363ccc7355bc2a476920d52ca57c29df188.json](<../../provenance/scenario-871a833d8c720f7483cc0db544a7e363ccc7355bc2a476920d52ca57c29df188.json>)
- [训练上下文 context-382b5ee84e17093ab3736ad910bff68706c9048456b9de6d2520f55513184e3a.json](<../../provenance/context-382b5ee84e17093ab3736ad910bff68706c9048456b9de6d2520f55513184e3a.json>)
- [训练有效配置 config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json](<../../provenance/config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json>)

- `stand-camera-a01` 指标：`{"max_joint_velocity_utilization": 0.7051561355590821, "max_tilt": 0.0757775828242302, "termination_rate": 0.0, "tracking_xy_rmse": 0.11733194231235256, "tracking_yaw_rmse": 0.08257863385724817}`
- `stand-camera-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 149, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

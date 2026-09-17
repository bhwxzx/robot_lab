# 评估批次 matched-plane-20260917-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-16_15-45-12`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [日志](<raw/turn-right-a01/console.log>) |
| moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-a01/result.json>) / [日志](<raw/moving-turn-a01/console.log>) |
| moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-right-a01/result.json>) / [日志](<raw/moving-turn-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`f8956bb1f9b43bdc638a52dbaf0a3301b022a4f967b581f2ed40663eb797d16a`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-818b4242ca8ba3114cb9fddd50c9bd12515f5438657d8fc386939efc40706870.json](<../../provenance/scenario-818b4242ca8ba3114cb9fddd50c9bd12515f5438657d8fc386939efc40706870.json>)
- [场景来源 scenario-b8825558523408d767392fc13d5a67945b8b3c0579bf8b5a354056dee2aa7d70.json](<../../provenance/scenario-b8825558523408d767392fc13d5a67945b8b3c0579bf8b5a354056dee2aa7d70.json>)
- [场景来源 scenario-d8b3ebbac0e204a3fa5e962d6945427a90aa5fa301aaa0c7db25dfdbf6924ec0.json](<../../provenance/scenario-d8b3ebbac0e204a3fa5e962d6945427a90aa5fa301aaa0c7db25dfdbf6924ec0.json>)
- [训练上下文 context-382b5ee84e17093ab3736ad910bff68706c9048456b9de6d2520f55513184e3a.json](<../../provenance/context-382b5ee84e17093ab3736ad910bff68706c9048456b9de6d2520f55513184e3a.json>)
- [训练有效配置 config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json](<../../provenance/config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json>)

- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9991875648498535, "max_tilt": 0.16323137283325195, "termination_rate": 0.0, "tracking_xy_rmse": 0.07330808600269538, "tracking_yaw_rmse": 0.31113644714044714}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.1428607940673827, "max_tilt": 0.22102390229701996, "termination_rate": 0.0, "tracking_xy_rmse": 0.16845860880395996, "tracking_yaw_rmse": 0.3533195764190554}`
- `moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.5107056617736816, "max_tilt": 0.1936391443014145, "termination_rate": 0.0, "tracking_xy_rmse": 0.14564678165816497, "tracking_yaw_rmse": 0.34032158842736016}`
- `moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

# 评估批次 matched-plane-20260918-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-17_23-13-31`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [日志](<raw/turn-right-a01/console.log>) |
| moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-a01/result.json>) / [日志](<raw/moving-turn-a01/console.log>) |
| moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-right-a01/result.json>) / [日志](<raw/moving-turn-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ac3ade5cb64737ceb3daf8f7253476b533fa3309d86164ebcf9f4caeb6e331ab`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-4c3989e753042c4238f705851976b1acc121ffb818d5bdf476ef51493a93b121.json](<../../provenance/scenario-4c3989e753042c4238f705851976b1acc121ffb818d5bdf476ef51493a93b121.json>)
- [场景来源 scenario-9b44b559773b25bab9d13cbac1409fabd7ed99db0290ae997848b63a8c1dc2db.json](<../../provenance/scenario-9b44b559773b25bab9d13cbac1409fabd7ed99db0290ae997848b63a8c1dc2db.json>)
- [场景来源 scenario-d78030c2bd59269c2c5dc0c4392410b3396c878378f5a793ca7eb58472939800.json](<../../provenance/scenario-d78030c2bd59269c2c5dc0c4392410b3396c878378f5a793ca7eb58472939800.json>)
- [训练上下文 context-3e2b1131a0506080679af2bf9a632fbcbcd31d87f9630dc424066d02446720a2.json](<../../provenance/context-3e2b1131a0506080679af2bf9a632fbcbcd31d87f9630dc424066d02446720a2.json>)
- [训练有效配置 config-ae9fa62a220fa36ed27740376d3bb85527a5563bc8c5a940a66b15ce667b9fda.json](<../../provenance/config-ae9fa62a220fa36ed27740376d3bb85527a5563bc8c5a940a66b15ce667b9fda.json>)

- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0005787849426269, "max_tilt": 0.18386679887771606, "termination_rate": 0.0, "tracking_xy_rmse": 0.08306029534365478, "tracking_yaw_rmse": 0.21625933685524118}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.4717296600341796, "max_tilt": 0.2259231060743332, "termination_rate": 0.0, "tracking_xy_rmse": 0.16760442017035904, "tracking_yaw_rmse": 0.32800553628463497}`
- `moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.912691879272461, "max_tilt": 0.2110438346862793, "termination_rate": 0.0, "tracking_xy_rmse": 0.16358122377181453, "tracking_yaw_rmse": 0.33868767841285646}`
- `moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

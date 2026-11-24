# 评估批次 flat-leg-roa-34000-plane-20261124-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-23_19-36-15`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，1000 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [日志](<raw/stand-a01/console.log>) |
| forward-a01 | completed | forward，1000 / 1 / 42 | complete | [结果](<raw/forward-a01/result.json>) / [日志](<raw/forward-a01/console.log>) |
| backward-a01 | completed | backward，1000 / 1 / 42 | complete | [结果](<raw/backward-a01/result.json>) / [日志](<raw/backward-a01/console.log>) |
| turn-left-a01 | completed | turn-left，1000 / 1 / 42 | complete | [结果](<raw/turn-left-a01/result.json>) / [日志](<raw/turn-left-a01/console.log>) |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [日志](<raw/turn-right-a01/console.log>) |
| forward-left-a01 | completed | forward-left，1000 / 1 / 42 | complete | [结果](<raw/forward-left-a01/result.json>) / [日志](<raw/forward-left-a01/console.log>) |
| forward-right-a01 | completed | forward-right，1000 / 1 / 42 | complete | [结果](<raw/forward-right-a01/result.json>) / [日志](<raw/forward-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ca0269c71f3b983efb7fb8f3643d037c18d36b44afa557da7f09cff4d682f217`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-299c573c2ca9c18b2b4dea3d4a3d424d13fdb9fa4364451bc2f19f1c73c0f010.json](<../../provenance/scenario-299c573c2ca9c18b2b4dea3d4a3d424d13fdb9fa4364451bc2f19f1c73c0f010.json>)
- [场景来源 scenario-861c71f00bf59ab49ccf6903b2f8b007f6b1c4640a99bcc040610dbef43084c9.json](<../../provenance/scenario-861c71f00bf59ab49ccf6903b2f8b007f6b1c4640a99bcc040610dbef43084c9.json>)
- [场景来源 scenario-a6813d8d631967f86ee05b2fc9d7f8d432cffb088fafa00b25dc499e69a83c8a.json](<../../provenance/scenario-a6813d8d631967f86ee05b2fc9d7f8d432cffb088fafa00b25dc499e69a83c8a.json>)
- [场景来源 scenario-bb650a3ca76c56f855ca64adec284b124ddebff9e0aa510302078394a98e219a.json](<../../provenance/scenario-bb650a3ca76c56f855ca64adec284b124ddebff9e0aa510302078394a98e219a.json>)
- [场景来源 scenario-e47a764050568d36e98beaf36440d0ee283ce462bdaaa3cb0bcd88fc664811fd.json](<../../provenance/scenario-e47a764050568d36e98beaf36440d0ee283ce462bdaaa3cb0bcd88fc664811fd.json>)
- [场景来源 scenario-f3a9d0f154ddb41ee50ad8a6538fcea0cf2622baf46f7b8690618b9a06f31232.json](<../../provenance/scenario-f3a9d0f154ddb41ee50ad8a6538fcea0cf2622baf46f7b8690618b9a06f31232.json>)
- [场景来源 scenario-f65dcb06dcbb674f02209ecfa5dbf19bffe0a4d9e0cff6fe9231ed4b80191def.json](<../../provenance/scenario-f65dcb06dcbb674f02209ecfa5dbf19bffe0a4d9e0cff6fe9231ed4b80191def.json>)
- [训练上下文 context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json](<../../provenance/context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json>)
- [训练有效配置 config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json](<../../provenance/config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 1.0000004768371582, "max_tilt": 0.08907771855592728, "termination_rate": 0.0, "tracking_xy_rmse": 0.05458778297655647, "tracking_yaw_rmse": 0.06191388832880261}`
- `stand-a01` 命令调度：`[{"start_step": 0, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-a01` 指标：`{"max_joint_velocity_utilization": 1.139384651184082, "max_tilt": 0.10784410685300827, "termination_rate": 0.0, "tracking_xy_rmse": 0.11405443768811184, "tracking_yaw_rmse": 0.09073530303214206}`
- `forward-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0.4, 0, 0]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `backward-a01` 指标：`{"max_joint_velocity_utilization": 1.0000004768371582, "max_tilt": 0.11869847029447556, "termination_rate": 0.0, "tracking_xy_rmse": 0.13097929610402642, "tracking_yaw_rmse": 0.09614688850996059}`
- `backward-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [-0.4, 0, 0]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 1.0000004768371582, "max_tilt": 0.10270047187805176, "termination_rate": 0.0, "tracking_xy_rmse": 0.0764534839427952, "tracking_yaw_rmse": 0.1994257190898011}`
- `turn-left-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0, 0, 0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.2833826065063476, "max_tilt": 0.1982387900352478, "termination_rate": 0.0, "tracking_xy_rmse": 0.07492833938435413, "tracking_yaw_rmse": 0.2936123549508984}`
- `turn-right-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0, 0, -0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-left-a01` 指标：`{"max_joint_velocity_utilization": 1.4676122665405273, "max_tilt": 0.1464247703552246, "termination_rate": 0.0, "tracking_xy_rmse": 0.11382481199452844, "tracking_yaw_rmse": 0.3001432677708735}`
- `forward-left-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0.4, 0, 0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0297574043273925, "max_tilt": 0.13600437343120575, "termination_rate": 0.0, "tracking_xy_rmse": 0.11121529166281514, "tracking_yaw_rmse": 0.23921649796283492}`
- `forward-right-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 749, "command": [0.4, 0, -0.5]}, {"start_step": 750, "end_step": 999, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

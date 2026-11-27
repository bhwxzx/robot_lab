# 评估批次 plane-speed-tracking-50000-20261127-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-52-54`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| backward-a01 | completed | backward，1400 / 1 / 42 | complete | [结果](<raw/backward-a01/result.json>) / [日志](<raw/backward-a01/console.log>) |
| forward-a01 | completed | forward，1400 / 1 / 42 | complete | [结果](<raw/forward-a01/result.json>) / [日志](<raw/forward-a01/console.log>) |
| pure-left-a01 | completed | pure-left，1400 / 1 / 42 | complete | [结果](<raw/pure-left-a01/result.json>) / [日志](<raw/pure-left-a01/console.log>) |
| pure-right-a01 | completed | pure-right，1400 / 1 / 42 | complete | [结果](<raw/pure-right-a01/result.json>) / [日志](<raw/pure-right-a01/console.log>) |
| forward-left-a01 | completed | forward-left，1400 / 1 / 42 | complete | [结果](<raw/forward-left-a01/result.json>) / [日志](<raw/forward-left-a01/console.log>) |
| forward-right-a01 | completed | forward-right，1400 / 1 / 42 | complete | [结果](<raw/forward-right-a01/result.json>) / [日志](<raw/forward-right-a01/console.log>) |
| backward-left-a01 | completed | backward-left，1400 / 1 / 42 | complete | [结果](<raw/backward-left-a01/result.json>) / [日志](<raw/backward-left-a01/console.log>) |
| backward-right-a01 | completed | backward-right，1400 / 1 / 42 | complete | [结果](<raw/backward-right-a01/result.json>) / [日志](<raw/backward-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c349e6b54ec7500f32633f0ad6ba09b9f327d012f2660b0da6c1648398a70a28`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-3ff61bfa2e25a3bee3b407e37774b652edd501ea80918c45a89cfbad3e1a86a6.json](<../../provenance/scenario-3ff61bfa2e25a3bee3b407e37774b652edd501ea80918c45a89cfbad3e1a86a6.json>)
- [场景来源 scenario-423d349201847b87cfb18b25f85002e0243dc38dd9b32982d9f9f4f725f5c67a.json](<../../provenance/scenario-423d349201847b87cfb18b25f85002e0243dc38dd9b32982d9f9f4f725f5c67a.json>)
- [场景来源 scenario-56e189500cd93e02506fe956386356b0a41a6c6596d60c66649250cb6f809e0e.json](<../../provenance/scenario-56e189500cd93e02506fe956386356b0a41a6c6596d60c66649250cb6f809e0e.json>)
- [场景来源 scenario-5e8b214150d391627a5927448c543ed9751b70206881f2c70d182cbaa0cf0613.json](<../../provenance/scenario-5e8b214150d391627a5927448c543ed9751b70206881f2c70d182cbaa0cf0613.json>)
- [场景来源 scenario-6a3765a3fd9b02ccd19da5d7f604522479f7c6609453abec90ecb682b935f850.json](<../../provenance/scenario-6a3765a3fd9b02ccd19da5d7f604522479f7c6609453abec90ecb682b935f850.json>)
- [场景来源 scenario-6a3789f0a37d4b9871f42bd262399cab698585c3b849c1dd3cc4f00a53daa71a.json](<../../provenance/scenario-6a3789f0a37d4b9871f42bd262399cab698585c3b849c1dd3cc4f00a53daa71a.json>)
- [场景来源 scenario-ebea73241a921e12c32cb783e5ab380bd7758864a523781e2225a05a2dc2b6cd.json](<../../provenance/scenario-ebea73241a921e12c32cb783e5ab380bd7758864a523781e2225a05a2dc2b6cd.json>)
- [场景来源 scenario-ff7a47c71445e69d0c7144f1ebed37b7ce39eb9d08c6690006c182058d776b49.json](<../../provenance/scenario-ff7a47c71445e69d0c7144f1ebed37b7ce39eb9d08c6690006c182058d776b49.json>)
- [训练上下文 context-2d3f38dd571a365c668e6ed8ea9c4a7b12e9708f69ba80ed9de29cebf8e5b0be.json](<../../provenance/context-2d3f38dd571a365c668e6ed8ea9c4a7b12e9708f69ba80ed9de29cebf8e5b0be.json>)
- [训练有效配置 config-1d6789705ef276a862d4a55fa53e9467862a570176e4049c27016ba36052c981.json](<../../provenance/config-1d6789705ef276a862d4a55fa53e9467862a570176e4049c27016ba36052c981.json>)

- `backward-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.15236017107963562, "termination_rate": 0.0, "tracking_xy_rmse": 0.2575791980976706, "tracking_yaw_rmse": 0.08525587070198613}`
- `backward-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [-0.2, 0, 0]}, {"start_step": 500, "end_step": 899, "command": [-0.4, 0, 0]}, {"start_step": 900, "end_step": 1299, "command": [-0.6, 0, 0]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.132972851395607, "termination_rate": 0.0, "tracking_xy_rmse": 0.23771178137411855, "tracking_yaw_rmse": 0.08221495152144194}`
- `forward-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [0.2, 0, 0]}, {"start_step": 500, "end_step": 899, "command": [0.4, 0, 0]}, {"start_step": 900, "end_step": 1299, "command": [0.6, 0, 0]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `pure-left-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.1270379275083542, "termination_rate": 0.0, "tracking_xy_rmse": 0.23486552901073635, "tracking_yaw_rmse": 0.2494990584177093}`
- `pure-left-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [0, 0, 0.15]}, {"start_step": 500, "end_step": 899, "command": [0, 0, 0.25]}, {"start_step": 900, "end_step": 1299, "command": [0, 0, 0.35]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `pure-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.1270379275083542, "termination_rate": 0.0, "tracking_xy_rmse": 0.23276624072845228, "tracking_yaw_rmse": 0.2682162933721206}`
- `pure-right-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [0, 0, -0.15]}, {"start_step": 500, "end_step": 899, "command": [0, 0, -0.25]}, {"start_step": 900, "end_step": 1299, "command": [0, 0, -0.35]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-left-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.2695111632347107, "termination_rate": 0.0, "tracking_xy_rmse": 0.23941373408732758, "tracking_yaw_rmse": 0.2580800086570319}`
- `forward-left-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [0.4, 0, 0.15]}, {"start_step": 500, "end_step": 899, "command": [0.4, 0, 0.25]}, {"start_step": 900, "end_step": 1299, "command": [0.4, 0, 0.35]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-right-a01` 指标：`{"max_joint_velocity_utilization": 1.1709480285644531, "max_tilt": 0.1270379275083542, "termination_rate": 0.0, "tracking_xy_rmse": 0.23221340159170725, "tracking_yaw_rmse": 0.2652317272387657}`
- `forward-right-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [0.4, 0, -0.15]}, {"start_step": 500, "end_step": 899, "command": [0.4, 0, -0.25]}, {"start_step": 900, "end_step": 1299, "command": [0.4, 0, -0.35]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `backward-left-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.1270379275083542, "termination_rate": 0.0, "tracking_xy_rmse": 0.2562150558638756, "tracking_yaw_rmse": 0.2429956523332534}`
- `backward-left-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [-0.4, 0, 0.15]}, {"start_step": 500, "end_step": 899, "command": [-0.4, 0, 0.25]}, {"start_step": 900, "end_step": 1299, "command": [-0.4, 0, 0.35]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `backward-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0962221145629882, "max_tilt": 0.14967365562915802, "termination_rate": 0.0, "tracking_xy_rmse": 0.25256225658881765, "tracking_yaw_rmse": 0.2624577461695059}`
- `backward-right-a01` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [-0.4, 0, -0.15]}, {"start_step": 500, "end_step": 899, "command": [-0.4, 0, -0.25]}, {"start_step": 900, "end_step": 1299, "command": [-0.4, 0, -0.35]}, {"start_step": 1300, "end_step": 1399, "command": [0, 0, 0]}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

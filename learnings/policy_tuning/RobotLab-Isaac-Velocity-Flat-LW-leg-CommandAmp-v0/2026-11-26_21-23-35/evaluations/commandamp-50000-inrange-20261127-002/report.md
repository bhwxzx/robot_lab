# 评估批次 commandamp-50000-inrange-20261127-002

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| turn-inrange-a02 | completed | turn-inrange，960 / 16 / 42 | complete | [结果](<raw/turn-inrange-a02/result.json>) / [日志](<raw/turn-inrange-a02/console.log>) |
| hold-stand-a02 | completed | hold-stand，700 / 16 / 42 | complete | [结果](<raw/hold-stand-a02/result.json>) / [日志](<raw/hold-stand-a02/console.log>) |
| hold-left-a02 | completed | hold-left，700 / 16 / 42 | complete | [结果](<raw/hold-left-a02/result.json>) / [日志](<raw/hold-left-a02/console.log>) |
| hold-right-a02 | completed | hold-right，700 / 16 / 42 | complete | [结果](<raw/hold-right-a02/result.json>) / [日志](<raw/hold-right-a02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-315686eec97294206886089039c2965e91ace5ed51b6f771d8c4426d452c20c9.json](<../../provenance/scenario-315686eec97294206886089039c2965e91ace5ed51b6f771d8c4426d452c20c9.json>)
- [场景来源 scenario-9c0787902291c7cf423e9743dde1053738ab7ebdde0adcbce181be565216d500.json](<../../provenance/scenario-9c0787902291c7cf423e9743dde1053738ab7ebdde0adcbce181be565216d500.json>)
- [场景来源 scenario-a7d4b3f6adaf57e6e54514d842e0bef5096686c627c7990f932d6ff875dab9b4.json](<../../provenance/scenario-a7d4b3f6adaf57e6e54514d842e0bef5096686c627c7990f932d6ff875dab9b4.json>)
- [场景来源 scenario-de4c7d6568ee73c3382309390c9144e17d685ea0707b01227444be8fabffda94.json](<../../provenance/scenario-de4c7d6568ee73c3382309390c9144e17d685ea0707b01227444be8fabffda94.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `turn-inrange-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.2095124872654573, "tracking_yaw_rmse": 0.2720334647325634}`
- `turn-inrange-a02` 命令调度：`[{"start_step": 0, "end_step": 159, "command": [0, 0, 0]}, {"start_step": 160, "end_step": 319, "command": [0, 0, 0.35]}, {"start_step": 320, "end_step": 479, "command": [0, 0, -0.35]}, {"start_step": 480, "end_step": 639, "command": [0.4, 0, 0.35]}, {"start_step": 640, "end_step": 799, "command": [0.4, 0, -0.35]}, {"start_step": 800, "end_step": 959, "command": [0, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-stand-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.05568058214501297, "tracking_yaw_rmse": 0.06504584778359235}`
- `hold-stand-a02` 命令调度：`[{"start_step": 0, "end_step": 699, "command": [0, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-left-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.2358314506947755, "tracking_yaw_rmse": 0.30804655098704325}`
- `hold-left-a02` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 699, "command": [0, 0, 0.35]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-right-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.2358535962585587, "tracking_yaw_rmse": 0.30249685053185055}`
- `hold-right-a02` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 699, "command": [0, 0, -0.35]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

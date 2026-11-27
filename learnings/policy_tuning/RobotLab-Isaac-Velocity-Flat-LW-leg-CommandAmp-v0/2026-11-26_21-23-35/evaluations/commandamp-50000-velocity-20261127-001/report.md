# 评估批次 commandamp-50000-velocity-20261127-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| straight-a01 | completed | straight，960 / 16 / 42 | complete | [结果](<raw/straight-a01/result.json>) / [日志](<raw/straight-a01/console.log>) |
| turn-arc-a01 | completed | turn-arc，960 / 16 / 42 | complete | [结果](<raw/turn-arc-a01/result.json>) / [日志](<raw/turn-arc-a01/console.log>) |
| stand-boundary-a01 | completed | stand-boundary，960 / 16 / 42 | complete | [结果](<raw/stand-boundary-a01/result.json>) / [日志](<raw/stand-boundary-a01/console.log>) |
| straight-training-a01 | completed | straight-training，960 / 16 / 42 | complete | [结果](<raw/straight-training-a01/result.json>) / [日志](<raw/straight-training-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-7496084bf093f403caa64ac5af5c81eb1ab91e49505c1bff82d04c46c2d89564.json](<../../provenance/scenario-7496084bf093f403caa64ac5af5c81eb1ab91e49505c1bff82d04c46c2d89564.json>)
- [场景来源 scenario-7d89eae6dae6902f409a1b2a618b58a31a209330a9645c1240dd100e21d01da0.json](<../../provenance/scenario-7d89eae6dae6902f409a1b2a618b58a31a209330a9645c1240dd100e21d01da0.json>)
- [场景来源 scenario-8b55a688d926dd35ab2f7344dd26ef98c3ef5d9bea0e16c09a070575f51dd3c5.json](<../../provenance/scenario-8b55a688d926dd35ab2f7344dd26ef98c3ef5d9bea0e16c09a070575f51dd3c5.json>)
- [场景来源 scenario-9713845ac5d96f31efc5a2ab93592f32eeabb6a751e080cdc3369569b2a3781a.json](<../../provenance/scenario-9713845ac5d96f31efc5a2ab93592f32eeabb6a751e080cdc3369569b2a3781a.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `straight-a01` 指标：`{"max_joint_velocity_utilization": 1.4040807723999023, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.23866226044483452, "tracking_yaw_rmse": 0.1337179698862979}`
- `straight-a01` 命令调度：`[{"start_step": 0, "end_step": 159, "command": [0, 0, 0]}, {"start_step": 160, "end_step": 319, "command": [0.4, 0, 0]}, {"start_step": 320, "end_step": 479, "command": [0.8, 0, 0]}, {"start_step": 480, "end_step": 639, "command": [0, 0, 0]}, {"start_step": 640, "end_step": 799, "command": [-0.4, 0, 0]}, {"start_step": 800, "end_step": 959, "command": [-0.8, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `turn-arc-a01` 指标：`{"max_joint_velocity_utilization": 1.3884784698486328, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.20935752264093233, "tracking_yaw_rmse": 0.3248022869700308}`
- `turn-arc-a01` 命令调度：`[{"start_step": 0, "end_step": 159, "command": [0, 0, 0]}, {"start_step": 160, "end_step": 319, "command": [0, 0, 0.5]}, {"start_step": 320, "end_step": 479, "command": [0, 0, -0.5]}, {"start_step": 480, "end_step": 639, "command": [0.4, 0, 0.35]}, {"start_step": 640, "end_step": 799, "command": [0.4, 0, -0.35]}, {"start_step": 800, "end_step": 959, "command": [0, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `stand-boundary-a01` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.15549172357729388, "tracking_yaw_rmse": 0.10189587696910887}`
- `stand-boundary-a01` 命令调度：`[{"start_step": 0, "end_step": 159, "command": [0, 0, 0]}, {"start_step": 160, "end_step": 319, "command": [0.1, 0, 0]}, {"start_step": 320, "end_step": 479, "command": [0.12, 0, 0]}, {"start_step": 480, "end_step": 639, "command": [0, 0, 0.1]}, {"start_step": 640, "end_step": 799, "command": [0, 0, 0.12]}, {"start_step": 800, "end_step": 959, "command": [0, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `straight-training-a01` 指标：`{"max_joint_velocity_utilization": 1.2904970169067382, "max_tilt": 0.33941224217414856, "termination_rate": 0.0, "tracking_xy_rmse": 0.2647174363124286, "tracking_yaw_rmse": 0.16880731509146157}`
- `straight-training-a01` 命令调度：`[{"start_step": 0, "end_step": 159, "command": [0, 0, 0]}, {"start_step": 160, "end_step": 319, "command": [0.4, 0, 0]}, {"start_step": 320, "end_step": 479, "command": [0.8, 0, 0]}, {"start_step": 480, "end_step": 639, "command": [0, 0, 0]}, {"start_step": 640, "end_step": 799, "command": [-0.4, 0, 0]}, {"start_step": 800, "end_step": 959, "command": [-0.8, 0, 0]}]`；训练配置覆盖：`{}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

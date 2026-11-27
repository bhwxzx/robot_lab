# 评估批次 commandamp-50000-speed-holds-20261127-003

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| hold-forward-a03 | completed | hold-forward，500 / 16 / 42 | complete | [结果](<raw/hold-forward-a03/result.json>) / [日志](<raw/hold-forward-a03/console.log>) |
| hold-reverse-a03 | completed | hold-reverse，500 / 16 / 42 | complete | [结果](<raw/hold-reverse-a03/result.json>) / [日志](<raw/hold-reverse-a03/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-381b0fe3f8de8974770538867c6dc223ee28e4225e159664625aa91de5e6d67a.json](<../../provenance/scenario-381b0fe3f8de8974770538867c6dc223ee28e4225e159664625aa91de5e6d67a.json>)
- [场景来源 scenario-c827e4416b15f2ce27ea9e2c7fed21da95c5f6c01aa0ad507af543b41df66ee3.json](<../../provenance/scenario-c827e4416b15f2ce27ea9e2c7fed21da95c5f6c01aa0ad507af543b41df66ee3.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `hold-forward-a03` 指标：`{"max_joint_velocity_utilization": 1.1436590194702148, "max_tilt": 0.2962498366832733, "termination_rate": 0.0, "tracking_xy_rmse": 0.24635692731855519, "tracking_yaw_rmse": 0.15376188756363124}`
- `hold-forward-a03` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [0.8, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-reverse-a03` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.29361125656938547, "tracking_yaw_rmse": 0.19170117053370633}`
- `hold-reverse-a03` 命令调度：`[{"start_step": 0, "end_step": 99, "command": [0, 0, 0]}, {"start_step": 100, "end_step": 499, "command": [-0.8, 0, 0]}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

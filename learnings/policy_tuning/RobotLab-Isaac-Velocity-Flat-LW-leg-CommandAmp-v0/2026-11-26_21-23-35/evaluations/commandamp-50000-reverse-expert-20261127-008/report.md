# 评估批次 commandamp-50000-reverse-expert-20261127-008

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| reverse-80-a08 | completed | reverse-80，500 / 16 / 42 | complete | [结果](<raw/reverse-80-a08/result.json>) / [日志](<raw/reverse-80-a08/console.log>) |
| forward-80-a08 | completed | forward-80，500 / 16 / 42 | complete | [结果](<raw/forward-80-a08/result.json>) / [日志](<raw/forward-80-a08/console.log>) |
| reverse-40-a08 | completed | reverse-40，500 / 16 / 42 | complete | [结果](<raw/reverse-40-a08/result.json>) / [日志](<raw/reverse-40-a08/console.log>) |
| forward-40-a08 | completed | forward-40，500 / 16 / 42 | complete | [结果](<raw/forward-40-a08/result.json>) / [日志](<raw/forward-40-a08/console.log>) |
| reverse-20-a08 | completed | reverse-20，500 / 16 / 42 | complete | [结果](<raw/reverse-20-a08/result.json>) / [日志](<raw/reverse-20-a08/console.log>) |
| forward-20-a08 | completed | forward-20，500 / 16 / 42 | complete | [结果](<raw/forward-20-a08/result.json>) / [日志](<raw/forward-20-a08/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-0d34ab28b15252de3f379c2c53a159b8853e78a3f72d40abdcd362d922b0c26e.json](<../../provenance/scenario-0d34ab28b15252de3f379c2c53a159b8853e78a3f72d40abdcd362d922b0c26e.json>)
- [场景来源 scenario-106fce4a3ebbdb9dc8a7fd8ceb894f91730040abe3a98832fe0a3e2f3049bb11.json](<../../provenance/scenario-106fce4a3ebbdb9dc8a7fd8ceb894f91730040abe3a98832fe0a3e2f3049bb11.json>)
- [场景来源 scenario-46b4ba077c645a65cb5611aaaebe200b1bd5f12fb7788d3f3cc6e57fb166004a.json](<../../provenance/scenario-46b4ba077c645a65cb5611aaaebe200b1bd5f12fb7788d3f3cc6e57fb166004a.json>)
- [场景来源 scenario-69f3768243fcaa04ad575620763e768ed6da30fc3540915b7f0c196b9da7cdb4.json](<../../provenance/scenario-69f3768243fcaa04ad575620763e768ed6da30fc3540915b7f0c196b9da7cdb4.json>)
- [场景来源 scenario-ae139141a487aada162bd208a337a5b98c431a31dc213843110e25d7dfef5a69.json](<../../provenance/scenario-ae139141a487aada162bd208a337a5b98c431a31dc213843110e25d7dfef5a69.json>)
- [场景来源 scenario-bd1d067f8c1790ef7bd98573373a4ea5c7752e6607314e9cc124ad3e868ce94b.json](<../../provenance/scenario-bd1d067f8c1790ef7bd98573373a4ea5c7752e6607314e9cc124ad3e868ce94b.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `reverse-80-a08` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.2974589433633728, "tracking_yaw_rmse": 0.20505446512640765}`
- `reverse-80-a08` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-80-a08` 指标：`{"max_joint_velocity_utilization": 1.2630085945129395, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.24614150726758532, "tracking_yaw_rmse": 0.13646985746561682}`
- `forward-80-a08` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-40-a08` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.25101628245775676, "tracking_yaw_rmse": 0.13693625012867192}`
- `reverse-40-a08` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-40-a08` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22137778657321489, "tracking_yaw_rmse": 0.10373881954404418}`
- `forward-40-a08` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-20-a08` 指标：`{"max_joint_velocity_utilization": 1.1645997047424317, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.23857248951466364, "tracking_yaw_rmse": 0.12779412161017786}`
- `reverse-20-a08` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.2, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-20-a08` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22323545444475376, "tracking_yaw_rmse": 0.10481432884715944}`
- `forward-20-a08` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.2, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。

## 观察与限制

- Paired randomizations/velocity schedules match longitudinal grid 007. Analysis window steps 200 through 499.
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [population-recorder.json](<population-recorder.json>)
- [manifest.json](<../commandamp-50000-longitudinal-grid-20261127-007/manifest.json>)
- [amp-contact-all-envs.npz](<raw/forward-20-a08/amp-contact-all-envs.npz>)
- [amp-contact-all-envs.npz](<raw/forward-40-a08/amp-contact-all-envs.npz>)
- [amp-contact-all-envs.npz](<raw/forward-80-a08/amp-contact-all-envs.npz>)
- [amp-contact-all-envs.npz](<raw/reverse-20-a08/amp-contact-all-envs.npz>)
- [amp-contact-all-envs.npz](<raw/reverse-40-a08/amp-contact-all-envs.npz>)
- [amp-contact-all-envs.npz](<raw/reverse-80-a08/amp-contact-all-envs.npz>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

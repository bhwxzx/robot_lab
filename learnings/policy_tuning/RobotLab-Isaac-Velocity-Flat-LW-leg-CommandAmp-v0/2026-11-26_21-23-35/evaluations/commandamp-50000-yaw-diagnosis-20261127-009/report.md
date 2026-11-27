# 评估批次 commandamp-50000-yaw-diagnosis-20261127-009

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| pure-left-35-a09 | completed | pure-left-35，500 / 16 / 42 | complete | [结果](<raw/pure-left-35-a09/result.json>) / [日志](<raw/pure-left-35-a09/console.log>) |
| pure-right-35-a09 | completed | pure-right-35，500 / 16 / 42 | complete | [结果](<raw/pure-right-35-a09/result.json>) / [日志](<raw/pure-right-35-a09/console.log>) |
| forward-left-35-a09 | completed | forward-left-35，500 / 16 / 42 | complete | [结果](<raw/forward-left-35-a09/result.json>) / [日志](<raw/forward-left-35-a09/console.log>) |
| forward-right-35-a09 | completed | forward-right-35，500 / 16 / 42 | complete | [结果](<raw/forward-right-35-a09/result.json>) / [日志](<raw/forward-right-35-a09/console.log>) |
| pure-left-25-a09 | completed | pure-left-25，500 / 16 / 42 | complete | [结果](<raw/pure-left-25-a09/result.json>) / [日志](<raw/pure-left-25-a09/console.log>) |
| pure-right-25-a09 | completed | pure-right-25，500 / 16 / 42 | complete | [结果](<raw/pure-right-25-a09/result.json>) / [日志](<raw/pure-right-25-a09/console.log>) |
| nominal-left-35-a09 | completed | nominal-left-35，500 / 16 / 42 | complete | [结果](<raw/nominal-left-35-a09/result.json>) / [日志](<raw/nominal-left-35-a09/console.log>) |
| nominal-right-35-a09 | completed | nominal-right-35，500 / 16 / 42 | complete | [结果](<raw/nominal-right-35-a09/result.json>) / [日志](<raw/nominal-right-35-a09/console.log>) |
| pure-left-15-a09 | completed | pure-left-15，500 / 16 / 42 | complete | [结果](<raw/pure-left-15-a09/result.json>) / [日志](<raw/pure-left-15-a09/console.log>) |
| pure-right-15-a09 | completed | pure-right-15，500 / 16 / 42 | complete | [结果](<raw/pure-right-15-a09/result.json>) / [日志](<raw/pure-right-15-a09/console.log>) |
| reverse-left-35-a09 | completed | reverse-left-35，500 / 16 / 42 | complete | [结果](<raw/reverse-left-35-a09/result.json>) / [日志](<raw/reverse-left-35-a09/console.log>) |
| reverse-right-35-a09 | completed | reverse-right-35，500 / 16 / 42 | complete | [结果](<raw/reverse-right-35-a09/result.json>) / [日志](<raw/reverse-right-35-a09/console.log>) |
| pure-left-10-a09 | completed | pure-left-10，500 / 16 / 42 | complete | [结果](<raw/pure-left-10-a09/result.json>) / [日志](<raw/pure-left-10-a09/console.log>) |
| pure-right-10-a09 | completed | pure-right-10，500 / 16 / 42 | complete | [结果](<raw/pure-right-10-a09/result.json>) / [日志](<raw/pure-right-10-a09/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-0123afd9bec3dafa3386d1d1dd74d2137ace75a7f96e86a7d197d4f627d189f6.json](<../../provenance/scenario-0123afd9bec3dafa3386d1d1dd74d2137ace75a7f96e86a7d197d4f627d189f6.json>)
- [场景来源 scenario-17629a7419c6a6b3ae3ca23b24f26a5d93719f1e83b05d2b4c22b441f6b3af56.json](<../../provenance/scenario-17629a7419c6a6b3ae3ca23b24f26a5d93719f1e83b05d2b4c22b441f6b3af56.json>)
- [场景来源 scenario-2036b2cc6b2c10685a21ea1b915d6d5a1f706c8be568fbc9aff3c2d7916f928f.json](<../../provenance/scenario-2036b2cc6b2c10685a21ea1b915d6d5a1f706c8be568fbc9aff3c2d7916f928f.json>)
- [场景来源 scenario-3b35d6488a343f1d77d010a4f00e0a2bc4129745b0fb0a6c6dcdd505aee692e2.json](<../../provenance/scenario-3b35d6488a343f1d77d010a4f00e0a2bc4129745b0fb0a6c6dcdd505aee692e2.json>)
- [场景来源 scenario-45d5beab4967dab4f6373cf128f8e35d96ef21647a1708a6dbb6fb6f642a5a99.json](<../../provenance/scenario-45d5beab4967dab4f6373cf128f8e35d96ef21647a1708a6dbb6fb6f642a5a99.json>)
- [场景来源 scenario-55f713f73991bd92f74d95bee0e0df198cdeccd72e8cb7b448ecf486935cbfbb.json](<../../provenance/scenario-55f713f73991bd92f74d95bee0e0df198cdeccd72e8cb7b448ecf486935cbfbb.json>)
- [场景来源 scenario-6db183337b6421f1e6d66184d2522377d452ebd1a895adde233f16c4639172f6.json](<../../provenance/scenario-6db183337b6421f1e6d66184d2522377d452ebd1a895adde233f16c4639172f6.json>)
- [场景来源 scenario-791a088cbdaa0d978774a185e7c13c3983fbd64d7134fd3cd9a634b1e444e832.json](<../../provenance/scenario-791a088cbdaa0d978774a185e7c13c3983fbd64d7134fd3cd9a634b1e444e832.json>)
- [场景来源 scenario-8545e1287ae4be6f69e11b5d3b8056fed2c31c88d6e92ac487f2ae1e71c50bd4.json](<../../provenance/scenario-8545e1287ae4be6f69e11b5d3b8056fed2c31c88d6e92ac487f2ae1e71c50bd4.json>)
- [场景来源 scenario-8dc939e47133b3967c3581e39315add0775358e2312262f83b1c54b66671cbcd.json](<../../provenance/scenario-8dc939e47133b3967c3581e39315add0775358e2312262f83b1c54b66671cbcd.json>)
- [场景来源 scenario-96a81b91a3dc1d89086a6c6caee262fa66793604eb3ff00914a496d593d90c2b.json](<../../provenance/scenario-96a81b91a3dc1d89086a6c6caee262fa66793604eb3ff00914a496d593d90c2b.json>)
- [场景来源 scenario-9b1e5b13e6f157158b4441665cc1721a2e32878e722e77394bd0d9824e1c0d8d.json](<../../provenance/scenario-9b1e5b13e6f157158b4441665cc1721a2e32878e722e77394bd0d9824e1c0d8d.json>)
- [场景来源 scenario-d62f99bc01b6c6d5fbc29c10550cfdd63820c4893c3897c4095681a9fdd29413.json](<../../provenance/scenario-d62f99bc01b6c6d5fbc29c10550cfdd63820c4893c3897c4095681a9fdd29413.json>)
- [场景来源 scenario-dbdb3998fef674fdf55c9d99e5c095ca44dd2601175da36f4c4d869d2b6d4a90.json](<../../provenance/scenario-dbdb3998fef674fdf55c9d99e5c095ca44dd2601175da36f4c4d869d2b6d4a90.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `pure-left-35-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22968083809903353, "tracking_yaw_rmse": 0.2964183126601082}`
- `pure-left-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-35-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22906660877254187, "tracking_yaw_rmse": 0.2950337750021955}`
- `pure-right-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-left-35-a09` 指标：`{"max_joint_velocity_utilization": 1.1484869003295899, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22321420415062562, "tracking_yaw_rmse": 0.2976732423105175}`
- `forward-left-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-right-35-a09` 指标：`{"max_joint_velocity_utilization": 1.1588099479675293, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22286060727741577, "tracking_yaw_rmse": 0.2880976974246962}`
- `forward-right-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-left-25-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.23053126112930272, "tracking_yaw_rmse": 0.2252810943725598}`
- `pure-left-25-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.25], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-25-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22980382866750018, "tracking_yaw_rmse": 0.22822164666215133}`
- `pure-right-25-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.25], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `nominal-left-35-a09` 指标：`{"max_joint_velocity_utilization": 1.1776507377624512, "max_tilt": 0.10311154276132584, "termination_rate": 0.0, "tracking_xy_rmse": 0.23368729484301046, "tracking_yaw_rmse": 0.31974052882131954}`
- `nominal-left-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `nominal-right-35-a09` 指标：`{"max_joint_velocity_utilization": 1.050358200073242, "max_tilt": 0.11075852811336517, "termination_rate": 0.0, "tracking_xy_rmse": 0.2295122396773987, "tracking_yaw_rmse": 0.33985821824587076}`
- `nominal-right-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `pure-left-15-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22968260729454035, "tracking_yaw_rmse": 0.1602432367596587}`
- `pure-left-15-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.15], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-15-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.22961658910392244, "tracking_yaw_rmse": 0.16496272222539266}`
- `pure-right-15-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.15], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-left-35-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.2497352641289436, "tracking_yaw_rmse": 0.3007070545944628}`
- `reverse-left-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-right-35-a09` 指标：`{"max_joint_velocity_utilization": 1.3284496307373046, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.251454576024823, "tracking_yaw_rmse": 0.2985618437960745}`
- `reverse-right-35-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-left-10-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.05993435121872333, "tracking_yaw_rmse": 0.1123158763437811}`
- `pure-left-10-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.1], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-10-a09` 指标：`{"max_joint_velocity_utilization": 1.1314333915710448, "max_tilt": 0.24553410708904266, "termination_rate": 0.0, "tracking_xy_rmse": 0.05875318515761778, "tracking_yaw_rmse": 0.12402376650828553}`
- `pure-right-10-a09` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.1], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。

## 观察与限制

- Seed 42, 16 matched environments; no external pushes or actor sensor noise. Steps 0-99 stationary, then 400 control steps at the specified command. Analyze steps 200-499 inclusive. Only nominal cases reuse the nominal environment of batch 006 with the full configured command range.
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [population-recorder.json](<population-recorder.json>)
- [manifest.json](<../commandamp-50000-reverse-expert-20261127-008/manifest.json>)
- [manifest.json](<../commandamp-50000-stand-population-20261127-006/manifest.json>)
- [yaw-diagnostics-all-envs.npz](<raw/forward-left-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/forward-right-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/nominal-left-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/nominal-right-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-left-10-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-left-15-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-left-25-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-left-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-right-10-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-right-15-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-right-25-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-right-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/reverse-left-35-a09/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/reverse-right-35-a09/yaw-diagnostics-all-envs.npz>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

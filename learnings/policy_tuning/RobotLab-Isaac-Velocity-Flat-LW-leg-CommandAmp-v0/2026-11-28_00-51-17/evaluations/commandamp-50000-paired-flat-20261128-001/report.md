# 评估批次 commandamp-50000-paired-flat-20261128-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-28_00-51-17`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-nominal-nopush-n01 | completed | stand-nominal-nopush，700 / 16 / 42 | complete | [结果](<raw/stand-nominal-nopush-n01/result.json>) / [日志](<raw/stand-nominal-nopush-n01/console.log>) |
| stand-randomized-nopush-n01 | completed | stand-randomized-nopush，700 / 16 / 42 | complete | [结果](<raw/stand-randomized-nopush-n01/result.json>) / [日志](<raw/stand-randomized-nopush-n01/console.log>) |
| forward-20-n01 | completed | forward-20，500 / 16 / 42 | complete | [结果](<raw/forward-20-n01/result.json>) / [日志](<raw/forward-20-n01/console.log>) |
| reverse-20-n01 | completed | reverse-20，500 / 16 / 42 | complete | [结果](<raw/reverse-20-n01/result.json>) / [日志](<raw/reverse-20-n01/console.log>) |
| forward-40-n01 | completed | forward-40，500 / 16 / 42 | complete | [结果](<raw/forward-40-n01/result.json>) / [日志](<raw/forward-40-n01/console.log>) |
| reverse-40-n01 | completed | reverse-40，500 / 16 / 42 | complete | [结果](<raw/reverse-40-n01/result.json>) / [日志](<raw/reverse-40-n01/console.log>) |
| forward-60-n01 | completed | forward-60，500 / 16 / 42 | complete | [结果](<raw/forward-60-n01/result.json>) / [日志](<raw/forward-60-n01/console.log>) |
| reverse-60-n01 | completed | reverse-60，500 / 16 / 42 | complete | [结果](<raw/reverse-60-n01/result.json>) / [日志](<raw/reverse-60-n01/console.log>) |
| forward-80-n01 | completed | forward-80，500 / 16 / 42 | complete | [结果](<raw/forward-80-n01/result.json>) / [日志](<raw/forward-80-n01/console.log>) |
| reverse-80-n01 | completed | reverse-80，500 / 16 / 42 | complete | [结果](<raw/reverse-80-n01/result.json>) / [日志](<raw/reverse-80-n01/console.log>) |
| pure-left-15-n01 | completed | pure-left-15，500 / 16 / 42 | complete | [结果](<raw/pure-left-15-n01/result.json>) / [日志](<raw/pure-left-15-n01/console.log>) |
| pure-right-15-n01 | completed | pure-right-15，500 / 16 / 42 | complete | [结果](<raw/pure-right-15-n01/result.json>) / [日志](<raw/pure-right-15-n01/console.log>) |
| pure-left-25-n01 | completed | pure-left-25，500 / 16 / 42 | complete | [结果](<raw/pure-left-25-n01/result.json>) / [日志](<raw/pure-left-25-n01/console.log>) |
| pure-right-25-n01 | completed | pure-right-25，500 / 16 / 42 | complete | [结果](<raw/pure-right-25-n01/result.json>) / [日志](<raw/pure-right-25-n01/console.log>) |
| pure-left-35-n01 | completed | pure-left-35，500 / 16 / 42 | complete | [结果](<raw/pure-left-35-n01/result.json>) / [日志](<raw/pure-left-35-n01/console.log>) |
| pure-right-35-n01 | completed | pure-right-35，500 / 16 / 42 | complete | [结果](<raw/pure-right-35-n01/result.json>) / [日志](<raw/pure-right-35-n01/console.log>) |
| forward-left-35-n01 | completed | forward-left-35，500 / 16 / 42 | complete | [结果](<raw/forward-left-35-n01/result.json>) / [日志](<raw/forward-left-35-n01/console.log>) |
| forward-right-35-n01 | completed | forward-right-35，500 / 16 / 42 | complete | [结果](<raw/forward-right-35-n01/result.json>) / [日志](<raw/forward-right-35-n01/console.log>) |
| reverse-left-35-n01 | completed | reverse-left-35，500 / 16 / 42 | complete | [结果](<raw/reverse-left-35-n01/result.json>) / [日志](<raw/reverse-left-35-n01/console.log>) |
| reverse-right-35-n01 | completed | reverse-right-35，500 / 16 / 42 | complete | [结果](<raw/reverse-right-35-n01/result.json>) / [日志](<raw/reverse-right-35-n01/console.log>) |
| stand-boundary-n01 | completed | stand-boundary，960 / 16 / 42 | complete | [结果](<raw/stand-boundary-n01/result.json>) / [日志](<raw/stand-boundary-n01/console.log>) |
| straight-training-n01 | completed | straight-training，960 / 16 / 42 | complete | [结果](<raw/straight-training-n01/result.json>) / [日志](<raw/straight-training-n01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`1b85e21ff519e3580af43016ed595955bc4afb9b03bf237a2ba8c6b625682737`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-0289a8724c04ff4db38f2effcc1110aea733e548b3207dac30290ff310b36e74.json](<../../provenance/scenario-0289a8724c04ff4db38f2effcc1110aea733e548b3207dac30290ff310b36e74.json>)
- [场景来源 scenario-147a3d496f67f2741d82c90daebf170df944ba07cd2b8d5727be9dffb17d3d8c.json](<../../provenance/scenario-147a3d496f67f2741d82c90daebf170df944ba07cd2b8d5727be9dffb17d3d8c.json>)
- [场景来源 scenario-325dab227884349dea63e1cf7ec1da99d25565277f84251fb271f20b37d65f7e.json](<../../provenance/scenario-325dab227884349dea63e1cf7ec1da99d25565277f84251fb271f20b37d65f7e.json>)
- [场景来源 scenario-617146defcf7f2e2af1ba12437c0fe1795655212748958e697d29fc84e3aabd7.json](<../../provenance/scenario-617146defcf7f2e2af1ba12437c0fe1795655212748958e697d29fc84e3aabd7.json>)
- [场景来源 scenario-6a371388768f4368671055fb26f406b1bd8a7367b2d8e88ad2b83e070de79cc6.json](<../../provenance/scenario-6a371388768f4368671055fb26f406b1bd8a7367b2d8e88ad2b83e070de79cc6.json>)
- [场景来源 scenario-6f2a4056e012d351f8a7b3e40efe887aa6ff84c30d40fba78f1b1e07fe79d09a.json](<../../provenance/scenario-6f2a4056e012d351f8a7b3e40efe887aa6ff84c30d40fba78f1b1e07fe79d09a.json>)
- [场景来源 scenario-7a5f8a0adaa27c3d4eb31a626c964f1d22c66e69acb026dd47c8ce2e66e59ecf.json](<../../provenance/scenario-7a5f8a0adaa27c3d4eb31a626c964f1d22c66e69acb026dd47c8ce2e66e59ecf.json>)
- [场景来源 scenario-81546f4a03d7b32d183f7d403b0d37c97bc8f077f6bb708086bb0b66090d90e4.json](<../../provenance/scenario-81546f4a03d7b32d183f7d403b0d37c97bc8f077f6bb708086bb0b66090d90e4.json>)
- [场景来源 scenario-9bb6258af5e5d981cd99dabd2a3aa06ad07a7b3dbc121bca32a645ed75d8931b.json](<../../provenance/scenario-9bb6258af5e5d981cd99dabd2a3aa06ad07a7b3dbc121bca32a645ed75d8931b.json>)
- [场景来源 scenario-aa8e6cb09bb901307de9795c571e8fca415eaa38b3f01c1635d6c9f3714326f3.json](<../../provenance/scenario-aa8e6cb09bb901307de9795c571e8fca415eaa38b3f01c1635d6c9f3714326f3.json>)
- [场景来源 scenario-ad59e52aa614c20e1015bacb25e767ae02f636794e0fdcfab2cc40204ab93def.json](<../../provenance/scenario-ad59e52aa614c20e1015bacb25e767ae02f636794e0fdcfab2cc40204ab93def.json>)
- [场景来源 scenario-b5369028fec56ac7fcb37f40034ce0fa72abff82e513cb38567a654eced5bd8c.json](<../../provenance/scenario-b5369028fec56ac7fcb37f40034ce0fa72abff82e513cb38567a654eced5bd8c.json>)
- [场景来源 scenario-b8ed1d6a946684b5dfd9747f018fd668feb652e3ef69b453e9f6c06670b1b84d.json](<../../provenance/scenario-b8ed1d6a946684b5dfd9747f018fd668feb652e3ef69b453e9f6c06670b1b84d.json>)
- [场景来源 scenario-bc6fdca4e3cc1205fbbd06a17b9814a8b069d2312415099f679ceb7d735f2275.json](<../../provenance/scenario-bc6fdca4e3cc1205fbbd06a17b9814a8b069d2312415099f679ceb7d735f2275.json>)
- [场景来源 scenario-bcb387239bf9268427ed8205819223e0105af39625f19d1a7740eedb69567a2e.json](<../../provenance/scenario-bcb387239bf9268427ed8205819223e0105af39625f19d1a7740eedb69567a2e.json>)
- [场景来源 scenario-cf77cdee1e114163b70d4b714e097da67f2b9998c048adda887829b1e152ca87.json](<../../provenance/scenario-cf77cdee1e114163b70d4b714e097da67f2b9998c048adda887829b1e152ca87.json>)
- [场景来源 scenario-da532b311ee31385046e724a1147c42b8425c480b06a0765d09bd7ee01d3ca42.json](<../../provenance/scenario-da532b311ee31385046e724a1147c42b8425c480b06a0765d09bd7ee01d3ca42.json>)
- [场景来源 scenario-db8681de8af92a5bc103b1e59e8892bce0cb300969f12eba448f907eb514afbf.json](<../../provenance/scenario-db8681de8af92a5bc103b1e59e8892bce0cb300969f12eba448f907eb514afbf.json>)
- [场景来源 scenario-ea55b63a98a7a3d6f839044d955a71fd4460cc78dbdf1fc781b93dd2c4d4ca05.json](<../../provenance/scenario-ea55b63a98a7a3d6f839044d955a71fd4460cc78dbdf1fc781b93dd2c4d4ca05.json>)
- [场景来源 scenario-eaedc2fee08b8a93672ad8830a5de6bb0eb31e0dcaaf205439de7f726e81b665.json](<../../provenance/scenario-eaedc2fee08b8a93672ad8830a5de6bb0eb31e0dcaaf205439de7f726e81b665.json>)
- [场景来源 scenario-f0bed3b555938e8efc6a275557088ad7139bc78700d577da61b340507e689473.json](<../../provenance/scenario-f0bed3b555938e8efc6a275557088ad7139bc78700d577da61b340507e689473.json>)
- [场景来源 scenario-f0d4fdfe4cc0d226fe8ce96ed7dbdfda8638ba7994acaaa26821f92fef31dc18.json](<../../provenance/scenario-f0d4fdfe4cc0d226fe8ce96ed7dbdfda8638ba7994acaaa26821f92fef31dc18.json>)
- [训练上下文 context-29a20b11252b6ccb430fe5e084e35e5b9a68267288a599a06641187578c45807.json](<../../provenance/context-29a20b11252b6ccb430fe5e084e35e5b9a68267288a599a06641187578c45807.json>)
- [训练有效配置 config-2b5d32152c9fb6782543e4a77bd75562ddd5f16b33c5b966a530f905b0df874c.json](<../../provenance/config-2b5d32152c9fb6782543e4a77bd75562ddd5f16b33c5b966a530f905b0df874c.json>)

- `stand-nominal-nopush-n01` 指标：`{"max_joint_velocity_utilization": 0.33992044925689696, "max_tilt": 0.05304308980703354, "termination_rate": 0.0, "tracking_xy_rmse": 0.014108024444432382, "tracking_yaw_rmse": 0.016141779327400997}`
- `stand-nominal-nopush-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 699, "start_step": 0}]`；训练配置覆盖：`{"commands.base_velocity.ranges.ang_vel_z": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_x": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_y": [0.0, 0.0], "events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `stand-randomized-nopush-n01` 指标：`{"max_joint_velocity_utilization": 1.1455259323120117, "max_tilt": 0.15367138385772705, "termination_rate": 0.0, "tracking_xy_rmse": 0.05494760578833611, "tracking_yaw_rmse": 0.06132183586091237}`
- `stand-randomized-nopush-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 699, "start_step": 0}]`；训练配置覆盖：`{"commands.base_velocity.ranges.ang_vel_z": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_x": [0.0, 0.0], "commands.base_velocity.ranges.lin_vel_y": [0.0, 0.0], "events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": true}`。
- `forward-20-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.21901203442459932, "tracking_yaw_rmse": 0.10474756988587258}`
- `forward-20-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.2, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-20-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.23124335054363396, "tracking_yaw_rmse": 0.10973109771522185}`
- `reverse-20-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.2, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-40-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.16589973866939545, "termination_rate": 0.0, "tracking_xy_rmse": 0.2198356463696882, "tracking_yaw_rmse": 0.1090150799573337}`
- `forward-40-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-40-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.24342681251419349, "tracking_yaw_rmse": 0.11873898090323183}`
- `reverse-40-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-60-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.19984786212444305, "termination_rate": 0.0, "tracking_xy_rmse": 0.22767313271312098, "tracking_yaw_rmse": 0.1066564955767222}`
- `forward-60-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.6, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-60-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.26090667604520384, "tracking_yaw_rmse": 0.13446456679461818}`
- `reverse-60-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.6, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-80-n01` 指标：`{"max_joint_velocity_utilization": 1.1465131759643554, "max_tilt": 0.21100781857967377, "termination_rate": 0.0, "tracking_xy_rmse": 0.24154047709980356, "tracking_yaw_rmse": 0.12202352483154304}`
- `forward-80-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-80-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.2715678243944544, "tracking_yaw_rmse": 0.15752948166635225}`
- `reverse-80-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-left-15-n01` 指标：`{"max_joint_velocity_utilization": 1.1246569633483887, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.22300270897289592, "tracking_yaw_rmse": 0.15604054017321212}`
- `pure-left-15-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.15], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-15-n01` 指标：`{"max_joint_velocity_utilization": 1.0905972480773927, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.22241055367812385, "tracking_yaw_rmse": 0.1640585030840259}`
- `pure-right-15-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.15], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-left-25-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15193602442741394, "termination_rate": 0.0, "tracking_xy_rmse": 0.22078913260023525, "tracking_yaw_rmse": 0.22140654427508263}`
- `pure-left-25-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.25], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-25-n01` 指标：`{"max_joint_velocity_utilization": 1.1104143142700196, "max_tilt": 0.17570051550865173, "termination_rate": 0.0, "tracking_xy_rmse": 0.2187140759118819, "tracking_yaw_rmse": 0.21796979640633063}`
- `pure-right-25-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.25], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-left-35-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.2300437092781067, "termination_rate": 0.0, "tracking_xy_rmse": 0.2146950341125619, "tracking_yaw_rmse": 0.270199484213821}`
- `pure-left-35-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `pure-right-35-n01` 指标：`{"max_joint_velocity_utilization": 1.209156322479248, "max_tilt": 0.21939703822135925, "termination_rate": 0.0, "tracking_xy_rmse": 0.21627565674782953, "tracking_yaw_rmse": 0.2606438708655704}`
- `pure-right-35-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-left-35-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.2233162671327591, "termination_rate": 0.0, "tracking_xy_rmse": 0.2028850235982188, "tracking_yaw_rmse": 0.23838784836632787}`
- `forward-left-35-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `forward-right-35-n01` 指标：`{"max_joint_velocity_utilization": 1.0917036056518554, "max_tilt": 0.23666317760944366, "termination_rate": 0.0, "tracking_xy_rmse": 0.20670012546481645, "tracking_yaw_rmse": 0.23798356039317298}`
- `forward-right-35-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-left-35-n01` 指标：`{"max_joint_velocity_utilization": 1.528310203552246, "max_tilt": 0.2443999946117401, "termination_rate": 0.0, "tracking_xy_rmse": 0.22484173308171454, "tracking_yaw_rmse": 0.2732126423395577}`
- `reverse-left-35-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `reverse-right-35-n01` 指标：`{"max_joint_velocity_utilization": 1.3682411193847657, "max_tilt": 0.2520306706428528, "termination_rate": 0.0, "tracking_xy_rmse": 0.22755927985188756, "tracking_yaw_rmse": 0.26328304636038596}`
- `reverse-right-35-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false, "rewards.track_ang_vel_z_exp.params.std": 0.5, "rewards.track_lin_vel_xy_exp.params.std": 0.5}`。
- `stand-boundary-n01` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.1509523870728939, "tracking_yaw_rmse": 0.09463022571195373}`
- `stand-boundary-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 159, "start_step": 0}, {"command": [0.1, 0, 0], "end_step": 319, "start_step": 160}, {"command": [0.12, 0, 0], "end_step": 479, "start_step": 320}, {"command": [0, 0, 0.1], "end_step": 639, "start_step": 480}, {"command": [0, 0, 0.12], "end_step": 799, "start_step": 640}, {"command": [0, 0, 0], "end_step": 959, "start_step": 800}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `straight-training-n01` 指标：`{"max_joint_velocity_utilization": 1.7222597122192382, "max_tilt": 0.27053722739219666, "termination_rate": 0.0, "tracking_xy_rmse": 0.2569785704638362, "tracking_yaw_rmse": 0.1383657474364241}`
- `straight-training-n01` 命令调度：`[{"command": [0, 0, 0], "end_step": 159, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 319, "start_step": 160}, {"command": [0.8, 0, 0], "end_step": 479, "start_step": 320}, {"command": [0, 0, 0], "end_step": 639, "start_step": 480}, {"command": [-0.4, 0, 0], "end_step": 799, "start_step": 640}, {"command": [-0.8, 0, 0], "end_step": 959, "start_step": 800}]`；训练配置覆盖：`{}`。

## 观察与限制

- 对照 11 月 26 日同迭代检查点的四个已封存批次，各用例逐字段复用相同场景。每项 16 个并行环境，固定种子 42。
- 恒速段按最后 6 秒统计航向系前向速度和世界竖直轴角速度。两组长静站均无推扰，另测指令边界和带训练随机化的直线换向。
- 恒速对照沿用旧评估的奖励参数覆盖。确定性 Actor 只读取其策略观测，比较依据为物理遥测。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [population-recorder.json](<population-recorder.json>)
- [speed-all-envs.npz](<raw/forward-20-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/forward-40-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/forward-60-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/forward-80-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/forward-left-35-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/forward-right-35-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/pure-left-15-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/pure-left-25-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/pure-left-35-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/pure-right-15-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/pure-right-25-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/pure-right-35-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/reverse-20-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/reverse-40-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/reverse-60-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/reverse-80-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/reverse-left-35-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/reverse-right-35-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/stand-boundary-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/stand-nominal-nopush-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/stand-randomized-nopush-n01/speed-all-envs.npz>)
- [speed-all-envs.npz](<raw/straight-training-n01/speed-all-envs.npz>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

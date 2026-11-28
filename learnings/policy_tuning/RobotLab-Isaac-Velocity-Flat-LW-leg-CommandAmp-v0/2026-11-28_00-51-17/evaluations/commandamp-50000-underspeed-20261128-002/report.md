# 评估批次 commandamp-50000-underspeed-20261128-002

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-28_00-51-17`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| reverse-80-d02 | completed | reverse-80，500 / 16 / 42 | complete | [结果](<raw/reverse-80-d02/result.json>) / [日志](<raw/reverse-80-d02/console.log>) |
| forward-80-d02 | completed | forward-80，500 / 16 / 42 | complete | [结果](<raw/forward-80-d02/result.json>) / [日志](<raw/forward-80-d02/console.log>) |
| pure-left-35-d02 | completed | pure-left-35，500 / 16 / 42 | complete | [结果](<raw/pure-left-35-d02/result.json>) / [日志](<raw/pure-left-35-d02/console.log>) |
| pure-right-35-d02 | completed | pure-right-35，500 / 16 / 42 | complete | [结果](<raw/pure-right-35-d02/result.json>) / [日志](<raw/pure-right-35-d02/console.log>) |
| reverse-40-d02 | completed | reverse-40，500 / 16 / 42 | complete | [结果](<raw/reverse-40-d02/result.json>) / [日志](<raw/reverse-40-d02/console.log>) |
| forward-40-d02 | completed | forward-40，500 / 16 / 42 | complete | [结果](<raw/forward-40-d02/result.json>) / [日志](<raw/forward-40-d02/console.log>) |
| pure-left-15-d02 | completed | pure-left-15，500 / 16 / 42 | complete | [结果](<raw/pure-left-15-d02/result.json>) / [日志](<raw/pure-left-15-d02/console.log>) |
| pure-right-15-d02 | completed | pure-right-15，500 / 16 / 42 | complete | [结果](<raw/pure-right-15-d02/result.json>) / [日志](<raw/pure-right-15-d02/console.log>) |
| forward-left-35-d02 | completed | forward-left-35，500 / 16 / 42 | complete | [结果](<raw/forward-left-35-d02/result.json>) / [日志](<raw/forward-left-35-d02/console.log>) |
| reverse-left-35-d02 | completed | reverse-left-35，500 / 16 / 42 | complete | [结果](<raw/reverse-left-35-d02/result.json>) / [日志](<raw/reverse-left-35-d02/console.log>) |
| nominal-reverse-80-d02 | completed | nominal-reverse-80，500 / 16 / 42 | complete | [结果](<raw/nominal-reverse-80-d02/result.json>) / [日志](<raw/nominal-reverse-80-d02/console.log>) |
| nominal-pure-left-35-d02 | completed | nominal-pure-left-35，500 / 16 / 42 | complete | [结果](<raw/nominal-pure-left-35-d02/result.json>) / [日志](<raw/nominal-pure-left-35-d02/console.log>) |
| nominal-pure-right-35-d02 | completed | nominal-pure-right-35，500 / 16 / 42 | complete | [结果](<raw/nominal-pure-right-35-d02/result.json>) / [日志](<raw/nominal-pure-right-35-d02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`1b85e21ff519e3580af43016ed595955bc4afb9b03bf237a2ba8c6b625682737`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-06b2313d9f69ea8a5fee908d3f8f53438efde6be62b28ae3b471701a675d1c48.json](<../../provenance/scenario-06b2313d9f69ea8a5fee908d3f8f53438efde6be62b28ae3b471701a675d1c48.json>)
- [场景来源 scenario-0e8bb563fbc66749722766d2d26362bc5cd46496743821103c798fcda281661e.json](<../../provenance/scenario-0e8bb563fbc66749722766d2d26362bc5cd46496743821103c798fcda281661e.json>)
- [场景来源 scenario-22ddc494b7e094b669fcdbd7f95a0dd08d0a0dcb5f42d34d50fb544929a7eb28.json](<../../provenance/scenario-22ddc494b7e094b669fcdbd7f95a0dd08d0a0dcb5f42d34d50fb544929a7eb28.json>)
- [场景来源 scenario-381a403f14f88617af7286fcaf3743db95eb0e3457705261b08e73e8accb729e.json](<../../provenance/scenario-381a403f14f88617af7286fcaf3743db95eb0e3457705261b08e73e8accb729e.json>)
- [场景来源 scenario-39ea30e71fc9c2373e8fb811c3534dc9eae0cfe2733b0b19a958c81e92a70132.json](<../../provenance/scenario-39ea30e71fc9c2373e8fb811c3534dc9eae0cfe2733b0b19a958c81e92a70132.json>)
- [场景来源 scenario-528b891275163a7764fb4486b1b70c232285541691abc036134e3244d427de86.json](<../../provenance/scenario-528b891275163a7764fb4486b1b70c232285541691abc036134e3244d427de86.json>)
- [场景来源 scenario-537e47263d4b5fb0d98d4f47d3d4bc1b395c2a4493e390d3002333bcd719ad43.json](<../../provenance/scenario-537e47263d4b5fb0d98d4f47d3d4bc1b395c2a4493e390d3002333bcd719ad43.json>)
- [场景来源 scenario-93dc46cf38544fad036b1fdb95b4a82ad910324262562d653fe6b7f2edce240a.json](<../../provenance/scenario-93dc46cf38544fad036b1fdb95b4a82ad910324262562d653fe6b7f2edce240a.json>)
- [场景来源 scenario-9ba68e22ed5ae3ab5afa7c2e1c25125c6cf527216fcf0fb69948329d9604e732.json](<../../provenance/scenario-9ba68e22ed5ae3ab5afa7c2e1c25125c6cf527216fcf0fb69948329d9604e732.json>)
- [场景来源 scenario-a2e5aa16f1a24376c71c88a6daec41d6f23ff62f325232e7bb51e15e4c9fe7d9.json](<../../provenance/scenario-a2e5aa16f1a24376c71c88a6daec41d6f23ff62f325232e7bb51e15e4c9fe7d9.json>)
- [场景来源 scenario-c5d984692bac969dee8e31153ddcde34fd8d20950731dc8890dc97c788438417.json](<../../provenance/scenario-c5d984692bac969dee8e31153ddcde34fd8d20950731dc8890dc97c788438417.json>)
- [场景来源 scenario-c7b40dfb10d3e7149189a55ed0271d45af38012824ccc0455700786e199a74cb.json](<../../provenance/scenario-c7b40dfb10d3e7149189a55ed0271d45af38012824ccc0455700786e199a74cb.json>)
- [场景来源 scenario-fae59e76becd5160069167d49bd0a478a09938bcebaf95723005c5102db3e708.json](<../../provenance/scenario-fae59e76becd5160069167d49bd0a478a09938bcebaf95723005c5102db3e708.json>)
- [训练上下文 context-29a20b11252b6ccb430fe5e084e35e5b9a68267288a599a06641187578c45807.json](<../../provenance/context-29a20b11252b6ccb430fe5e084e35e5b9a68267288a599a06641187578c45807.json>)
- [训练有效配置 config-2b5d32152c9fb6782543e4a77bd75562ddd5f16b33c5b966a530f905b0df874c.json](<../../provenance/config-2b5d32152c9fb6782543e4a77bd75562ddd5f16b33c5b966a530f905b0df874c.json>)

- `reverse-80-d02` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.2715678243944544, "tracking_yaw_rmse": 0.15752948166635225}`
- `reverse-80-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `forward-80-d02` 指标：`{"max_joint_velocity_utilization": 1.1465131759643554, "max_tilt": 0.21100781857967377, "termination_rate": 0.0, "tracking_xy_rmse": 0.24154047709980356, "tracking_yaw_rmse": 0.12202352483154304}`
- `forward-80-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `pure-left-35-d02` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.2300437092781067, "termination_rate": 0.0, "tracking_xy_rmse": 0.2146950341125619, "tracking_yaw_rmse": 0.270199484213821}`
- `pure-left-35-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `pure-right-35-d02` 指标：`{"max_joint_velocity_utilization": 1.209156322479248, "max_tilt": 0.21939703822135925, "termination_rate": 0.0, "tracking_xy_rmse": 0.21627565674782953, "tracking_yaw_rmse": 0.2606438708655704}`
- `pure-right-35-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `reverse-40-d02` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.24342681251419349, "tracking_yaw_rmse": 0.11873898090323183}`
- `reverse-40-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `forward-40-d02` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.16589973866939545, "termination_rate": 0.0, "tracking_xy_rmse": 0.2198356463696882, "tracking_yaw_rmse": 0.1090150799573337}`
- `forward-40-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `pure-left-15-d02` 指标：`{"max_joint_velocity_utilization": 1.1246569633483887, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.22300270897289592, "tracking_yaw_rmse": 0.15604054017321212}`
- `pure-left-15-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.15], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `pure-right-15-d02` 指标：`{"max_joint_velocity_utilization": 1.0905972480773927, "max_tilt": 0.15089142322540283, "termination_rate": 0.0, "tracking_xy_rmse": 0.22241055367812385, "tracking_yaw_rmse": 0.1640585030840259}`
- `pure-right-15-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.15], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `forward-left-35-d02` 指标：`{"max_joint_velocity_utilization": 1.089055633544922, "max_tilt": 0.2233162671327591, "termination_rate": 0.0, "tracking_xy_rmse": 0.2028850235982188, "tracking_yaw_rmse": 0.23838784836632787}`
- `forward-left-35-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `reverse-left-35-d02` 指标：`{"max_joint_velocity_utilization": 1.528310203552246, "max_tilt": 0.2443999946117401, "termination_rate": 0.0, "tracking_xy_rmse": 0.22484173308171454, "tracking_yaw_rmse": 0.2732126423395577}`
- `reverse-left-35-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `nominal-reverse-80-d02` 指标：`{"max_joint_velocity_utilization": 1.2924079895019531, "max_tilt": 0.1381634771823883, "termination_rate": 0.0, "tracking_xy_rmse": 0.27719827079758885, "tracking_yaw_rmse": 0.187555958569437}`
- `nominal-reverse-80-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `nominal-pure-left-35-d02` 指标：`{"max_joint_velocity_utilization": 1.2064568519592285, "max_tilt": 0.2177274376153946, "termination_rate": 0.0, "tracking_xy_rmse": 0.22373400271112037, "tracking_yaw_rmse": 0.30182304326623277}`
- `nominal-pure-left-35-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `nominal-pure-right-35-d02` 指标：`{"max_joint_velocity_utilization": 1.0950896263122558, "max_tilt": 0.2620965838432312, "termination_rate": 0.0, "tracking_xy_rmse": 0.22052475186887957, "tracking_yaw_rmse": 0.3201029740039628}`
- `nominal-pure-right-35-d02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.35], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.add_joint_default_pos": null, "events.push_robot_hard": null, "events.randomize_actuator_gains": null, "events.randomize_apply_external_force_torque": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {}, "events.randomize_reset_base.params.velocity_range": {}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 使用 11 月 28 日的完整训练奖励，线速度跟踪标准差 0.50，偏航 0.30。稳态窗口为第 200 至 499 步。原始动作和环境物理过程保持训练定义。测量记录器沿用既有足端、接触、奖励和 Actor 输入读数，通过运行时指定的 28 维 AMP 帧宽读取新观测。
- 按实测的 5 N 支撑力门限测量步频和滑动。标称组使用平面和标称物理参数，随机化组保留训练地形、材料和机器人参数分布。两组均关闭外部推扰和策略观测噪声。
- 在 −0.8 m/s 指令下，同脚触地周期频率为 1.254 Hz，每周期后移 0.501 m，专家巡航估计为 0.630 m。足端在机身坐标系下的前后范围（P95 减 P05）为 24.5 cm，专家为 28.7 cm。
- 同给 −0.8 m/s，最接近指令和偏差最大的各四个环境分别达到 -0.712 和 -0.545 m/s。前组每步加权任务回报多 0.00885，风格回报少 0.01324，混合后少 0.00439。−0.4 m/s 未出现同样的净反转。分组中同时存在物理参数差异，这个比较只能描述实测回报的取舍。
- 原地 ±0.35 rad/s 时，每个同脚触地周期的机身航向变化为 +7.5°、-8.5°。专家每周期竖直轴转量约 +15.8°、-15.8°。两侧实测步频约 1.25 Hz。摆动中段的左右足前后位移差显示，策略和专家的双脚协同方式不同。
- 固定 ±0.35 rad/s 指令，将对应专家与同向 ±0.15 rad/s 慢速专家的有效十帧窗配对。判别器把对应速度排在慢速前面的比例，左转为 52.9%，右转为 47.6%。用相同十帧的有符号偏航速度均值排序，两侧都超过 99.9%。−0.8 与 −0.6 m/s 倒退的判别器排序比例为 80.1%，对应加权风格回报每步只差 0.000226。
- 冻结判别器，在真实高速倒退窗口中换入相近专家的纵向速度序列，原始判别分平均增加 0.013；换入专家的关节和足端通道，平均增加 0.955。拼接窗口仅用于检查网络对输入的响应。
- 在原地 ±0.35 rad/s，跟踪较好的四个环境每步混合回报仍多 0.00269、0.00187，风格项缩小了这一优势。标称平面上倒退为 -0.695 m/s，原地两侧为 +0.101、-0.088 rad/s。移除随机化后仍有欠速。
- 训练指令的 vx 和 wz 分别独立均匀采样，另有 2% 显式静站。按此配置，|vx|≤0.05 m/s 且 |wz|≥0.30 rad/s 的理论概率为 0.875%（左右合计）。前进和后退的采样范围对称，单凭该分布难以解释前后跟踪差距。
- 按训练配置逐帧复算了实际速度跟踪奖励，线速度标准差 0.50，偏航 0.30。所有稳态移动用例中 stop_motion、stand_still 均为零。已复核专家网格命中、28 维 AMP 中的物理角速度、Actor 历史内的原始指令及原生动作重放。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [population-recorder.json](<../../../2026-11-26_21-23-35/evaluations/commandamp-50000-yaw-diagnosis-20261127-009/population-recorder.json>)
- [yaw-diagnostics-all-envs.npz](<raw/forward-40-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/forward-80-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/forward-left-35-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/nominal-pure-left-35-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/nominal-pure-right-35-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/nominal-reverse-80-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-left-15-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-left-35-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-right-15-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/pure-right-35-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/reverse-40-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/reverse-80-d02/yaw-diagnostics-all-envs.npz>)
- [yaw-diagnostics-all-envs.npz](<raw/reverse-left-35-d02/yaw-diagnostics-all-envs.npz>)
- [actor-condition-response.csv](<analysis/actor-condition-response.csv>)
- [diagnosis.json](<analysis/diagnosis.json>)
- [expert-ranking.csv](<analysis/expert-ranking.csv>)
- [expert-sampling.csv](<analysis/expert-sampling.csv>)
- [feature-replacement.csv](<analysis/feature-replacement.csv>)
- [matched-env-rewards.csv](<analysis/matched-env-rewards.csv>)
- [motion-and-contact.csv](<analysis/motion-and-contact.csv>)
- [reward-by-term.csv](<analysis/reward-by-term.csv>)
- [task-and-style.csv](<analysis/task-and-style.csv>)
- [underspeed-diagnosis.png](<analysis/underspeed-diagnosis.png>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

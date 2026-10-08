# 评估批次 wheel-flat-final-std01-20261004-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-10-03_22-59-50`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| right-delay15-noise-a01 | completed | right-delay15-noise，4500 / 1 / 42 | complete | [结果](<raw/right-delay15-noise-a01/result.json>) / [日志](<raw/right-delay15-noise-a01/console.log>) |
| left-delay15-noise-a01 | completed | left-delay15-noise，4500 / 1 / 42 | complete | [结果](<raw/left-delay15-noise-a01/result.json>) / [日志](<raw/left-delay15-noise-a01/console.log>) |
| right-delay30-noise-a01 | completed | right-delay30-noise，4500 / 1 / 42 | complete | [结果](<raw/right-delay30-noise-a01/result.json>) / [日志](<raw/right-delay30-noise-a01/console.log>) |
| left-delay30-noise-a01 | completed | left-delay30-noise，4500 / 1 / 42 | complete | [结果](<raw/left-delay30-noise-a01/result.json>) / [日志](<raw/left-delay30-noise-a01/console.log>) |
| nominal-no-noise-a01 | completed | nominal-no-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-no-noise-a01/result.json>) / [日志](<raw/nominal-no-noise-a01/console.log>) |
| nominal-training-noise-a01 | completed | nominal-training-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-training-noise-a01/result.json>) / [日志](<raw/nominal-training-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`1101d72d83afd812ff780dee00da96da878c2892e6e6ee68f8a7ac75bf1c6c4f`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-0bbdaeac937a877a443b10c3d7013fe801b8fce75f868d8a05fb119ddd25979b.json](<../../provenance/scenario-0bbdaeac937a877a443b10c3d7013fe801b8fce75f868d8a05fb119ddd25979b.json>)
- [场景来源 scenario-14e95b3d2cec57230bd4940164300f600bf138977dc436e9f8c0fbdf56b96b55.json](<../../provenance/scenario-14e95b3d2cec57230bd4940164300f600bf138977dc436e9f8c0fbdf56b96b55.json>)
- [场景来源 scenario-27b8a334aaa6c83c8386450d127ec465e9ad2444221b8dcab9b7ccd51dc8ede2.json](<../../provenance/scenario-27b8a334aaa6c83c8386450d127ec465e9ad2444221b8dcab9b7ccd51dc8ede2.json>)
- [场景来源 scenario-a0d5942f380d6fcfd34ea8b6e6c2422c16d9756c69691fd29732a4d59310d7a5.json](<../../provenance/scenario-a0d5942f380d6fcfd34ea8b6e6c2422c16d9756c69691fd29732a4d59310d7a5.json>)
- [场景来源 scenario-a3b853b86945e4d1bac7acd2d254fc763ca54e2e52ce82f483a1130653d09243.json](<../../provenance/scenario-a3b853b86945e4d1bac7acd2d254fc763ca54e2e52ce82f483a1130653d09243.json>)
- [场景来源 scenario-c0260e519763583fc0ddcdb1ffaa5a264690944fc8fb6665dd0feab662cf6ec5.json](<../../provenance/scenario-c0260e519763583fc0ddcdb1ffaa5a264690944fc8fb6665dd0feab662cf6ec5.json>)
- [训练上下文 context-38a50c3610d7b191acd00b852ed66a027cff24dd507445209fda373a3adad85f.json](<../../provenance/context-38a50c3610d7b191acd00b852ed66a027cff24dd507445209fda373a3adad85f.json>)
- [训练有效配置 config-589ab57be35e67198ec2ad47c3fbc2f9a3af72618d6dd9ede503225de10a192d.json](<../../provenance/config-589ab57be35e67198ec2ad47c3fbc2f9a3af72618d6dd9ede503225de10a192d.json>)

- `right-delay15-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.34893463597153174, "max_tilt": 0.11224406957626343, "termination_rate": 0.0, "tracking_xy_rmse": 0.07759143704327641, "tracking_yaw_rmse": 0.25123971790235144}`
- `right-delay15-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, -0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `left-delay15-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.3829781792380593, "max_tilt": 0.11224406957626343, "termination_rate": 0.0, "tracking_xy_rmse": 0.07519991275305919, "tracking_yaw_rmse": 0.22548448628673892}`
- `left-delay15-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, 0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `right-delay30-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.46837267875671384, "max_tilt": 0.20099514722824097, "termination_rate": 0.0, "tracking_xy_rmse": 0.08392753455066966, "tracking_yaw_rmse": 0.24784874188848544}`
- `right-delay30-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, -0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 6, "scene.robot.actuators.foots.min_delay": 6, "scene.robot.actuators.legs.max_delay": 6, "scene.robot.actuators.legs.min_delay": 6, "scene.robot.actuators.wheels.max_delay": 6, "scene.robot.actuators.wheels.min_delay": 6, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `left-delay30-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.46837267875671384, "max_tilt": 0.20099514722824097, "termination_rate": 0.0, "tracking_xy_rmse": 0.08227678187181207, "tracking_yaw_rmse": 0.22966616850820962}`
- `left-delay30-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, 0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 6, "scene.robot.actuators.foots.min_delay": 6, "scene.robot.actuators.legs.max_delay": 6, "scene.robot.actuators.legs.min_delay": 6, "scene.robot.actuators.wheels.max_delay": 6, "scene.robot.actuators.wheels.min_delay": 6, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.22642199993133544, "max_tilt": 0.08061669766902924, "termination_rate": 0.0, "tracking_xy_rmse": 0.043897212967621994, "tracking_yaw_rmse": 0.04351922725971098}`
- `nominal-no-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.2521742329452977, "max_tilt": 0.09437724947929382, "termination_rate": 0.0, "tracking_xy_rmse": 0.04716243373478142, "tracking_yaw_rmse": 0.04355842023075738}`
- `nominal-training-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 最终 checkpoint model_49999.pt（内部下一迭代计数50000）；六个固定用例总计30000控制步、600仿真秒、单环境、seed42、无视频。实际启动命令由用户确认：bash scripts/start_roa.sh --terrain Flat --type wheel；未 resume。
- 本轮实际配置：站距连续奖励 std=0.1、权重3；双髋原始动作L1惩罚 -0.05；站距带 [.496,.536]m；腿延迟0–30ms；关节速度均匀噪声±1.5rad/s；恢复速度估计并从第一轮使用估计速度。
- 对比基线为10月2日45000策略的同场景已校验结果；旧策略推理次数0。额外的旧实际髋角、力矩与路径统计只读取已归档遥测。
- 30ms用例固定腿、足、轮的执行延迟均为30ms；足与轮超过训练延迟上限15ms，属于额外压力测试。
- 四个运动用例的指标算术平均（非合并RMSE）：

|阶段|XY误差旧→新 m/s|yaw误差旧→新 rad/s|双髋5–25Hz速度RMS旧→新 rad/s|
|---|---|---|---|
|straight_11_40s|0.0922→0.0813|0.1354→0.1407|0.1538→0.1858|
|turn_steady_42_70s|0.0927→0.0857|0.4056→0.3811|0.1418→0.1930|
|stop_steady_72_90s|0.0482→0.0539|0.0503→0.0576|0.0831→0.1244|

- 零指令稳态40–120s与运动停止后72–90s；净位移是区间起终点距离，并非运动路程。

|场景|平均XY速度旧→新 m/s|净位移旧→新 m|实际右/左髋均值旧→新 °|右/左力矩均值旧→新 Nm|
|---|---|---|---|---|
|right-delay15-noise|0.0559→0.0669|0.080→0.211|+0.32/-1.13→+0.93/-0.98|-22.56/+22.54→-12.54/+12.19|
|left-delay15-noise|0.0562→0.0613|0.180→0.103|+0.36/-1.20→+0.86/-0.92|-23.73/+23.60→-11.66/+11.04|
|right-delay30-noise|0.0701→0.0667|0.180→0.086|+0.46/-1.07→+0.58/-0.68|-22.73/+22.57→-9.15/+8.68|
|left-delay30-noise|0.0592→0.0682|0.153→0.132|+0.46/-0.92→+0.92/-0.95|-22.06/+21.84→-12.44/+12.01|
|nominal-no-noise|0.0003→0.0596|0.000→1.045|+0.53/+0.16→+1.15/-1.15|-16.77/+16.08→-15.57/+15.17|
|nominal-training-noise|0.0520→0.0598|0.681→0.740|+0.36/-1.17→+0.97/-1.02|-23.91/+23.81→-12.91/+12.53|

- 同一批Sep24实机输入，full_current 2×2目标Jacobian最大奇异值中位数；物理单位，局部静态敏感度。

|策略|起振Gq|增长Gq|起振Gdq|增长Gdq|
|---|---|---|---|---|
|DWAQ_old|4.4579|2.0701|0.082770|0.123151|
|ROA_0923_final|3.7812|3.5072|0.007451|0.077865|
|ROA_0928_final_no_velocity_scale0125|2.3168|2.1019|0.007420|0.056717|
|ROA_1001_model49999_noise15|0.5280|2.0929|0.002625|0.019087|
|ROA_1002_model45000_delay30|0.9502|1.6514|0.001184|0.016643|
|ROA_1003_final_std01|0.9954|1.5023|0.001801|0.017361|

- 站距 std=.1 与髋原始动作惩罚=-.05同时变化；本次对比不能把差异唯一归因于站距 std。奖励水平不是收敛或实机稳定证明。
- 原始实机历史动作和观测共享，旧策略可能遇到分布外历史；静态表不等于新策略在实机上已测试。仿真仅一个seed，遥测没有轮体位置与接触力，不能声称直接验证站距或接触稳定。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- right-delay15-noise直行20–40s实际右/左髋均值 [1.079647284094062, -1.1096871352969664]°；双髋外偏模态OLS斜率 -0.189°/min；各10s均值 [{'start_s': 20.0, 'mean': 1.1197921293456037}, {'start_s': 30.0, 'mean': 1.0695422900454254}]。
- left-delay15-noise直行20–40s实际右/左髋均值 [1.079647284094062, -1.1096871352969664]°；双髋外偏模态OLS斜率 -0.189°/min；各10s均值 [{'start_s': 20.0, 'mean': 1.1197921293456037}, {'start_s': 30.0, 'mean': 1.0695422900454254}]。
- right-delay30-noise直行20–40s实际右/左髋均值 [0.9989821901639232, -1.0229437351873496]°；双髋外偏模态OLS斜率 0.280°/min；各10s均值 [{'start_s': 20.0, 'mean': 1.0028471057054007}, {'start_s': 30.0, 'mean': 1.0190788196458733}]。
- left-delay30-noise直行20–40s实际右/左髋均值 [0.9989821901639232, -1.0229437351873496]°；双髋外偏模态OLS斜率 0.280°/min；各10s均值 [{'start_s': 20.0, 'mean': 1.0028471057054007}, {'start_s': 30.0, 'mean': 1.0190788196458733}]。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [metrics.json](<../../evidence/analysis/native-final-comparison-20261004-001/metrics.json>)
- [motion-standing-hip-comparison.png](<../../evidence/analysis/native-final-comparison-20261004-001/motion-standing-hip-comparison.png>)
- [metrics.json](<../../evidence/analysis/static-rightturn-model49999-20261004-001/metrics.json>)
- [hip-sensitivity-comparison.png](<../../evidence/analysis/static-rightturn-model49999-20261004-001/hip-sensitivity-comparison.png>)
- [native-assessment-controller-20261004-001.json](<../../evidence/source/native-assessment-controller-20261004-001.json>)
- [assessment-preflight-20261004-001.json](<../../evidence/source/assessment-preflight-20261004-001.json>)
- [final-training-log-summary-20261004-001.json](<../../evidence/training/final-training-log-summary-20261004-001.json>)

## 建议与待授权事项

- 先检查本站距收紧后的实际双髋偏置、持续力矩与跟踪权衡，结合仿真对比选择候选；当前证据不支持直接继续收紧到 std=.05。
- 下一步采用保持其余配置一致的单因素对照，报告实际髋角、力矩、速度跟踪与零指令漂移，代码和参数修改仍须另行批准。

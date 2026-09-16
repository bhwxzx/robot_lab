# 评估批次 native-assess-20260916-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-09-14_22-50-49`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| baseline-no-push-a01 | completed | baseline-no-push，2100 / 4 / 42 | complete | [结果](<raw/baseline-no-push-a01/result.json>) / [日志](<raw/baseline-no-push-a01/console.log>) |
| current-push-a01 | completed | current-push，2100 / 4 / 42 | complete | [结果](<raw/current-push-a01/result.json>) / [日志](<raw/current-push-a01/console.log>) |
| nominal-no-noise-a01 | completed | nominal-no-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-no-noise-a01/result.json>) / [日志](<raw/nominal-no-noise-a01/console.log>) |
| nominal-training-noise-a01 | completed | nominal-training-noise，6000 / 1 / 42 | complete | [结果](<raw/nominal-training-noise-a01/result.json>) / [日志](<raw/nominal-training-noise-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`2ceff27d14d1df0c878e05cf1ff831b056f36853fde787b78aa0733a39523119`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-28b4fb03f1caec457df893e3122cdda8b712c636b8de9460242e2493b0d68649.json](<../../provenance/scenario-28b4fb03f1caec457df893e3122cdda8b712c636b8de9460242e2493b0d68649.json>)
- [场景来源 scenario-7ea528bb3627cf48c25df3643b6f92dba9d085c99795e21624dde5f69d5188d5.json](<../../provenance/scenario-7ea528bb3627cf48c25df3643b6f92dba9d085c99795e21624dde5f69d5188d5.json>)
- [场景来源 scenario-82202821b5b264be0e2d8f501c5a5a5f27433b2c25ac9777af1e40688fd9cff9.json](<../../provenance/scenario-82202821b5b264be0e2d8f501c5a5a5f27433b2c25ac9777af1e40688fd9cff9.json>)
- [场景来源 scenario-8495c961180ce5f976c1b3fcb5d8da92ea93834c92b088d72c7ac8817c440af6.json](<../../provenance/scenario-8495c961180ce5f976c1b3fcb5d8da92ea93834c92b088d72c7ac8817c440af6.json>)
- [训练上下文 context-7a16aca18b98d3792386894490e923ea7d7998de3b0d5e7b0b09d28618c58382.json](<../../provenance/context-7a16aca18b98d3792386894490e923ea7d7998de3b0d5e7b0b09d28618c58382.json>)
- [训练有效配置 config-658f114711e7abfcbe65619acbd58e82f2462bf73095be9db9185323543ee5a4.json](<../../provenance/config-658f114711e7abfcbe65619acbd58e82f2462bf73095be9db9185323543ee5a4.json>)

- `baseline-no-push-a01` 指标：`{"max_joint_velocity_utilization": 0.6946263746781782, "max_tilt": 0.22794921696186066, "termination_rate": 0.0, "tracking_xy_rmse": 0.181773778721122, "tracking_yaw_rmse": 0.25906567211311404}`
- `baseline-no-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot": null}`。
- `current-push-a01` 指标：`{"max_joint_velocity_utilization": 0.954399686871153, "max_tilt": 0.38159650564193726, "termination_rate": 0.0, "tracking_xy_rmse": 0.2525991387619896, "tracking_yaw_rmse": 0.286649203964305}`
- `current-push-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 299, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 599, "start_step": 300}, {"command": [1.0, 0, 0], "end_step": 899, "start_step": 600}, {"command": [-0.5, 0, 0], "end_step": 1199, "start_step": 900}, {"command": [0.5, 0, 0.6], "end_step": 1499, "start_step": 1200}, {"command": [0.5, 0, -0.6], "end_step": 1799, "start_step": 1500}, {"command": [0, 0, 0], "end_step": 2099, "start_step": 1800}]`；训练配置覆盖：`{"episode_length_s": 60.0, "events.randomize_push_robot.params.velocity_range": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}}`。
- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.1642421086629232, "max_tilt": 0.0649469867348671, "termination_rate": 0.0, "tracking_xy_rmse": 0.03172055597977351, "tracking_yaw_rmse": 0.029062106170051682}`
- `nominal-no-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.232918406977798, "max_tilt": 0.08121123164892197, "termination_rate": 0.0, "tracking_xy_rmse": 0.051882836867826604, "tracking_yaw_rmse": 0.04772667451406608}`
- `nominal-training-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

# 评估批次 wheel-flat-final-delay30-20261003-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-10-02_14-30-02`。

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

- checkpoint SHA-256：`a34a087b57b6453157820a23eb101610786805aa0b32aee3f308f1ba46235acb`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-345ff007c165cacf4a2984801581dea2de58af1f0569f2cd1a9c8e93bf4e8e06.json](<../../provenance/scenario-345ff007c165cacf4a2984801581dea2de58af1f0569f2cd1a9c8e93bf4e8e06.json>)
- [场景来源 scenario-4e3890ad874e661b8d244215714430e0fc8f25a3a24caf46117abbc8084c0b61.json](<../../provenance/scenario-4e3890ad874e661b8d244215714430e0fc8f25a3a24caf46117abbc8084c0b61.json>)
- [场景来源 scenario-5d0a6b5b60dbd6c30207ffd777d098aa790a24b8e5fbde18e303a93b91378292.json](<../../provenance/scenario-5d0a6b5b60dbd6c30207ffd777d098aa790a24b8e5fbde18e303a93b91378292.json>)
- [场景来源 scenario-5e3299d8ec5d142110c815fa1c5ce768cde9d5beffdf222f9bb542dc61aa7c82.json](<../../provenance/scenario-5e3299d8ec5d142110c815fa1c5ce768cde9d5beffdf222f9bb542dc61aa7c82.json>)
- [场景来源 scenario-a6bbef86a19a97729db508d7ff959c7d5a06eeaae825971a980a9e53e2cba5b7.json](<../../provenance/scenario-a6bbef86a19a97729db508d7ff959c7d5a06eeaae825971a980a9e53e2cba5b7.json>)
- [场景来源 scenario-fbc1c4d2de7fbcffa1bd070e3661805154ba6cca8c8bd5e8b8b39e47ae258b50.json](<../../provenance/scenario-fbc1c4d2de7fbcffa1bd070e3661805154ba6cca8c8bd5e8b8b39e47ae258b50.json>)
- [训练上下文 context-8725ba0fd4fea12981095536d2e69ab4eff6607ab008d1517b807ab9d6400065.json](<../../provenance/context-8725ba0fd4fea12981095536d2e69ab4eff6607ab008d1517b807ab9d6400065.json>)
- [训练有效配置 config-dfa33f8c8d272c7a177f09ebd1cf9167d2a9413f6b398785dba55e8953adec94.json](<../../provenance/config-dfa33f8c8d272c7a177f09ebd1cf9167d2a9413f6b398785dba55e8953adec94.json>)

- `right-delay15-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.4013301386977687, "max_tilt": 0.13602174818515778, "termination_rate": 0.0, "tracking_xy_rmse": 0.06196252854800012, "tracking_yaw_rmse": 0.2802438019404488}`
- `right-delay15-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, -0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `left-delay15-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.32944440841674805, "max_tilt": 0.13602174818515778, "termination_rate": 0.0, "tracking_xy_rmse": 0.07015600690627613, "tracking_yaw_rmse": 0.23939289050054516}`
- `left-delay15-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, 0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `right-delay30-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.46837267875671384, "max_tilt": 0.20689892768859863, "termination_rate": 0.0, "tracking_xy_rmse": 0.06490512100534111, "tracking_yaw_rmse": 0.27545180080178144}`
- `right-delay30-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, -0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 6, "scene.robot.actuators.foots.min_delay": 6, "scene.robot.actuators.legs.max_delay": 6, "scene.robot.actuators.legs.min_delay": 6, "scene.robot.actuators.wheels.max_delay": 6, "scene.robot.actuators.wheels.min_delay": 6, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `left-delay30-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.46837267875671384, "max_tilt": 0.20689892768859863, "termination_rate": 0.0, "tracking_xy_rmse": 0.0726599314779749, "tracking_yaw_rmse": 0.24457569140308513}`
- `left-delay30-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}, {"command": [0.5, 0, 0], "end_step": 1999, "start_step": 500}, {"command": [0.5, 0, 0.6], "end_step": 3499, "start_step": 2000}, {"command": [0, 0, 0], "end_step": 4499, "start_step": 3500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 120.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 6, "scene.robot.actuators.foots.min_delay": 6, "scene.robot.actuators.legs.max_delay": 6, "scene.robot.actuators.legs.min_delay": 6, "scene.robot.actuators.wheels.max_delay": 6, "scene.robot.actuators.wheels.min_delay": 6, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-no-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.16058475971221925, "max_tilt": 0.06476787477731705, "termination_rate": 0.0, "tracking_xy_rmse": 0.03821160550530393, "tracking_yaw_rmse": 0.014023692281455951}`
- `nominal-no-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `nominal-training-noise-a01` 指标：`{"max_joint_velocity_utilization": 0.24916963866262726, "max_tilt": 0.08750902116298676, "termination_rate": 0.0, "tracking_xy_rmse": 0.039532730279529554, "tracking_yaw_rmse": 0.037743889533480034}`
- `nominal-training-noise-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 5999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 150.0, "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- Final checkpoint internal iteration50000; Native student, one env, seed42, no video.
- Training schedule completed; convergence remains indeterminate without approved criteria.
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [final-preflight-model49999-20261003-001.json](<../../evidence/source/final-preflight-model49999-20261003-001.json>)
- [metrics.json](<../../evidence/analysis/static-rightturn-model49999-20261003-001/metrics.json>)
- [final-evaluation-recovery-20261003-001.json](<../../evidence/source/final-evaluation-recovery-20261003-001.json>)
- [final-evaluation-controller-20261003-001.json](<../../evidence/source/final-evaluation-controller-20261003-001.json>)
- [metrics.json](<../../evidence/analysis/final-model49999-batch-20261003-001/metrics.json>)
- [final-evaluation-integrity-20261003-001.json](<../../evidence/training/final-evaluation-integrity-20261003-001.json>)
- [comparison.png](<../../evidence/analysis/final-model49999-batch-20261003-001/comparison.png>)
- [timeline.png](<../../evidence/analysis/final-model49999-batch-20261003-001/timeline.png>)
- [standing.png](<../../evidence/analysis/final-model49999-batch-20261003-001/standing.png>)
- [samples-and-jacobians.npz](<../../evidence/analysis/static-rightturn-model49999-20261003-001/samples-and-jacobians.npz>)
- [per-frame-comparison.csv](<../../evidence/analysis/static-rightturn-model49999-20261003-001/per-frame-comparison.csv>)
- [comparison.png](<../../evidence/analysis/static-rightturn-model49999-20261003-001/comparison.png>)

## 建议与待授权事项

- 按匹配历史对照检查转向与停车；修改方案仍需用户批准。

## 分阶段分析与匹配历史对照

六场景按历史完全相同的覆盖、指令、seed、环境数和分段窗口执行；新推理只针对当前最终 checkpoint。表中转弯稳态为 42–70 s，停车稳态为 72–90 s。世界航向和 body yaw-rate 分别记录在原始指标。

| 场景 | 直行 mean vx | 转弯 mean vx | 转弯 mean yaw-rate | yaw RMSE | 停车净位移 (18 s) | 双髋速度 5–25 Hz RMS |
|---|---:|---:|---:|---:|---:|---:|
| right-delay15-noise | 0.446 | 0.463 | -0.180 | 0.461 | 0.396 | 0.16865879411479498 |
| left-delay15-noise | 0.446 | 0.409 | +0.258 | 0.383 | 0.288 | 0.14731002191062179 |
| right-delay30-noise | 0.446 | 0.450 | -0.197 | 0.446 | 0.454 | 0.16721527666716066 |
| left-delay30-noise | 0.446 | 0.404 | +0.253 | 0.390 | 0.207 | 0.1591658406913049 |

速度单位 m/s，yaw-rate 单位 rad/s，位移 m。指令直行/转弯 vx=0.5，yaw-rate 分别为 0 / ±0.6。位移跨 reset 时为 unknown；首尾端点间隔比名义样本窗口短 0.02 s。

| 零指令站立 40–120 s | mean vx (m/s) | 本次净位移 (m) | 10/1 最终模型净位移 (m) |
|---|---:|---:|---:|
| nominal-no-noise | -0.000023 | 0.012001497265011327 | 0.00011640472628598329 |
| nominal-training-noise | +0.019223 | 1.5225435435804848 | 1.2497769900278384 |

四场景指标均值相对10/1最终模型：XY 跟踪误差 -20.1%，yaw 跟踪误差 +9.4%，双髋5–25Hz速度 RMS -23.0%，髋目标步差 RMS -51.2%，身体 roll/pitch角速度 RMS -21.9%。这是各场景统计量的算术均值变化，不是合并RMSE，也不是多seed统计证据。
闭环改善主要在 XY 跟踪和双髋/身体波动；左右转 yaw 跟踪仍不足且较旧最终模型退化。停车与噪声零指令漂移需按表和时序图分别评估，不能由目标平滑推导闭环稳定。
转弯场景完整记录 teacher/student 附加诊断，并核验 student动作与控制动作一致、teacher速度输入与预动作truth一致；两个站立场景沿用历史协议，未采集这些附加诊断。站立的动作、身体状态、关节和限值信号完整，附加诊断保持 unavailable。
汇总初次封存因错误要求站立附加诊断触发断言；六个Native进程均已exit0、原始结果有效。本报告仅从这些结果恢复封存，未重新执行任何策略推理。原控制器与恢复控制器的精确源码分别保留在 supporting evidence。

[完整闭环窗口、诊断与历史变化](<../../evidence/analysis/final-model49999-batch-20261003-001/metrics.json>)
[转弯匹配对照图](<../../evidence/analysis/final-model49999-batch-20261003-001/comparison.png>)
[右转时序图](<../../evidence/analysis/final-model49999-batch-20261003-001/timeline.png>)
[站立保持图](<../../evidence/analysis/final-model49999-batch-20261003-001/standing.png>)

## 历史实机输入双髋敏感度

使用 9/24 右转 GetDown 前 505 帧；主对齐 state lag1 / previous-action lag1，并检查三个相邻对齐。新模型 CPU 单线程推理 49490 个输入向量；旧 DWAQ 和五个旧 ROA 的数据复用。下表为相同增长段20帧；Gq/Gdq 为双髋物理目标对髋位置/速度观测局部 Jacobian 最大奇异值中位数，单位 rad/rad 和 rad/(rad/s)。

| 模型 | Gq | Gdq | 目标峰值 (rad) | 目标步差 RMS (rad) |
|---|---:|---:|---:|---:|
| DWAQ_old | 2.070 | 0.1232 | 1.810 | 1.071 |
| ROA_0923_final | 3.507 | 0.0779 | 1.348 | 0.909 |
| ROA_0926_final_scale0125 | 2.818 | 0.0301 | 0.604 | 0.437 |
| ROA_0928_final_no_velocity_scale0125 | 2.102 | 0.0567 | 0.853 | 0.563 |
| ROA_0929_model49999_estvel | 2.736 | 0.1477 | 2.167 | 1.251 |
| ROA_1001_model49999_noise15 | 2.093 | 0.0191 | 0.539 | 0.351 |
| ROA_1002_model49999_delay30 | 1.363 | 0.0196 | 0.453 | 0.241 |

相对10/1最终模型，Gq 比值 0.651，Gdq 比值 1.027；Gdq 相邻对齐范围 [0.6307957592328669, 1.0265722254377918]，因此不能宣称速度敏感度稳健下降。目标峰值/步差 RMS 比值分别 0.839/0.685。
冻结当前髋 actor 通路、冻结全历史髋、置零速度码的增长段目标步差 RMS 改变分别为 -6.02%、+9.62%、+12.37%。
历史记录保留旧策略 previous action；旧髋比例0.25，当前0.125，DWAQ历史5帧、ROA10帧。输入时序/传感器龄期仍有不确定性。这是冻结输入分析，不是当前策略的实机闭环，也不支持单参数因果归因。
[静态数据、输入与计算源码](<../../evidence/analysis/static-rightturn-model49999-20261003-001/metrics.json>)
[历史输入双髋对照图](<../../evidence/analysis/static-rightturn-model49999-20261003-001/comparison.png>)

训练最后 TensorBoard step49999、checkpoint internal iter50000、目标50000且训练进程退出，支持已完成计划训练；没有批准的收敛阈值，不作收敛判断。相关脏源码/YAML/旧证据hash未变。本次未改源码/参数、未控制训练、未实机复现、未导出/部署、未安装/删除用户文件、未提交推送。

仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

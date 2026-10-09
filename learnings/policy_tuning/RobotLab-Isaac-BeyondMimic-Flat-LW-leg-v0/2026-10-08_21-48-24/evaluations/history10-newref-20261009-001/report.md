# 评估批次 history10-newref-20261009-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-08_21-48-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| legacy-randomized-no-push-a01 | completed | legacy-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/legacy-randomized-no-push-a01/result.json>) / [日志](<raw/legacy-randomized-no-push-a01/console.log>) |
| current-randomized-no-push-a01 | completed | current-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/current-randomized-no-push-a01/result.json>) / [日志](<raw/current-randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`505e713f83d0e925d29889839c7b8774c7aca1055e32c3e13814313079a1b507`；runner：`OnPolicyRunner`。
- [场景来源 scenario-2437f0c2c1a55befbe7153d46b90bf8286b5376f3c31bd63db859e502c4c764a.json](<../../provenance/scenario-2437f0c2c1a55befbe7153d46b90bf8286b5376f3c31bd63db859e502c4c764a.json>)
- [场景来源 scenario-57a41491d849048f735e01f1c6dc3642764239ccecc0ef9b5db8cd51118a2e83.json](<../../provenance/scenario-57a41491d849048f735e01f1c6dc3642764239ccecc0ef9b5db8cd51118a2e83.json>)
- [场景来源 scenario-64d7ccf0a2963330a952b03476aaf0fce9a71ab677b4eeaaf877b6426a4035c7.json](<../../provenance/scenario-64d7ccf0a2963330a952b03476aaf0fce9a71ab677b4eeaaf877b6426a4035c7.json>)
- [场景来源 scenario-e0f45ecbd5867b5d0af2d2ed42c2cda8d494531757535ef398eb21103bbcd25f.json](<../../provenance/scenario-e0f45ecbd5867b5d0af2d2ed42c2cda8d494531757535ef398eb21103bbcd25f.json>)
- [场景来源 scenario-f2061951b2a20fa80a50762509fe55f7880116042bfc8593280f56f0998dbf80.json](<../../provenance/scenario-f2061951b2a20fa80a50762509fe55f7880116042bfc8593280f56f0998dbf80.json>)
- [训练上下文 context-90b7446755ca742a4ba366b6fa557c9959974dffc20a7e526a3fbc22c6be7f1b.json](<../../provenance/context-90b7446755ca742a4ba366b6fa557c9959974dffc20a7e526a3fbc22c6be7f1b.json>)
- [训练有效配置 config-6bc3906349a38b20aca8bdde3f6e31cf111ec38e6277c0db84c1d1ec95f9b329.json](<../../provenance/config-6bc3906349a38b20aca8bdde3f6e31cf111ec38e6277c0db84c1d1ec95f9b329.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `legacy-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `legacy-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_others": null}`。
- `current-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `current-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8f5dec53148b5229b28055b8af968d18dc757e873f1a04bd423b0e8bfc6b2a6b", "events.push_robot": null}`。

## 观察与限制

- 本次5场景按用户批准预算执行；源脚本确认命令为bash scripts/start_beyondmimic.sh --type leg。
- 训练提交ba742c5与评估提交0eb1b88不同，已核验相关Leg训练输入字节一致；新增Wheel工作保持原状。
- 完整20ms控制遥测与0–169帧全部5ms物理遥测保留；历史批次不重跑、不覆盖。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [evaluation-preflight-20261009-001.json](<../../evidence/training/evaluation-preflight-20261009-001.json>)
- [evaluation-analysis-driver-20261009-001.json](<../../evidence/training/evaluation-analysis-driver-20261009-001.json>)
- [manifest.json](<../../../2026-10-06_11-58-04/evaluations/history10-landing-contact-20261006-001/manifest.json>)
- [manifest.json](<../../../2026-10-07_00-51-40/evaluations/torque-power-history10-20261007-001/manifest.json>)
- [manifest.json](<../../../2026-10-07_15-57-49/evaluations/torque-margin-history10-20261008-001/manifest.json>)
- [case_metrics_comparison.csv](<plots/case_metrics_comparison.csv>)
- [joint_effort_comparison.csv](<plots/joint_effort_comparison.csv>)
- [phase_metrics_comparison.csv](<plots/phase_metrics_comparison.csv>)
- [cycle_metrics_comparison.csv](<plots/cycle_metrics_comparison.csv>)
- [contact_timing_comparison.csv](<plots/contact_timing_comparison.csv>)
- [posture_comparison.png](<plots/posture_comparison.png>)
- [posture_comparison.svg](<plots/posture_comparison.svg>)
- [effort_impact_by_case.png](<plots/effort_impact_by_case.png>)
- [effort_impact_by_case.svg](<plots/effort_impact_by_case.svg>)
- [phase_load_comparison.png](<plots/phase_load_comparison.png>)
- [phase_load_comparison.svg](<plots/phase_load_comparison.svg>)
- [analysis-audit.json](<plots/analysis-audit.json>)
- [evaluation-impact-20261009-001.json](<../../evidence/training/evaluation-impact-20261009-001.json>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

## 完整周期姿态与负载分析

主统计只使用从第0帧到参考末帧完整结束的成功周期；失败及截断尾段单列。以下均为描述统计。

| 运行 / 条件 | 完整周期 | roll RMS/峰值 deg | 足力矩RMS/峰值 Nm | 小腿力矩RMS/峰值 Nm | 足/小腿速度峰值 rad/s | 绝对机械功率均值 W | 落地双轮Fz峰值 kN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-08_21-48-24 / play-equivalent | 5 | 1.248/4.162 | 2.10/18.66 | 22.89/99.83 | 10.00/11.74 | 99.56 | 4.533 |
| 2026-10-08_21-48-24 / zero-delay | 5 | 1.262/3.561 | 2.13/17.70 | 22.73/99.90 | 7.50/10.22 | 99.94 | 2.485 |
| 2026-10-08_21-48-24 / fixed-15ms | 5 | 1.134/3.984 | 2.09/17.25 | 23.39/98.47 | 8.61/12.52 | 101.80 | 4.531 |
| 2026-10-08_21-48-24 / legacy-randomized-no-push | 5 | 2.606/13.338 | 4.16/21.05 | 25.50/120.00 | 8.59/10.90 | 117.20 | 2.463 |
| 2026-10-08_21-48-24 / current-randomized-no-push | 5 | 2.042/7.883 | 4.50/27.00 | 26.83/108.33 | 8.85/12.24 | 122.45 | 3.031 |
| 2026-10-06_11-58-04 / play-equivalent | 5 | 0.921/3.753 | 1.84/16.95 | 24.67/114.37 | 8.81/17.64 | 130.36 | 2.108 |
| 2026-10-06_11-58-04 / zero-delay | 5 | 0.956/3.362 | 1.84/18.15 | 24.47/114.44 | 5.16/16.62 | 128.12 | 1.959 |
| 2026-10-06_11-58-04 / fixed-15ms | 5 | 0.953/3.319 | 1.82/18.32 | 24.82/111.15 | 6.69/17.70 | 134.08 | 1.707 |
| 2026-10-06_11-58-04 / randomized-no-push | 5 | 1.910/6.095 | 5.47/27.00 | 28.69/120.00 | 19.23/17.37 | 156.71 | 2.841 |
| 2026-10-07_00-51-40 / play-equivalent | 5 | 1.951/4.711 | 2.19/15.76 | 23.54/115.44 | 7.31/17.37 | 112.89 | 2.730 |
| 2026-10-07_00-51-40 / zero-delay | 5 | 2.004/4.149 | 1.93/13.78 | 23.27/115.53 | 4.47/14.92 | 109.47 | 1.963 |
| 2026-10-07_00-51-40 / fixed-15ms | 5 | 2.081/4.902 | 2.38/15.54 | 24.16/114.68 | 5.77/18.11 | 119.07 | 2.327 |
| 2026-10-07_00-51-40 / legacy-randomized-no-push | 5 | 2.236/6.405 | 5.27/27.00 | 28.63/120.00 | 15.66/17.19 | 148.20 | 2.177 |
| 2026-10-07_00-51-40 / current-randomized-no-push | 5 | 2.091/6.733 | 5.19/27.00 | 25.19/116.41 | 10.77/16.07 | 135.85 | 3.062 |
| 2026-10-07_15-57-49 / play-equivalent | 5 | 1.191/4.501 | 2.72/20.44 | 23.54/116.22 | 10.00/17.20 | 119.29 | 4.590 |
| 2026-10-07_15-57-49 / zero-delay | 5 | 1.101/4.641 | 2.67/20.42 | 23.32/114.85 | 7.18/15.32 | 116.23 | 1.889 |
| 2026-10-07_15-57-49 / fixed-15ms | 5 | 1.233/4.088 | 2.74/20.04 | 24.14/117.15 | 8.42/18.00 | 127.66 | 2.740 |
| 2026-10-07_15-57-49 / legacy-randomized-no-push | 5 | 2.378/8.936 | 5.10/27.00 | 27.90/120.00 | 10.11/19.19 | 152.95 | 2.494 |
| 2026-10-07_15-57-49 / current-randomized-no-push | 5 | 2.775/12.320 | 5.57/27.00 | 27.19/120.00 | 9.97/17.85 | 155.86 | 3.112 |

## 图表与明细

- [case_metrics_comparison.csv](plots/case_metrics_comparison.csv)
- [joint_effort_comparison.csv](plots/joint_effort_comparison.csv)
- [phase_metrics_comparison.csv](plots/phase_metrics_comparison.csv)
- [cycle_metrics_comparison.csv](plots/cycle_metrics_comparison.csv)
- [contact_timing_comparison.csv](plots/contact_timing_comparison.csv)
- [posture_comparison.png](plots/posture_comparison.png)
- [posture_comparison.svg](plots/posture_comparison.svg)
- [effort_impact_by_case.png](plots/effort_impact_by_case.png)
- [effort_impact_by_case.svg](plots/effort_impact_by_case.svg)
- [phase_load_comparison.png](plots/phase_load_comparison.png)
- [phase_load_comparison.svg](plots/phase_load_comparison.svg)

- [计算方法与校验回执](plots/analysis-audit.json)。参考计划腾空窗口：旧版本1.30–1.55s，新版本1.30–1.60s；实际腾空由四个接触传感器Fz<10N识别，5ms采样。

## 判断范围

最新策略训练使用10/8参考、10帧590维输入，髋角归零权重−1；全程足/小腿L2、功率、torque-margin权重为0。训练末1000轮身体位置误差由1.97cm升至2.22cm；随机起始训练的motion_finished不是整套动作成功率。
历史10/6、10/7两批只复用已校验原始遥测。轨迹、奖励及随机化配置有变化，表中差异是整体方案差异，不能归因于单一奖励项。力矩是执行器驱动力矩；被动冲击不能由其峰值代替。5ms接触峰值是刚完成物理步内的平均法向接触力。
完整遥测不等于训练收敛或普遍鲁棒性；当前没有获批准的收敛判据。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

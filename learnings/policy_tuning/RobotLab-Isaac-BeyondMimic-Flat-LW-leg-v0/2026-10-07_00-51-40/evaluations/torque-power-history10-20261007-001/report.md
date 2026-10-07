# 评估批次 torque-power-history10-20261007-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-leg-v0`；训练：`2026-10-07_00-51-40`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-equivalent-a01 | completed | play-equivalent，1000 / 1 / 42 | complete | [结果](<raw/play-equivalent-a01/result.json>) / [日志](<raw/play-equivalent-a01/console.log>) |
| zero-delay-a01 | completed | zero-delay，1000 / 1 / 42 | complete | [结果](<raw/zero-delay-a01/result.json>) / [日志](<raw/zero-delay-a01/console.log>) |
| fixed-15ms-a01 | completed | fixed-15ms，1000 / 1 / 42 | complete | [结果](<raw/fixed-15ms-a01/result.json>) / [日志](<raw/fixed-15ms-a01/console.log>) |
| legacy-randomized-no-push-a01 | completed | legacy-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/legacy-randomized-no-push-a01/result.json>) / [日志](<raw/legacy-randomized-no-push-a01/console.log>) |
| current-randomized-no-push-a01 | completed | current-randomized-no-push，1000 / 1 / 42 | complete | [结果](<raw/current-randomized-no-push-a01/result.json>) / [日志](<raw/current-randomized-no-push-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`2380037696242f615aded679ba6f3f2f3270d74474f8e2490d0c9841e5f27925`；runner：`OnPolicyRunner`。
- [场景来源 scenario-3e68fd7366658d3112534e93fc1cc5baff976670cc9e642a46e3e1f6e1dbb6ef.json](<../../provenance/scenario-3e68fd7366658d3112534e93fc1cc5baff976670cc9e642a46e3e1f6e1dbb6ef.json>)
- [场景来源 scenario-48ed0ea444f1ba335997c127ee55d1b0a1eb5a323fe7a8e49d121dbecf424e34.json](<../../provenance/scenario-48ed0ea444f1ba335997c127ee55d1b0a1eb5a323fe7a8e49d121dbecf424e34.json>)
- [场景来源 scenario-50e9e75a1c6515818f3b6aedbad8925eb16ea6cd2d00f30f506a711d486e41bc.json](<../../provenance/scenario-50e9e75a1c6515818f3b6aedbad8925eb16ea6cd2d00f30f506a711d486e41bc.json>)
- [场景来源 scenario-ad1bb696227cf47af56e0d9766b6fd377aaa0d3a0824c0e2faed0945b551a56f.json](<../../provenance/scenario-ad1bb696227cf47af56e0d9766b6fd377aaa0d3a0824c0e2faed0945b551a56f.json>)
- [场景来源 scenario-ada5ca152e772701c6ba0bc05124560d7fd905da7da3d65a002b14f08d50ca60.json](<../../provenance/scenario-ada5ca152e772701c6ba0bc05124560d7fd905da7da3d65a002b14f08d50ca60.json>)
- [训练上下文 context-9981d6f319349c92ee28c5bf8f2f05f288c066c10443588f37ae81d80eb7f875.json](<../../provenance/context-9981d6f319349c92ee28c5bf8f2f05f288c066c10443588f37ae81d80eb7f875.json>)
- [训练有效配置 config-e5ff9e639e5ad2e852f01df73b4ad1c98009fb6a6d2f56c659d6d6253cab5ca0.json](<../../provenance/config-e5ff9e639e5ad2e852f01df73b4ad1c98009fb6a6d2f56c659d6d6253cab5ca0.json>)

- `play-equivalent-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-equivalent-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false}`。
- `zero-delay-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `zero-delay-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 0, "scene.robot.actuators.foots.min_delay": 0, "scene.robot.actuators.legs.max_delay": 0, "scene.robot.actuators.legs.min_delay": 0, "scene.robot.actuators.wheels.max_delay": 0, "scene.robot.actuators.wheels.min_delay": 0}`。
- `fixed-15ms-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-15ms-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3}`。
- `legacy-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `legacy-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_others": null}`。
- `current-randomized-no-push-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `current-randomized-no-push-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 166], "evaluation.motion_sha256": "9ff53e159acedd59f89179c75f08e1e2521af3573e267f0192d60e07586b3ec1", "events.push_robot": null}`。

## 观察与限制

- 获批准的固定五组预算：每组seed42、env1、1000控制步、600秒超时，无视频。
- 5ms驱动力矩/速度与完整动作周期分别统计；机器数据、峰值邻域和相位图保留于plots。
- 保全核验：312个旧证据与17个源码/参考文件；变化列表见analysis-audit.json。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [evaluation-preflight-20261007-001.json](<../../evidence/training/evaluation-preflight-20261007-001.json>)
- [manifest.json](<../../../2026-10-06_11-58-04/evaluations/history10-landing-contact-20261006-001/manifest.json>)
- [analysis-audit.json](<plots/analysis-audit.json>)
- [case_metrics_comparison.csv](<plots/case_metrics_comparison.csv>)
- [contact_and_joint_peak_audit.csv](<plots/contact_and_joint_peak_audit.csv>)
- [cycle_metrics_comparison.csv](<plots/cycle_metrics_comparison.csv>)
- [effort_motion_comparison.png](<plots/effort_motion_comparison.png>)
- [effort_motion_comparison.svg](<plots/effort_motion_comparison.svg>)
- [joint_effort_comparison.csv](<plots/joint_effort_comparison.csv>)
- [phase_metrics_comparison.csv](<plots/phase_metrics_comparison.csv>)
- [randomized_phase_comparison.png](<plots/randomized_phase_comparison.png>)
- [randomized_phase_comparison.svg](<plots/randomized_phase_comparison.svg>)

## 建议与待授权事项

- 依据分阶段力矩、超速、姿态及落地峰值共同判断下一步；参数/代码更改与新增测试需单独给出具体方案获批准。
- 不将训练完成或高奖励视为收敛；当前批次不提交、不推送、不部署。

## 新旧策略的详细比较

新策略：2026-10-07_00-51-40/model_49999.pt；旧策略：2026-10-06_11-58-04/model_49999.pt。旧四组证据仅校验并复用，未重跑。前三组覆盖完全匹配；legacy DR 显式关闭新增增益与全刚体质量随机化，匹配旧物理条件；current DR 使用当前训练随机化。单一 seed、单个环境，结果不代表一般成功率。

关节出力下表采用全部 4000 个 5ms 样本，包含末尾删失周期。姿态、功率、错轮和落地指标仅使用完整成功周期。数值以“旧 → 新”展示；当前随机化只有新策略结果。

| 条件 | 完整成功/失败 | 双足τ RMS/峰值 Nm | 双小腿τ RMS/峰值 Nm | 足速度峰值 rad/s | 足超速最长 ms | 全10关节绝对机械功率均值 W |
| --- | --- | --- | --- | --- | --- | --- |
| Play | 5/0 | 1.84 → 2.22 / 18.12 → 15.76 | 24.72 → 23.66 / 114.37 → 115.44 | 8.81 → 7.31 | 0 → 0 | 130.36 → 112.89 |
| Zero delay | 5/0 | 1.84 → 1.94 / 18.15 → 13.78 | 24.49 → 23.30 / 114.44 → 115.53 | 5.16 → 4.47 | 0 → 0 | 128.12 → 109.47 |
| 15ms delay | 5/0 | 1.83 → 2.38 / 18.32 → 15.54 | 24.84 → 24.19 / 111.15 → 114.68 | 6.69 → 5.77 | 0 → 0 | 134.08 → 119.07 |
| Legacy DR | 5/0 | 5.29 → 5.13 / 27.00 → 27.00 | 28.90 → 28.60 / 120.00 → 120.00 | 19.23 → 15.66 | 10 → 10 | 156.71 → 148.20 |
| Current DR | 5/0 | 5.60 / 27.00 | 25.85 / 116.57 | 13.30 | 5 | 135.85 |

| 条件 | 空中附近后仰峰值 deg (60–79帧) | 双轮前后差峰值 cm (80–166帧) | 落地双轮合力峰值 kN (75–99帧) | 双轮持续支撑比例 % (80–166帧) | 起跳峰值高度 m |
| --- | --- | --- | --- | --- | --- |
| Play | 12.965 → 10.764 | 3.605 → 3.343 | 2.108 → 2.730 | 100.000 → 100.000 | 0.729–0.745 |
| Zero delay | 10.277 → 7.726 | 3.958 → 2.460 | 1.959 → 1.963 | 100.000 → 100.000 | 0.728–0.728 |
| 15ms delay | 13.734 → 11.581 | 2.754 → 3.610 | 1.707 → 2.327 | 100.000 → 100.000 | 0.750–0.750 |
| Legacy DR | 13.995 → 10.135 | 6.266 → 3.526 | 2.841 → 2.177 | 100.000 → 99.828 | 0.717–0.761 |
| Current DR | 13.983 | 6.782 | 3.062 | 99.483 | 0.728–0.804 |

分阶段采用参考帧：准备 0–54、蹬伸 55–64、空中 65–74、落地缓冲 75–99、轮态 80–166，落地与轮态窗口有重叠。实际离地/触地时刻见逐周期 CSV；“空中附近”窗口包含蹬伸末尾及早期落地，不能全按腾空解释。

足关节/小腿峰值与短时超速的具体位置：
- play-equivalent-a01：velocity峰值位于 episode 4、第65帧、1.315s，right_foot_joint；实加τ=-1.353Nm、请求τ=-1.353Nm、速度=-7.306rad/s。
- play-equivalent-a01：torque峰值位于 episode 3、第61帧、1.235s，right_shank_joint；实加τ=-115.442Nm、请求τ=-115.442Nm、速度=-6.503rad/s。
- zero-delay-a01：velocity峰值位于 episode 0、第63帧、1.265s，left_foot_joint；实加τ=-0.714Nm、请求τ=-0.714Nm、速度=4.466rad/s。
- zero-delay-a01：torque峰值位于 episode 0、第61帧、1.225s，right_shank_joint；实加τ=-115.534Nm、请求τ=-115.534Nm、速度=-6.364rad/s。
- fixed-15ms-a01：velocity峰值位于 episode 0、第63帧、1.280s，left_foot_joint；实加τ=0.280Nm、请求τ=0.280Nm、速度=5.771rad/s。
- fixed-15ms-a01：torque峰值位于 episode 0、第61帧、1.240s，right_shank_joint；实加τ=-114.683Nm、请求τ=-114.683Nm、速度=-6.505rad/s。
- legacy-randomized-no-push-a01：velocity峰值位于 episode 3、第16帧、0.340s，left_foot_joint；实加τ=21.508Nm、请求τ=21.508Nm、速度=15.657rad/s。
- legacy-randomized-no-push-a01：torque峰值位于 episode 0、第60帧、1.210s，right_shank_joint；实加τ=-120.000Nm、请求τ=-122.751Nm、速度=-4.828rad/s。
- current-randomized-no-push-a01：velocity峰值位于 episode 5、第15帧、0.310s，left_foot_joint；实加τ=0.291Nm、请求τ=0.291Nm、速度=13.296rad/s。
- current-randomized-no-push-a01：torque峰值位于 episode 5、第61帧、1.225s，right_shank_joint；实加τ=-116.567Nm、请求τ=-116.567Nm、速度=-6.735rad/s。

训练记录末200轮：身体位置误差 2.330→2.124cm，锚点位置误差19.269→22.730cm，关节速度误差范数4.580→5.054rad/s。训练已正常完成50,000轮；未提供获批准的收敛判据，因此收敛结论为 indeterminate。奖励权重、动作平滑和随机化共同改变，不能把结果归因于单一力矩奖励。

力矩为执行器 PD 实加驱动力矩，不能代表被动碰撞载荷。绝对机械功率Σ|τ·qdot|不等于电池功耗；控制奖励读取20ms末帧，不能保证捕获5ms尖峰。硬力矩限幅与速度约束分别解释，观察到速度超过10rad/s不意味着驱动力矩限幅失效。

仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

# 评估批次 wheel-to-leg-reference-control-20261024-003

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-22_11-01-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-20000-a02 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-20000-a02/result.json>) / [日志](<raw/play-ground-20000-a02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`26774757655698e0e883013c43fb9280588feea665f8f54290d632ab59e70e8f`；runner：`OnPolicyRunner`。
- [场景来源 scenario-07850d6fb04248de6ff4b300b5a7831e60d69ca145ec912dbbdf8b045ec30d03.json](<../../provenance/scenario-07850d6fb04248de6ff4b300b5a7831e60d69ca145ec912dbbdf8b045ec30d03.json>)
- [训练上下文 context-ab1926263b39282d14a30d34fe2f9ce3bfb366737e17a605c910a91fe4d32a93.json](<../../provenance/context-ab1926263b39282d14a30d34fe2f9ce3bfb366737e17a605c910a91fe4d32a93.json>)
- [训练有效配置 config-2b8138a9140d4981c43c123827efa551ad634ed8a5541ffdef4da69136065cd8.json](<../../provenance/config-2b8138a9140d4981c43c123827efa551ad634ed8a5541ffdef4da69136065cd8.json>)

- `play-ground-20000-a02` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-20000-a02` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.motion_file": "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/wheel_to_leg/2026-10-20/wheel_to_leg_transform_50hz.npz", "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。

## 观察与限制

- 本批次为旧参考 20000 轮的有界对照，显式覆盖 commands.motion.motion_file 为该运行保存的旧参考路径；此前 001 因遗漏路径覆盖被 motion hash 校验拦截，失败日志保留，不作为有效动作证据。
- 保存 YAML 完整语义比较：新旧 agent 配置相同，env 仅参考轨迹路径和日志目录不同；腿部 delay 均 0–3。测试匹配 20000 轮、1000 控制步、单环境、seed 42、固定 frame-zero 开始；关闭观测噪声与事件随机化，保留执行器延迟。
- 旧参考 20000：完整 frame-zero 周期 5，非 motion_finished 终止 0；平均竖直速度峰值 0.770 m/s，参考 2.262 m/s，峰值比例 34.1%；速度 RMSE 1.247 m/s，高度 RMSE 11.26 cm；无记录支撑时长均值 45.0 ms；大腿触地 5/5，连杆法向峰值 3.706 kN，落地对地竖直合力峰值 3.733 kN。
- 旧参考 20000 准备阶段：俯仰峰值 4.42°，横滚 4.08°，髋镜像残差峰值 11.61°，双轮 ≥1 N 比例均值 90.9%。
- 新参考 20000：完整 frame-zero 周期 5，非 motion_finished 终止 0；平均竖直速度峰值 0.735 m/s，参考 1.967 m/s，峰值比例 37.3%；速度 RMSE 1.003 m/s，高度 RMSE 11.33 cm；无记录支撑时长均值 48.0 ms；大腿触地 5/5，连杆法向峰值 3.123 kN，落地对地竖直合力峰值 3.393 kN。
- 新参考 20000 准备阶段：俯仰峰值 4.13°，横滚 4.02°，髋镜像残差峰值 7.14°，双轮 ≥1 N 比例均值 100.0%。
- 新参考 49999（复用）：完整 frame-zero 周期 5，非 motion_finished 终止 0；平均竖直速度峰值 0.604 m/s，参考 1.967 m/s，峰值比例 30.7%；速度 RMSE 1.027 m/s，高度 RMSE 11.82 cm；无记录支撑时长均值 56.0 ms；大腿触地 5/5，连杆法向峰值 5.583 kN，落地对地竖直合力峰值 5.583 kN。
- 新参考 49999（复用） 准备阶段：俯仰峰值 8.40°，横滚 3.09°，髋镜像残差峰值 11.53°，双轮 ≥1 N 比例均值 80.2%。
- 新参考速度与高度目标不同，误差改善不自动等于跳跃更强；相同最终 49999 轮的旧参考策略不存在，本对照不能替代该证据。
- 没有批准的收敛判据；仿真回放不证明硬件安全。代码、保存参数、模型与持久设置均未修改。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-10-23_17-15-35/evaluations/wheel-to-leg-reference-control-20261024-002/manifest.json>)
- [manifest.json](<../../../2026-10-23_17-15-35/evaluations/wheel-to-leg-latest-20261024-001/manifest.json>)
- [manifest.json](<../wheel-to-leg-reference-control-20261024-001/manifest.json>)
- [reference-comparison.png](<derived/reference-comparison.png>)
- [reference-comparison.json](<derived/reference-comparison.json>)

## 建议与待授权事项

- 结合匹配轮数的起跳、腾空、姿态与大腿触地共同判断；不依据单个奖励或 RMSE 自动调整训练参数。

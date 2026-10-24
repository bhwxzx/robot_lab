# 评估批次 wheel-to-leg-reference-control-20261024-002

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-23_17-15-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-20000-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-20000-a01/result.json>) / [日志](<raw/play-ground-20000-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`5ce238ec3ee06efdc2ab81214d04c742577fb5530e9f5721c7c4f3a0e1152303`；runner：`OnPolicyRunner`。
- [场景来源 scenario-ebd221c60e7e404c940398f9ee21f53eda0916325ad298dfe22a2dc84d8e7da9.json](<../../provenance/scenario-ebd221c60e7e404c940398f9ee21f53eda0916325ad298dfe22a2dc84d8e7da9.json>)
- [训练上下文 context-31fe3312130c166323eabe787896949a86485ca19e8e7c82501665fa73443d70.json](<../../provenance/context-31fe3312130c166323eabe787896949a86485ca19e8e7c82501665fa73443d70.json>)
- [训练有效配置 config-ee803ef0c412062bd850db8a4c15ea80ea0339fcbf91322934748b94e4d6c844.json](<../../provenance/config-ee803ef0c412062bd850db8a4c15ea80ea0339fcbf91322934748b94e4d6c844.json>)

- `play-ground-20000-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-20000-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。

## 观察与限制

- 按用户问题进行参考轨迹对照；两个训练运行保存的 agent.yaml 完全相同，env.yaml 仅 motion_file 和日志目录不同；腿部 delay 均为 0–3。使用各自 model_20000.pt 匹配训练轮数。
- 新旧均采用固定 frame-zero 开始、1000 控制步、1 环境、seed 42，无观测噪声和事件随机化；保留执行器延迟；物理步 5 ms；不录视频、不训练、不修改代码。
- 本运行 20000 轮：完整周期 5，非 motion_finished 终止 0；平均竖直峰值 0.735 m/s，对应参考 1.967 m/s，比例 37.3%；速度 RMSE 1.003 m/s，高度 RMSE 11.33 cm；连续无记录支撑均值 48.0 ms。
- 准备阶段俯仰峰值 4.13°、横滚 4.02°、髋镜像残差 7.14°；双轮 ≥1 N 比例均值 100.0%。大腿触地 5/5；单连杆峰值 3.123 kN，落地记录竖直合力峰值 3.393 kN。
- 新旧参考目标不同，不能将更低跟踪误差直接视作更强起跳；单 seed 不能证明普遍成功率，未提供批准的收敛判据。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [reference-comparison.json](<derived/reference-comparison.json>)

## 建议与待授权事项

- 结合相同训练轮数的起跳、姿态、接触和冲击共同判断；本次仅做有限测试，不调整奖励参数。

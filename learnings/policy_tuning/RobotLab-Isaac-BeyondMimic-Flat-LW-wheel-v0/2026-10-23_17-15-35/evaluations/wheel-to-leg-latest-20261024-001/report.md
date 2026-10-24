# 评估批次 wheel-to-leg-latest-20261024-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-23_17-15-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`753736869951f7ca4921802914be2cf9b290c8cd99d622b01eccbe6dca21d3c2`；runner：`OnPolicyRunner`。
- [场景来源 scenario-7256e78806d886355904454d02d05d861b782c261cf871fc1d69e2d9da0c1250.json](<../../provenance/scenario-7256e78806d886355904454d02d05d861b782c261cf871fc1d69e2d9da0c1250.json>)
- [场景来源 scenario-ebd221c60e7e404c940398f9ee21f53eda0916325ad298dfe22a2dc84d8e7da9.json](<../../provenance/scenario-ebd221c60e7e404c940398f9ee21f53eda0916325ad298dfe22a2dc84d8e7da9.json>)
- [训练上下文 context-31fe3312130c166323eabe787896949a86485ca19e8e7c82501665fa73443d70.json](<../../provenance/context-31fe3312130c166323eabe787896949a86485ca19e8e7c82501665fa73443d70.json>)
- [训练有效配置 config-ee803ef0c412062bd850db8a4c15ea80ea0339fcbf91322934748b94e4d6c844.json](<../../provenance/config-ee803ef0c412062bd850db8a4c15ea80ea0339fcbf91322934748b94e4d6c844.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- play-ground-a01：1000 控制步 / 4000 物理步；motion_finished 完成片段 5，非 motion_finished 终止 0，末尾未完成片段 1。随机相位的完成片段不能当作完整起跳。
- 固定从 frame 0 完成 5 个周期：平均竖直速度峰值 0.604 m/s，实际 NPZ 参考峰值 1.967 m/s（1.28 s）；1.20–1.30 s 速度 RMSE 1.027 m/s，1.20–1.70 s 高度 RMSE 11.82 cm；记录接触全部低于 1 N 的最长连续时长均值 56.0 ms。
- 准备阶段（排除最初 20 ms）双轮同时 ≥1 N 比例均值 80.2%；单轮 <1 N 最长右/左 120/85 ms，双轮同时 <1 N 最长 5 ms。失载不等同已测得明显抬轮高度，缺少烘焙碰撞形状的接触间隙。
- 准备阶段机身俯仰绝对峰值 8.40°（参考 1.75°），横滚 3.09°；髋镜像残差峰值 11.53°。蹬伸横滚峰值 4.08°，髋镜像残差峰值 11.12°。
- play-ground-a01 全程：足端同时竖直合力峰值 1.477 kN；1.40–2.00 s 全部记录对地竖直合力峰值 5.583 kN，约 21.29 倍本组重力（质量 26.73 kg）。大腿/小腿单连杆峰值 5.583 kN，参考时间 1.525 s；大腿接触完成片段 5/5，尾段 1。
- play-ground-a01 驱动力矩达到配置限幅：未记录到。速度超过保存配置名义上限：未记录到。这些是仿真驱动统计，不能证明碰撞关节反力矩或硬件热负载安全。
- training-random-phase-a01：1000 控制步 / 4000 物理步；motion_finished 完成片段 12，非 motion_finished 终止 0，末尾未完成片段 1。随机相位的完成片段不能当作完整起跳。
- training-random-phase-a01 全程：足端同时竖直合力峰值 1.574 kN；1.40–2.00 s 全部记录对地竖直合力峰值 5.967 kN，约 21.81 倍本组重力（质量 27.89 kg）。大腿/小腿单连杆峰值 5.277 kN，参考时间 1.600 s；大腿接触完成片段 4/12，尾段 1。
- training-random-phase-a01 驱动力矩达到配置限幅：未记录到。速度超过保存配置名义上限：left_foot_joint 10.26/10 rad/s；right_foot_joint 10.07/10 rad/s。这些是仿真驱动统计，不能证明碰撞关节反力矩或硬件热负载安全。
- 训练日志完成至 iteration 49998；末尾两段各 1000 轮的统计保存在分析 JSON。没有用户批准的收敛判据，收敛结论为 indeterminate；训练随机起始相位下的 motion_finished 比例不能替代完整跳跃成功率。
- 新旧参考轨迹、训练配置均有变化；本批次匹配测试种子、步数、环境数和覆盖规格，比较只能作描述性比较，不能归因于单项奖励或轨迹改动。
- 5 ms 接触力为物理步平均法向力，排除摩擦；驱动力矩不含碰撞被动反力矩。单 seed、无视频的有限回放不能证明普遍成功率或硬件安全。
- 检查点、保存 YAML、现有导出、参考轨迹及 Isaac Sim 用户设置字节均保持不变，用户设置 mtime 未变化。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [training-origin-20261024-001.json](<../../evidence/training/training-origin-20261024-001.json>)
- [training-tail-20261024-001.json](<../../evidence/training/training-tail-20261024-001.json>)
- [joint-torque-speed-play-ground-a01.png](<derived/joint-torque-speed-play-ground-a01.png>)
- [tracking-preparation.png](<derived/tracking-preparation.png>)
- [landing-play-ground-a01.png](<derived/landing-play-ground-a01.png>)
- [joint-torque-speed-training-random-phase-a01.png](<derived/joint-torque-speed-training-random-phase-a01.png>)
- [landing-training-random-phase-a01.png](<derived/landing-training-random-phase-a01.png>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [manifest.json](<../../../2026-10-22_17-40-29/evaluations/wheel-to-leg-latest-20261023-001/manifest.json>)
- [manifest.json](<../../../2026-10-22_17-40-29/evaluations/wheel-to-leg-mechanics-20261023-002/manifest.json>)

## 建议与待授权事项

- 按准备阶段支撑与机身/髋姿态、完整跳跃跟踪、落地连杆触地和关节速度共同判断；当前证据不支持自动调整参数。
- 下一步如需扩大验证，应使用额外种子的有界回放和新批次；本次未启动训练或修改代码。

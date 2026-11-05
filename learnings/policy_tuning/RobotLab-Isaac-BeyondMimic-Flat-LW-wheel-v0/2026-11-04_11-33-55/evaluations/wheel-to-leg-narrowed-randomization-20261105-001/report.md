# 评估批次 wheel-to-leg-narrowed-randomization-20261105-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-04_11-33-55`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| narrowed-a-initial-a01 | completed | narrowed-a-initial，1000 / 1 / 42 | complete | [结果](<raw/narrowed-a-initial-a01/result.json>) / [日志](<raw/narrowed-a-initial-a01/console.log>) |
| narrowed-b-initial-physics-a01 | completed | narrowed-b-initial-physics，1000 / 1 / 42 | complete | [结果](<raw/narrowed-b-initial-physics-a01/result.json>) / [日志](<raw/narrowed-b-initial-physics-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`facc6f107806efbf1ca9e488b9498108fd8a0f80e8e4f289b917d3983f9f2373`；runner：`OnPolicyRunner`。
- [场景来源 scenario-39efa0185247880bb36b553f8aa09e4ea20a8fdfc8e8b5f256f6790c627aab91.json](<../../provenance/scenario-39efa0185247880bb36b553f8aa09e4ea20a8fdfc8e8b5f256f6790c627aab91.json>)
- [场景来源 scenario-e186ca6a599cf5e87687f749fe7a3ada4b516c25f2de12aa4b669590fe35edc5.json](<../../provenance/scenario-e186ca6a599cf5e87687f749fe7a3ada4b516c25f2de12aa4b669590fe35edc5.json>)
- [训练上下文 context-e4d02f485b16ecb51ad9b6d9359d58540acec0d7801acf1ccc6c0d088822e8dc.json](<../../provenance/context-e4d02f485b16ecb51ad9b6d9359d58540acec0d7801acf1ccc6c0d088822e8dc.json>)
- [训练有效配置 config-1f5227e89a34dcbc5f56911f10e043fbf5df7b966aeb8c90fd7b4aa2adb8a893.json](<../../provenance/config-1f5227e89a34dcbc5f56911f10e043fbf5df7b966aeb8c90fd7b4aa2adb8a893.json>)

- `narrowed-a-initial-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `narrowed-a-initial-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [-0.03, 0.03], "commands.motion.pose_range": {"pitch": [-0.03, 0.03], "roll": [-0.03, 0.03], "x": [-0.01, 0.01], "y": [-0.01, 0.01], "yaw": [-0.05, 0.05], "z": [-0.005, 0.005]}, "commands.motion.velocity_range": {"pitch": [-0.1, 0.1], "roll": [-0.1, 0.1], "x": [-0.1, 0.1], "y": [-0.1, 0.1], "yaw": [-0.1, 0.1], "z": [-0.05, 0.05]}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_anchor_roll_horizontal.weight": 0.0}`。
- `narrowed-b-initial-physics-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `narrowed-b-initial-physics-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [-0.03, 0.03], "commands.motion.pose_range": {"pitch": [-0.03, 0.03], "roll": [-0.03, 0.03], "x": [-0.01, 0.01], "y": [-0.01, 0.01], "yaw": [-0.05, 0.05], "z": [-0.005, 0.005]}, "commands.motion.velocity_range": {"pitch": [-0.1, 0.1], "roll": [-0.1, 0.1], "x": [-0.1, 0.1], "y": [-0.1, 0.1], "yaw": [-0.1, 0.1], "z": [-0.05, 0.05]}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.base_com.params.com_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]}, "events.physics_material.params.dynamic_friction_range": [0.6, 1.0], "events.physics_material.params.restitution_range": [0.0, 0.1], "events.physics_material.params.static_friction_range": [0.7, 1.2], "events.randomize_actuator_gains.params.damping_distribution_params": [0.9, 1.1], "events.randomize_actuator_gains.params.stiffness_distribution_params": [0.9, 1.1], "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [0.0, 1.0], "events.randomize_rigid_body_mass_others.params.mass_distribution_params": [0.95, 1.05], "rewards.motion_anchor_roll_horizontal.weight": 0.0}`。

## 观察与限制

- 用户于2026-11-05批准两组收窄评估：A只收窄初始状态；B在A基础上收窄物理参数。均为 Native mean action、固定frame 0、seed42、1环境、1000控制步、4000物理样本，无视频。
- A初始位置x/y ±0.01m、z ±0.005m；roll/pitch ±0.03rad、yaw ±0.05rad；线速度x/y ±0.1m/s、z ±0.05m/s；三轴角速度 ±0.1rad/s；关节初始位置 ±0.03rad。
- B额外：COM各轴±0.01m；基座附加质量0–1kg、其他连杆质量倍率0.95–1.05；PD增益倍率0.9–1.1；静摩擦0.7–1.2、动摩擦0.6–1.0、恢复系数0–0.1，make_consistent保持true。
- 观测噪声、关节零位偏差、执行延迟保留训练设置；不修改训练奖励、终止项或工作区代码。横滚水平奖励沿用训练有效值0。
- 完整训练随机化基线直接复用并重新校验本日已有结果，未重跑；A初始物性与基线逐字段核对，B摩擦和恢复系数运行时范围已核对。
- 这些是用户批准的诊断范围，未经实机标定。单seed及单startup物性抽样不能建立总体成功率；不能把分组结果归因于单个参数。
- 所有关节力矩来自驱动器applied_torque，不能代表全部碰撞载荷；速度上限用于对照配置名义值，并非硬件验收阈值。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- full-training-randomization: 完整动作=5，失败=0，截尾=1；髋镜像残差(1–1.8s)=13.480°；右髋20ms变化=2.580°；俯仰RMSE=4.563°，最大|俯仰|=18.122°，最大|横滚|=12.098°；最大anchor z误差=0.09280m，>0.15m样本=0/1000；大小腿触地峰值=0.000N；轮距=0.4613–0.5192m，越界样本=54.23%。
- full-training-randomization: 起跳窗口1–1.5s峰值向上速度=1.431m/s，基座最高=0.7276m；1.5–2.0s高度RMSE=0.03217m；完整控制/物理样本=1000/4000。
- narrowed-a-initial-a01: 完整动作=5，失败=0，截尾=1；髋镜像残差(1–1.8s)=10.724°；右髋20ms变化=2.282°；俯仰RMSE=3.400°，最大|俯仰|=13.733°，最大|横滚|=13.658°；最大anchor z误差=0.10954m，>0.15m样本=0/1000；大小腿触地峰值=0.000N；轮距=0.5031–0.5251m，越界样本=0.00%。
- narrowed-a-initial-a01: 起跳窗口1–1.5s峰值向上速度=1.494m/s，基座最高=0.7158m；1.5–2.0s高度RMSE=0.03511m；完整控制/物理样本=1000/4000。
- narrowed-b-initial-physics-a01: 完整动作=5，失败=0，截尾=1；髋镜像残差(1–1.8s)=7.896°；右髋20ms变化=2.120°；俯仰RMSE=2.881°，最大|俯仰|=14.670°，最大|横滚|=10.115°；最大anchor z误差=0.14127m，>0.15m样本=0/1000；大小腿触地峰值=0.000N；轮距=0.4926–0.5181m，越界样本=1.72%。
- narrowed-b-initial-physics-a01: 起跳窗口1–1.5s峰值向上速度=1.514m/s，基座最高=0.7245m；1.5–2.0s高度RMSE=0.03780m；完整控制/物理样本=1000/4000。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [protection-before-narrowed-20261105-001.json](<../../evidence/training/protection-before-narrowed-20261105-001.json>)
- [manifest.json](<../wheel-to-leg-fixed-start-20261105-001/manifest.json>)
- [randomization-comparison.png](<derived/randomization-comparison.png>)
- [joint-loads.csv](<derived/joint-loads.csv>)

## 建议与待授权事项

- 以收窄范围评估预期条件下的动作，同时保留完整训练随机化压力测试结果。若继续优化，先核对实际部署参数并提出具体方案，等待批准。

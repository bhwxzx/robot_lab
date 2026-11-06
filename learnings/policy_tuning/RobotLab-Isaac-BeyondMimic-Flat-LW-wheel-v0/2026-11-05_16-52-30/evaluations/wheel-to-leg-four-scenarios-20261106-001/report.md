# 评估批次 wheel-to-leg-four-scenarios-20261106-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-05_16-52-30`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |
| narrowed-a-initial-a01 | completed | narrowed-a-initial，1000 / 1 / 42 | complete | [结果](<raw/narrowed-a-initial-a01/result.json>) / [日志](<raw/narrowed-a-initial-a01/console.log>) |
| narrowed-b-initial-physics-a01 | completed | narrowed-b-initial-physics，1000 / 1 / 42 | complete | [结果](<raw/narrowed-b-initial-physics-a01/result.json>) / [日志](<raw/narrowed-b-initial-physics-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`6a382dc1575e5e0a569524a01f8e659adc6d2f1a6c15ec0974573509a69b093a`；runner：`OnPolicyRunner`。
- [场景来源 scenario-07de2481f31dbc6ff02e66f83d2283844c8b2ff622b4c010ed3ce0bc393d0bcd.json](<../../provenance/scenario-07de2481f31dbc6ff02e66f83d2283844c8b2ff622b4c010ed3ce0bc393d0bcd.json>)
- [场景来源 scenario-3a8324bb1c13801c45abcdd2acc7cf1cbd30dd0c426bfd4e2a2d55a60047157a.json](<../../provenance/scenario-3a8324bb1c13801c45abcdd2acc7cf1cbd30dd0c426bfd4e2a2d55a60047157a.json>)
- [场景来源 scenario-76c805eaa071a7069c1c60ab6fe9562178709f4cfed88856942c4194d5d437f5.json](<../../provenance/scenario-76c805eaa071a7069c1c60ab6fe9562178709f4cfed88856942c4194d5d437f5.json>)
- [场景来源 scenario-dc86c2640beda1ee94c6f2a98aea819eeb459ac19cd84670ba872d01d5ff0948.json](<../../provenance/scenario-dc86c2640beda1ee94c6f2a98aea819eeb459ac19cd84670ba872d01d5ff0948.json>)
- [训练上下文 context-bd73601117f1ceac93fe29b3b98dcccac68373ffeb5917ebd8c6fe5919fdcf79.json](<../../provenance/context-bd73601117f1ceac93fe29b3b98dcccac68373ffeb5917ebd8c6fe5919fdcf79.json>)
- [训练有效配置 config-e1864a72515805f82d95762cd3fb61a354299a6a7e318cdd7cdee51258a54437.json](<../../provenance/config-e1864a72515805f82d95762cd3fb61a354299a6a7e318cdd7cdee51258a54437.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "terminations.anchor_pos.params.threshold": 0.15}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "terminations.anchor_pos.params.threshold": 0.15}`。
- `narrowed-a-initial-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `narrowed-a-initial-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [-0.03, 0.03], "commands.motion.pose_range": {"pitch": [-0.03, 0.03], "roll": [-0.03, 0.03], "x": [-0.01, 0.01], "y": [-0.01, 0.01], "yaw": [-0.05, 0.05], "z": [-0.005, 0.005]}, "commands.motion.velocity_range": {"pitch": [-0.1, 0.1], "roll": [-0.1, 0.1], "x": [-0.1, 0.1], "y": [-0.1, 0.1], "yaw": [-0.1, 0.1], "z": [-0.05, 0.05]}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "terminations.anchor_pos.params.threshold": 0.15}`。
- `narrowed-b-initial-physics-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `narrowed-b-initial-physics-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [-0.03, 0.03], "commands.motion.pose_range": {"pitch": [-0.03, 0.03], "roll": [-0.03, 0.03], "x": [-0.01, 0.01], "y": [-0.01, 0.01], "yaw": [-0.05, 0.05], "z": [-0.005, 0.005]}, "commands.motion.velocity_range": {"pitch": [-0.1, 0.1], "roll": [-0.1, 0.1], "x": [-0.1, 0.1], "y": [-0.1, 0.1], "yaw": [-0.1, 0.1], "z": [-0.05, 0.05]}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.base_com.params.com_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]}, "events.physics_material.params.dynamic_friction_range": [0.6, 1.0], "events.physics_material.params.restitution_range": [0.0, 0.1], "events.physics_material.params.static_friction_range": [0.7, 1.2], "events.randomize_actuator_gains.params.damping_distribution_params": [0.9, 1.1], "events.randomize_actuator_gains.params.stiffness_distribution_params": [0.9, 1.1], "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [0.0, 1.0], "events.randomize_rigid_body_mass_others.params.mass_distribution_params": [0.95, 1.05], "terminations.anchor_pos.params.threshold": 0.15}`。

## 观察与限制

- 用户于2026-11-06确认本场次由 bash scripts/start_beyondmimic.sh --type wheel 启动，无额外CLI/Hydra覆盖。8192环境、50000轮、seed42、PPO/OnPolicyRunner。
- 训练HEAD=6d3ea62；保存的唯一源码差异是anchor_pos z-only阈值0.25→0.15m。已在内存重建并核对Git blob，评估通过scenario override复现。工作区源码未修改。
- 本版启用PPO镜像数据增强及mirror loss，系数0.1；横滚水平奖励0.5。相对上一版还改变终止阈值，不能将运动变化只归因于PPO镜像增强。
- 末尾1000轮：motion_finished均值0.9958808，symmetry loss均值0.0001696；采样归一化熵0.8380608、最高概率0.4397672。未保存实际采样直方图，未提供本场次批准的收敛标准，不能据训练完成或低镜像损失认定收敛/运动对称。
- 四组各1000步、1环境、seed42、Native mean action、固定从frame0开始，无视频。B为收窄初始状态及物理参数的主评估；完整训练随机化为压力测试；A与无随机化用于比较。
- A：初始位置x/y±0.01m、z±0.005m；roll/pitch±0.03rad、yaw±0.05rad；线速度x/y±0.1m/s、z±0.05m/s；三轴角速度±0.1rad/s；关节初始位置±0.03rad。
- B在A上增加：COM各轴±0.01m；基座附加质量0–1kg；其他连杆质量倍率0.95–1.05；PD增益倍率0.9–1.1；静摩擦0.7–1.2、动摩擦0.6–1.0、恢复系数0–0.1，make_consistent=true。观测噪声、关节零位偏差和执行延迟沿用训练。
- 旧版四组结果直接复用已校验的2026-11-04_11-33-55证据，未重复运行。所有比较的参考相位、窗口和统计公式一致，但终止阈值和训练奖励不同。
- 单seed、单startup物性抽样；重复周期不能推断总体成功率。这些收窄范围未经实物标定；完整遥测不能替代实物验证。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整动作=5, 失败=0, 截尾=1；髋残差(1–1.8s)=0.835°（旧3.936°）；右髋20ms变化=0.780°；俯仰RMSE=3.316°，最大|俯仰|=12.425°，最大|横滚|=1.266°（旧8.467°）。
- fixed-start-nominal-a01: 最大anchor z误差=0.05360m，>0.15m=0/1000；最大xy误差=0.0728m；大小腿触地峰值=0.000N；双轮阶段轮距=0.5160–0.5193m，越界物理样本=0.00%。
- fixed-start-nominal-a01: 起跳1–1.5s峰值向上速度=1.710m/s，最高基座=0.6888m；1.5–2.0s高度RMSE=0.02812m。
- fixed-start-nominal-a01: 关节负载异常对照：未观察到驱动力矩饱和或超过配置名义速度。驱动力矩不含全部被动碰撞载荷，配置名义值不是硬件验收阈值。
- fixed-start-randomized-a01: 完整动作=5, 失败=0, 截尾=1；髋残差(1–1.8s)=8.114°（旧13.480°）；右髋20ms变化=2.636°；俯仰RMSE=2.952°，最大|俯仰|=13.309°，最大|横滚|=6.577°（旧12.098°）。
- fixed-start-randomized-a01: 最大anchor z误差=0.08286m，>0.15m=0/1000；最大xy误差=0.3457m；大小腿触地峰值=0.000N；双轮阶段轮距=0.4580–0.5215m，越界物理样本=42.00%。
- fixed-start-randomized-a01: 起跳1–1.5s峰值向上速度=2.053m/s，最高基座=0.7015m；1.5–2.0s高度RMSE=0.03126m。
- fixed-start-randomized-a01: 关节负载异常对照：left_foot_joint:驱动力矩峰值27.000N·m,饱和5物理样本,速度峰值9.433rad/s,超过名义速度0样本；right_foot_joint:驱动力矩峰值21.193N·m,饱和0物理样本,速度峰值10.199rad/s,超过名义速度4样本。驱动力矩不含全部被动碰撞载荷，配置名义值不是硬件验收阈值。
- narrowed-a-initial-a01: 完整动作=5, 失败=0, 截尾=1；髋残差(1–1.8s)=11.618°（旧10.724°）；右髋20ms变化=3.283°；俯仰RMSE=3.060°，最大|俯仰|=13.368°，最大|横滚|=9.957°（旧13.658°）。
- narrowed-a-initial-a01: 最大anchor z误差=0.09406m，>0.15m=0/1000；最大xy误差=0.2397m；大小腿触地峰值=0.000N；双轮阶段轮距=0.5042–0.5207m，越界物理样本=0.00%。
- narrowed-a-initial-a01: 起跳1–1.5s峰值向上速度=2.019m/s，最高基座=0.7059m；1.5–2.0s高度RMSE=0.02003m。
- narrowed-a-initial-a01: 关节负载异常对照：left_foot_joint:驱动力矩峰值24.571N·m,饱和0物理样本,速度峰值10.051rad/s,超过名义速度1样本；right_foot_joint:驱动力矩峰值25.808N·m,饱和0物理样本,速度峰值11.648rad/s,超过名义速度2样本。驱动力矩不含全部被动碰撞载荷，配置名义值不是硬件验收阈值。
- narrowed-b-initial-physics-a01: 完整动作=5, 失败=0, 截尾=1；髋残差(1–1.8s)=3.133°（旧7.896°）；右髋20ms变化=1.740°；俯仰RMSE=2.748°，最大|俯仰|=12.670°，最大|横滚|=3.013°（旧10.115°）。
- narrowed-b-initial-physics-a01: 最大anchor z误差=0.09345m，>0.15m=0/1000；最大xy误差=0.1959m；大小腿触地峰值=0.000N；双轮阶段轮距=0.5032–0.5188m，越界物理样本=0.00%。
- narrowed-b-initial-physics-a01: 起跳1–1.5s峰值向上速度=1.865m/s，最高基座=0.6963m；1.5–2.0s高度RMSE=0.02437m。
- narrowed-b-initial-physics-a01: 关节负载异常对照：left_foot_joint:驱动力矩峰值27.000N·m,饱和1物理样本,速度峰值8.832rad/s,超过名义速度0样本。驱动力矩不含全部被动碰撞载荷，配置名义值不是硬件验收阈值。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [manifest.json](<../../../2026-11-04_11-33-55/evaluations/wheel-to-leg-fixed-start-20261105-001/manifest.json>)
- [manifest.json](<../../../2026-11-04_11-33-55/evaluations/wheel-to-leg-narrowed-randomization-20261105-001/manifest.json>)
- [four-scenario-phases.png](<derived/four-scenario-phases.png>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [training-origin-20261106-001.json](<../../evidence/training/training-origin-20261106-001.json>)
- [training-windows-20261106-001.json](<../../evidence/training/training-windows-20261106-001.json>)
- [config-diff-20261106-001.json](<../../evidence/training/config-diff-20261106-001.json>)
- [protection-before-20261106-001.json](<../../evidence/training/protection-before-20261106-001.json>)
- [training-source-20261106-001.json](<../../evidence/source/training-source-20261106-001.json>)
- [training-source-20261106-001.patch](<../../evidence/source/training-source-20261106-001.patch>)

## 建议与待授权事项

- 优先审阅收窄物性B组的起跳髋/横滚、落地高度和关节冲击，同时保留完整训练随机化压力测试。进一步参数调整须先给具体方案并获批准。

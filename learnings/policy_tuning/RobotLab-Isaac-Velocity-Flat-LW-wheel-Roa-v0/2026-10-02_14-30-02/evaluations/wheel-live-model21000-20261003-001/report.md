# 评估批次 wheel-live-model21000-20261003-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0`；训练：`2026-10-02_14-30-02`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| quick-right15-a01 | failed | quick-right15-noise，2000 / 1 / 42 | — | — / [日志](<raw/quick-right15-a01/console.log>) |
| quick-right15-cpu-a02 | completed | quick-right15-noise-cpu，2000 / 1 / 42 | complete | [结果](<raw/quick-right15-cpu-a02/result.json>) / [日志](<raw/quick-right15-cpu-a02/console.log>) |
| quick-right15-gpu1t-a03 | failed | quick-right15-noise，2000 / 1 / 42 | — | — / [日志](<raw/quick-right15-gpu1t-a03/console.log>) |
| micro-right15-gpu1t-a04 | completed | micro-right15-noise，500 / 1 / 42 | complete | [结果](<raw/micro-right15-gpu1t-a04/result.json>) / [日志](<raw/micro-right15-gpu1t-a04/console.log>) |

## 策略与场景

- checkpoint SHA-256：`330ac2360e010eb4622bfd3d29f65753a14f78a266b1509d641f879e047579c8`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-2f1ebef0525a116fd3eb0294712c5d7bda10e4b332e9d7da016ee8a15698ab46.json](<../../provenance/scenario-2f1ebef0525a116fd3eb0294712c5d7bda10e4b332e9d7da016ee8a15698ab46.json>)
- [场景来源 scenario-f9db28396319e09a6adabec951c27366d0179c2c307a711b4bca9223ed73f2ae.json](<../../provenance/scenario-f9db28396319e09a6adabec951c27366d0179c2c307a711b4bca9223ed73f2ae.json>)
- [训练上下文 context-8725ba0fd4fea12981095536d2e69ab4eff6607ab008d1517b807ab9d6400065.json](<../../provenance/context-8725ba0fd4fea12981095536d2e69ab4eff6607ab008d1517b807ab9d6400065.json>)
- [训练有效配置 config-dfa33f8c8d272c7a177f09ebd1cf9167d2a9413f6b398785dba55e8953adec94.json](<../../provenance/config-dfa33f8c8d272c7a177f09ebd1cf9167d2a9413f6b398785dba55e8953adec94.json>)

- `quick-right15-a01` 命令调度：`[{"command": [0.0, 0.0, 0.0], "end_step": 249, "start_step": 0}, {"command": [0.5, 0.0, 0.0], "end_step": 749, "start_step": 250}, {"command": [0.5, 0.0, -0.6], "end_step": 1499, "start_step": 750}, {"command": [0.0, 0.0, 0.0], "end_step": 1999, "start_step": 1500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 60.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `quick-right15-a01` 未完成原因：material_training_throughput_drop；仅停止本次评估进程；没有发布完整结果，不能作为策略性能证据。
- `quick-right15-cpu-a02` 指标：`{"max_joint_velocity_utilization": 0.0, "max_tilt": 0.0, "termination_rate": 1.0, "tracking_xy_rmse": 0.2795084971874737, "tracking_yaw_rmse": 0.36742346871752496}`
- `quick-right15-cpu-a02` 命令调度：`[{"command": [0.0, 0.0, 0.0], "end_step": 249, "start_step": 0}, {"command": [0.5, 0.0, 0.0], "end_step": 749, "start_step": 250}, {"command": [0.5, 0.0, -0.6], "end_step": 1499, "start_step": 750}, {"command": [0.0, 0.0, 0.0], "end_step": 1999, "start_step": 1500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 60.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "sim.device": "cpu", "terminations.terrain_out_of_bounds": null}`。
- `quick-right15-gpu1t-a03` 命令调度：`[{"command": [0.0, 0.0, 0.0], "end_step": 249, "start_step": 0}, {"command": [0.5, 0.0, 0.0], "end_step": 749, "start_step": 250}, {"command": [0.5, 0.0, -0.6], "end_step": 1499, "start_step": 750}, {"command": [0.0, 0.0, 0.0], "end_step": 1999, "start_step": 1500}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 60.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。
- `quick-right15-gpu1t-a03` 未完成原因：material_training_throughput_drop；仅停止本次评估进程；没有发布完整结果，不能作为策略性能证据。
- `micro-right15-gpu1t-a04` 指标：`{"max_joint_velocity_utilization": 0.3422254504579486, "max_tilt": 0.14926360547542572, "termination_rate": 0.0, "tracking_xy_rmse": 0.0964328781253919, "tracking_yaw_rmse": 0.2774873294895456}`
- `micro-right15-gpu1t-a04` 命令调度：`[{"command": [0.0, 0.0, 0.0], "end_step": 49, "start_step": 0}, {"command": [0.5, 0.0, 0.0], "end_step": 199, "start_step": 50}, {"command": [0.5, 0.0, -0.6], "end_step": 349, "start_step": 200}, {"command": [0.0, 0.0, 0.0], "end_step": 499, "start_step": 350}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 60.0, "evaluation.roa_mode": "student", "events.add_joint_default_pos": null, "events.randomize_actuator_gains": null, "events.randomize_com_positions": null, "events.randomize_push_robot": null, "events.randomize_reset_base.params.pose_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_base.params.velocity_range": {"pitch": [0.0, 0.0], "roll": [0.0, 0.0], "x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0], "z": [0.0, 0.0]}, "events.randomize_reset_joints.params.position_range": [0.0, 0.0], "events.randomize_reset_joints.params.velocity_range": [0.0, 0.0], "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "events.randomize_rigid_body_material.params.dynamic_friction_range": [1.0, 1.0], "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "events.randomize_rigid_body_material.params.static_friction_range": [1.0, 1.0], "observations.policy.enable_corruption": true, "scene.robot.actuators.foots.max_delay": 3, "scene.robot.actuators.foots.min_delay": 3, "scene.robot.actuators.legs.max_delay": 3, "scene.robot.actuators.legs.min_delay": 3, "scene.robot.actuators.wheels.max_delay": 3, "scene.robot.actuators.wheels.min_delay": 3, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane", "terminations.terrain_out_of_bounds": null}`。

## 观察与限制

- 结论：当前 model21000 具备短时直行跟踪能力；本次右转窗口明显欠跟踪，停车后仍有残余运动。历史增长阶段的静态敏感度降低，但正常阶段的位置敏感度升高、目标峰值增大，不能判断实机抖动已解决。
- 证据有效性：唯一用于连续运动性能判断的用例是 micro-right15-gpu1t-a04。两个 40 秒 GPU 尝试因训练吞吐保护被停止，没有完整结果；CPU 用例虽然 completed/遥测完整，但每一步非法接触重置，根状态保持重置值，因此排除其速度、漂移、姿态指标。CPU/Fabric 路径的具体根因未定位，不能把该现象判为策略摔倒或零漂移。没有视频或实机复现。
- 对象：训练 2026-10-02_14-30-02 的开始评估时最新完整快照 model_21000.pt，内部迭代 21001；训练继续后出现更晚 checkpoint，本报告固定上述 SHA，不追逐中途更新。OnPolicyRunnerROA / ROAPPO / student，速度估计开启，从开始使用估计速度 [1,1,0,1]，10×39 oldest-first 历史。原始训练启动 HEAD 未独立证明；当前 HEAD、相关源文件哈希、原训练 YAML 与 dirty diff 块均已核实。
- 训练有效设置：关节速度零均值均匀噪声 ±1.5 rad/s；双髋动作 scale=0.125；腿部 actuator delay 0–30 ms，但 wheels/foots 仍为 0–15 ms；腿部原始动作一阶/二阶差分权重 -0.5/-0.15；双髋动作幅值惩罚停用；轮距允许区间 [0.496,0.536] m。多个设置同时改变，且当前21k与旧策略最终50k不同，不能归因到某一个参数。
- 有效短测场景：isaacsim-5.1 / Native GPU / 1 env / seed42 / 500控制步（10秒、50Hz）；固定平地、摩擦1、零位姿/速度初始化、15ms三组执行器延迟；保留训练观测噪声，关闭推力、质量、增益、COM等随机化。0–1秒零指令，1–4秒直行，4–7秒右转，7–10秒停车；下列后段窗口仍然只有2秒，不代表长期稳态能力。
- 有效 Native GPU 短测（各统计窗内无重置）：

| 窗口 | 指令 | 平均 vx (m/s) | vx RMSE (m/s) | 平均 yaw (rad/s) | yaw RMSE (rad/s) | 窗内净 XY 位移 (m) |
| --- | --- | --- | --- | --- | --- | --- |
| 直行 2–4 s | vx=0.5，yaw=0 | 0.469261 | 0.088384 | 0.026981 | 0.179835 | 0.928077 |
| 右转 5–7 s | vx=0.5，yaw=-0.6 | 0.534410 | 0.069645 | -0.193725 | 0.455905 | 1.053938 |
| 停车 7–10 s | vx=0，yaw=0 | 0.074166 | 0.133300 | 0.005023 | 0.096250 | 0.214677 |
| 停车后段 8–10 s | vx=0，yaw=0 | 0.053908 | 0.062720 | -0.004211 | 0.052718 | 0.104249 |
- 整个有效10秒：XY RMSE=0.096433 m/s（按XY两分量平均平方定义），yaw RMSE=0.277487 rad/s；最大倾角=8.552°；无终止、超时、非法接触、关节速度/力矩限值违反。10秒未崩溃不代表长时稳定或已抑制实机共振。
- 停车3秒净XY位移0.214677m，停车后段2秒净XY位移0.104249m、平均vx=0.053908m/s；只说明该短窗仍有运动，不能推断长时漂移。右转后段平均yaw=-0.193725rad/s，指令-0.6rad/s；vx仍接近0.5m/s，提示该工况应优先核查yaw响应。
- 双髋仿真目标峰值：直行0.240812rad、转向0.253064rad、停车0.361874rad；各段5–25Hz实际关节速度RMS约0.161/0.163/0.115rad/s。该频段由50Hz遥测计算，不能识别真实机械共振，未与旧策略不同长度的仿真调度作改善百分比比较。
- 速度估计诊断：直行/转向/停车后段x估计RMSE分别0.038097/0.035400/0.050480m/s；短测不能证明估计误差是停车残余运动或转向不足的原因。
- 历史 9/24 右转抖动增长窗口（20 帧，58.146–58.526 s）的冻结输入分析。Gq/Gdq 为双髋当前物理位置/速度到物理目标角度的 2×2 雅可比最大奇异值的帧间中位数；它们不是闭环增益或稳定裕度。

| 策略 | Gq (rad/rad) | Gdq (rad/(rad/s)) | 双髋目标峰值 (rad) | 目标一阶差分 RMS (rad/控制步) |
| --- | --- | --- | --- | --- |
| 旧 DWAQ 6/03 | 2.070082 | 0.12315104 | 1.809609 | 1.070514 |
| ROA 9/23 | 3.507222 | 0.07786485 | 1.348157 | 0.909026 |
| ROA 9/26 | 2.818441 | 0.03011086 | 0.604422 | 0.437249 |
| ROA 9/28 无速度估计 | 2.101930 | 0.05671677 | 0.853134 | 0.563290 |
| ROA 9/29 | 2.735893 | 0.14773811 | 2.167267 | 1.250713 |
| ROA 10/1 最终 | 2.092938 | 0.01908667 | 0.539218 | 0.350990 |
| 当前 ROA 10/2 model21000 | 1.420700 | 0.00986735 | 0.612647 | 0.298363 |
- 相比10/1最终，增长段Gq变化-32.12%、Gdq变化-48.30%、目标一阶差分RMS变化-14.99%、目标幅值峰值变化+13.62%。四种预先规定时序对齐下，增长段Gq当前/旧=.668–.717、Gdq=.388–.606；降低仅针对该历史增长段。
- 敏感度改善不覆盖所有阶段；正常阶段的位置敏感度反而升高：

| 历史输入阶段 | 10/1 最终 Gq | 当前 Gq | 当前/10/1 Gq | 10/1 最终 Gdq | 当前 Gdq |
| --- | --- | --- | --- | --- | --- |
| 零指令 | 0.216811 | 1.424960 | 6.572 | 0.00197791 | 0.00219090 |
| 直行 | 0.309648 | 1.550107 | 5.006 | 0.00205424 | 0.00228491 |
| 抖动起始 | 0.527982 | 1.521776 | 2.882 | 0.00262470 | 0.00219825 |
| 抖动增长 | 2.092938 | 1.420700 | 0.679 | 0.01908667 | 0.00986735 |
- 静态数据口径：本次仅对当前策略新增49,490个输入向量计算（505帧），旧 DWAQ/ROA 均复用已完成并校验的历史结果，旧策略新增推理次数=0。沿用各策略原生历史长度和原实机raw-action历史；当前0.125 hip scale与录制旧策略0.25不同，冻结输入未形成新策略闭环轨迹，不是当前checkpoint的实机测试。
- 图表修正：原 comparison.png 的最后两个图例误沿用 Oct01 iter33000/final49999；数值、数组、checkpoint映射未变。以 comparison-labels-corrected.png 与 plot-label-correction.json 为准，原图仅保留作可审计来源，不用于解释策略名称。
- 训练恢复检查：PID574265持续运行、未被信号干预；迭代推进到22104，最近20次迭代吞吐53405.15步/秒，恢复到评估前约53–55k水平。有效短测前后53002→42107步/秒，存在短暂影响；两次长GPU尝试也因明显干扰触发保护。最近100次迭代平均reward=134.815，平均episode length=995.335/1000控制步，已采样loss与训练指标均有限。训练随机环境下的误差均值（XY约.463、yaw约.560）与此固定场景RMSE不属于同一口径。
- 收敛状态 indeterminate：没有用户批准的收敛判据；没有完成长时漂移、多seed或30ms边界验证。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [metrics.json](<../../evidence/analysis/live-model21000-20261003-001/metrics.json>)
- [timeline.png](<../../evidence/analysis/live-model21000-20261003-001/timeline.png>)
- [metrics.json](<../../evidence/analysis/static-rightturn-model21000-20261003-001/metrics.json>)
- [samples-and-jacobians.npz](<../../evidence/analysis/static-rightturn-model21000-20261003-001/samples-and-jacobians.npz>)
- [per-frame-comparison.csv](<../../evidence/analysis/static-rightturn-model21000-20261003-001/per-frame-comparison.csv>)
- [comparison-labels-corrected.png](<../../evidence/analysis/static-rightturn-model21000-20261003-001/comparison-labels-corrected.png>)
- [plot-label-correction.json](<../../evidence/analysis/static-rightturn-model21000-20261003-001/plot-label-correction.json>)
- [launch-wheel-live-model21000-20261003-001.json](<../../evidence/source/launch-wheel-live-model21000-20261003-001.json>)
- [launch-wheel-live-model21000cpu-20261003-001.json](<../../evidence/source/launch-wheel-live-model21000cpu-20261003-001.json>)
- [launch-wheel-live-model21000gpu1t-20261003-001.json](<../../evidence/source/launch-wheel-live-model21000gpu1t-20261003-001.json>)
- [launch-wheel-live-model21000micro-20261003-001.json](<../../evidence/source/launch-wheel-live-model21000micro-20261003-001.json>)
- [overlap-monitor-wheel-live-model21000-20261003-001.json](<../../evidence/source/overlap-monitor-wheel-live-model21000-20261003-001.json>)
- [overlap-monitor-wheel-live-model21000cpu-20261003-001.json](<../../evidence/source/overlap-monitor-wheel-live-model21000cpu-20261003-001.json>)
- [overlap-monitor-wheel-live-model21000gpu1t-20261003-001.json](<../../evidence/source/overlap-monitor-wheel-live-model21000gpu1t-20261003-001.json>)
- [overlap-monitor-wheel-live-model21000micro-20261003-001.json](<../../evidence/source/overlap-monitor-wheel-live-model21000micro-20261003-001.json>)
- [health-live21000micro-recovery-20261003-001.json](<../../evidence/health/health-live21000micro-recovery-20261003-001.json>)
- [overlap-controller-sources-wheel-live-model21000-20261003-001.json](<../../evidence/source/overlap-controller-sources-wheel-live-model21000-20261003-001.json>)
- [overlap-additional-controller-sources-wheel-live-model21000-20261003-001.json](<../../evidence/source/overlap-additional-controller-sources-wheel-live-model21000-20261003-001.json>)

## 建议与待授权事项

- 保留当前 checkpoint 作为候选。训练完成或GPU空闲后，用同一套长窗口的直行、左右转、停车场景，分别复核15ms与30ms腿部延迟；补足本次10秒测不到的持续转向和停车后长期残余运动。
- 下一步筛选 checkpoint 同时查看正常阶段与历史增长阶段 Gq/Gdq、物理目标幅值和一阶差分，避免只按增长段敏感度或训练reward排名。当前正常位置敏感度升高应和转向、停车闭环表现一起判断。
- 若完整匹配评估仍显示目标幅值或正常阶段位置响应过大，可提出恢复双髋幅值约束的独立训练对照；本次证据不支持直接归因或立即改权重。任何源代码/训练参数改动须先提交具体方案并获用户批准。

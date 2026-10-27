# 评估批次 wheel-to-leg-fixed-start-20261027-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-26_13-25-00`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`9e5c55bca25a9c0ca243082a52bf7b85eb37387577326139d17ecabe9a07a6f2`；runner：`OnPolicyRunner`。
- [场景来源 scenario-57c588bd58043de94d3796d29964129b1e9cdea32b4c23982439bfb1ed5400f2.json](<../../provenance/scenario-57c588bd58043de94d3796d29964129b1e9cdea32b4c23982439bfb1ed5400f2.json>)
- [场景来源 scenario-6f12fbc7341930f26c0324af7bb205f3b4e3bb7636afca924a04088077bfa419.json](<../../provenance/scenario-6f12fbc7341930f26c0324af7bb205f3b4e3bb7636afca924a04088077bfa419.json>)
- [训练上下文 context-43c3a8d3a70ffc7094fb7f77492d5e430245c06b6fbcec01d3a9bee5f22e653f.json](<../../provenance/context-43c3a8d3a70ffc7094fb7f77492d5e430245c06b6fbcec01d3a9bee5f22e653f.json>)
- [训练有效配置 config-bb69f37ca0111f6a33b0a8fd238e29eda609ce02a47cf6dfac12e38aabeb5baa.json](<../../provenance/config-bb69f37ca0111f6a33b0a8fd238e29eda609ce02a47cf6dfac12e38aabeb5baa.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_body_lin_vel.weight": 3.0, "rewards.motion_body_ori.weight": 2.0, "rewards.motion_body_pos.weight": 2.0, "rewards.motion_global_anchor_ori.weight": 1.0, "rewards.motion_global_anchor_pos.weight": 2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_body_lin_vel.weight": 3.0, "rewards.motion_body_ori.weight": 2.0, "rewards.motion_body_pos.weight": 2.0, "rewards.motion_global_anchor_ori.weight": 1.0, "rewards.motion_global_anchor_pos.weight": 2.0}`。

## 观察与限制

- fixed-start-nominal-a01完整周期：1.20–1.70s高度RMSE均值3.094cm、范围3.094–3.094cm；全程横滚绝对峰值最大7.611°；髋/大腿/小腿/足部镜像残差峰值{"hip": 9.5940674794539, "thigh": 14.73589962721026, "shank": 37.58683153090365, "foot": 28.70845816655354}度。
- fixed-start-nominal-a01失载后再接触至参考2.0s的落地记录法向力峰值最大1858.776N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-nominal-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-nominal-a01已到达起跳窗口的6段：竖直速度峰值1.251–1.251m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-nominal-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比0.8814、饱和物理样本0；关节速度最大名义限值比0.9660。
- fixed-start-randomized-a01完整周期：1.20–1.70s高度RMSE均值2.771cm、范围1.702–4.838cm；全程横滚绝对峰值最大13.682°；髋/大腿/小腿/足部镜像残差峰值{"hip": 25.724933780009476, "thigh": 37.583896470551245, "shank": 48.68705689452244, "foot": 48.45415085891959}度。
- fixed-start-randomized-a01失载后再接触至参考2.0s的落地记录法向力峰值最大1801.859N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-randomized-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-randomized-a01已到达起跳窗口的6段：竖直速度峰值1.166–1.601m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-randomized-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比1.0000、饱和物理样本16；关节速度最大名义限值比1.5585。
- 复用并校验2026-10-25_12-18-23旧server批次作为描述性对照，见analysis.json baseline_comparison；未重跑旧策略。奖励改变的单次训练比较不证明某项权重的因果效果。
- 训练过程窗口统计见training-windows-20261027-001.json。随机起始相位下的motion_finished训练指标不等同于完整起始周期的跳跃成功率；缺少批准的收敛判据。
- 名义组nominal_start关闭噪声、重置扰动、材质/质量/COM/增益随机化并将执行延迟设为0。随机化组randomized_start仅固定参考起始相位为0，保留训练随机化和0..3步延迟；两组均为确定性策略推理。
- 测试恢复训练保存的五项跟踪权重：基座位置2、基座姿态1、连杆位置2、连杆姿态2、连杆线速度3；undesired_contacts=-0.1、joint_symmetry=0、anchor_pos高度阈值0.15m，并保留大腿/小腿触地>1N失败终止。这五项只通过场景覆盖在评估进程内恢复，未改源码；随机起始相位不参与本批次。
- 5ms物理遥测包含4个足/轮连杆净法向力和4个大腿/小腿对地法向力。八连杆失载并非全机器人腾空的直接证明，base/hip未单独测量；净力不包含摩擦。applied_torque仅为驱动输出，不包含接触被动反力矩。
- 失败终止会截断后续落地过程：不得把未观察到的高度、冲击或接触按零处理，也不能直接以较低冲击证明训练改善。名义重复不是独立种子试验，单环境startup随机化也不覆盖全部物理参数组合。
- 所有模型、训练YAML、既有导出、USD、参考NPZ、相关源码及Isaac Sim持久配置字节与mtime均未变动。实际评估argv含进程内--/app/settings/persistent=false。未启动训练、导出、部署、安装、删除或Git提交。
- 没有用户批准的收敛判据，收敛结论为indeterminate。本批次只支持有限仿真观察；仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-10-25_12-18-23/evaluations/wheel-to-leg-fixed-start-20261026-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [fixed-start-motion-contact.png](<derived/fixed-start-motion-contact.png>)
- [training-windows-20261027-001.json](<../../evidence/training/training-windows-20261027-001.json>)
- [joint-symmetry-comparison.png](<derived/joint-symmetry-comparison.png>)
- [protection-before-20261027-001.json](<../../evidence/training/protection-before-20261027-001.json>)
- [protection-after-20261027-001.json](<../../evidence/training/protection-after-20261027-001.json>)
- [training-origin-20261027-001.json](<../../evidence/training/training-origin-20261027-001.json>)

## 建议与待授权事项

- 按分段数据审查真实起跳、准备轮子支撑、姿态与落地触地终止；保持现有配置和证据，不自动调整训练参数。

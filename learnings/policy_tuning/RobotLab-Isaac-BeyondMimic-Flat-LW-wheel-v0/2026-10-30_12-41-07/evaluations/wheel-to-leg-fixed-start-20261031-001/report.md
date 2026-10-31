# 评估批次 wheel-to-leg-fixed-start-20261031-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-30_12-41-07`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`dbdacc5ce4c9e215d99aa2427ede8a047250bf7a78dc9e053d43cafd6354969c`；runner：`OnPolicyRunner`。
- [场景来源 scenario-07886534431c54b1cfc7e4f91294d435ee53e8ff33dcab0c4c831cd22bd58ef5.json](<../../provenance/scenario-07886534431c54b1cfc7e4f91294d435ee53e8ff33dcab0c4c831cd22bd58ef5.json>)
- [场景来源 scenario-50ee69898d4aa1505dfbf7e2c312645d038d1f5bd7907e3ae789d534f092e8ec.json](<../../provenance/scenario-50ee69898d4aa1505dfbf7e2c312645d038d1f5bd7907e3ae789d534f092e8ec.json>)
- [训练上下文 context-9d520a4f640f48f44c580d29ab9ab210e42fd62ebe8bfc87220dec62c21cf611.json](<../../provenance/context-9d520a4f640f48f44c580d29ab9ab210e42fd62ebe8bfc87220dec62c21cf611.json>)
- [训练有效配置 config-39b31c0b58aec0a978be7dd9fa2de4277a99c76b5ce5f53a8b1809d91788fa70.json](<../../provenance/config-39b31c0b58aec0a978be7dd9fa2de4277a99c76b5ce5f53a8b1809d91788fa70.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 本机younghit场次2026-10-30_12-41-07/model_49999.pt，PPO/OnPolicyRunner，seed42，8192环境、50000轮正常完成；保存训练配置及启动来源已核验。
- 名义和训练随机化条件，各固定frame0、1环境、1000控制步/4000物理步、Native、无视频。名义组关闭训练随机化与观测噪声、delay0；随机化组保留训练随机化与0..3步执行延迟。
- 当前源代码与训练记录diff一致：hip_pos_zero_l2=-3，joint_symmetry=-0.5，action_rate=-0.2，action_smoothness=-0.075；起跳俯仰角速度/姿态权重为-0.03/0.5；轮接触奖励0.5，end1.25s、ramp0。
- 对照上一完成本机场次2026-10-29_11-07-40：仅轮接触奖励及其窗口、训练环境数发生实质变化；PPO参数相同。8192环境使总采样步数翻倍，不能独立归因于奖励。此处仅比较两个终版checkpoint，不是历史所有checkpoint排名。
- 执行包装器仅添加--kit_args=--/app/settings/persistent=false保护Isaac持久配置，实际argv保存在console.log。actor使用环境按term展开的10帧历史590维；检查reset历史。
- 单seed、随机化一次startup抽样；名义多周期为确定性重复。完整周期统计排除未完成尾段，风险极值保留所有观测。八连杆失载不涵盖base/hip；驱动力矩不涵盖所有被动碰撞载荷。没有已批准收敛阈值。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- nominal: 完整周期5，失败终止0，未完成尾段1；遥测complete，缺失信号[]。
- nominal 完整周期均值 起跳峰速: 上一场次1.315 → 当前1.120 m/s。
- nominal 完整周期均值 腾空高度RMSE: 上一场次2.862 → 当前1.904 cm。
- nominal 完整周期均值 准备最低高度: 上一场次0.501 → 当前0.546 m。
- nominal 完整周期均值 落地最低高度: 上一场次0.455 → 当前0.461 m。
- nominal 全采样峰值：roll -6.421°；pitch -8.114°（相位1.560s）；水平anchor误差6.307cm；八连杆总Fz1.873kN；大腿/小腿地面接触0.000N，自碰0.000N。
- nominal 起跳1.0..1.5s髋镜像误差峰值：9.332 → 4.334°，当前相位1.260s，左髋5.677°、右髋-1.342°。
- nominal whole俯仰跟踪RMSE：3.021 → 2.810°。
- nominal initialization俯仰跟踪RMSE：4.082 → 4.406°。
- nominal takeoff俯仰跟踪RMSE：1.595 → 1.121°。
- nominal flight俯仰跟踪RMSE：3.060 → 3.317°。
- nominal landing俯仰跟踪RMSE：2.628 → 2.955°。
- nominal left_foot_joint: 速度峰值7.378rad/s，驱动力矩峰值15.338Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal left_hip_joint: 速度峰值1.339rad/s，驱动力矩峰值42.841Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal left_shank_joint: 速度峰值13.355rad/s，驱动力矩峰值102.180Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal left_thigh_joint: 速度峰值7.130rad/s，驱动力矩峰值27.958Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal left_wheel_joint: 速度峰值9.077rad/s，驱动力矩峰值5.022Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal right_foot_joint: 速度峰值4.295rad/s，驱动力矩峰值14.653Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal right_hip_joint: 速度峰值0.724rad/s，驱动力矩峰值25.093Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal right_shank_joint: 速度峰值5.906rad/s，驱动力矩峰值66.678Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal right_thigh_joint: 速度峰值6.337rad/s，驱动力矩峰值29.369Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- nominal right_wheel_joint: 速度峰值6.351rad/s，驱动力矩峰值4.100Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized: 完整周期5，失败终止0，未完成尾段1；遥测complete，缺失信号[]。
- randomized 完整周期均值 起跳峰速: 上一场次1.236 → 当前1.251 m/s。
- randomized 完整周期均值 腾空高度RMSE: 上一场次2.107 → 当前2.353 cm。
- randomized 完整周期均值 准备最低高度: 上一场次0.541 → 当前0.516 m。
- randomized 完整周期均值 落地最低高度: 上一场次0.461 → 当前0.450 m。
- randomized 全采样峰值：roll -12.903°；pitch -14.862°（相位0.440s）；水平anchor误差45.380cm；八连杆总Fz2.176kN；大腿/小腿地面接触0.000N，自碰0.000N。
- randomized 起跳1.0..1.5s髋镜像误差峰值：10.658 → 9.480°，当前相位1.420s，左髋4.041°、右髋5.439°。
- randomized whole俯仰跟踪RMSE：8.173 → 3.550°。
- randomized initialization俯仰跟踪RMSE：13.640 → 5.727°。
- randomized takeoff俯仰跟踪RMSE：5.946 → 2.841°。
- randomized flight俯仰跟踪RMSE：5.608 → 2.476°。
- randomized landing俯仰跟踪RMSE：2.150 → 1.376°。
- randomized left_foot_joint: 速度峰值10.027rad/s，驱动力矩峰值27.000Nm；超过模型名义速度2物理样本，力矩饱和20样本；每样本5ms，累计时长不等于连续时长。
- randomized left_hip_joint: 速度峰值3.086rad/s，驱动力矩峰值50.813Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized left_shank_joint: 速度峰值14.215rad/s，驱动力矩峰值114.449Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized left_thigh_joint: 速度峰值10.121rad/s，驱动力矩峰值44.434Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized left_wheel_joint: 速度峰值14.121rad/s，驱动力矩峰值9.767Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized right_foot_joint: 速度峰值10.316rad/s，驱动力矩峰值19.282Nm；超过模型名义速度2物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized right_hip_joint: 速度峰值4.443rad/s，驱动力矩峰值44.301Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized right_shank_joint: 速度峰值7.250rad/s，驱动力矩峰值75.690Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized right_thigh_joint: 速度峰值9.647rad/s，驱动力矩峰值44.498Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- randomized right_wheel_joint: 速度峰值11.321rad/s，驱动力矩峰值5.578Nm；超过模型名义速度0物理样本，力矩饱和0样本；每样本5ms，累计时长不等于连续时长。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [protection-before-20261031-001.json](<../../evidence/training/protection-before-20261031-001.json>)
- [training-origin-20261031-001.json](<../../evidence/training/training-origin-20261031-001.json>)
- [training-windows-20261031-001.json](<../../evidence/training/training-windows-20261031-001.json>)
- [analysis.json](<derived/analysis.json>)
- [hip-takeoff-comparison.png](<derived/hip-takeoff-comparison.png>)
- [hip-takeoff-details.json](<derived/hip-takeoff-details.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [joint-symmetry-comparison.png](<derived/joint-symmetry-comparison.png>)
- [jump-contact-comparison.png](<derived/jump-contact-comparison.png>)
- [pitch-phase-comparison.png](<derived/pitch-phase-comparison.png>)
- [pitch-stage-analysis.json](<derived/pitch-stage-analysis.json>)

## 建议与待授权事项

- 结合髋镜像误差、俯仰、起跳速度、水平漂移及足部限值评估取舍；参数调整或进一步训练须先提出具体方案并获得批准。

# 评估批次 wheel-to-leg-fixed-start-20261029-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-28_13-14-55`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c558610c3195ca01a75fa85a2a08e39b7cba712db48a057d3300140b3a32cb97`；runner：`OnPolicyRunner`。
- [场景来源 scenario-51095e0f7dbedfd19d2cdd95822dffd0eeb316a13fa443f29157411c62bb1753.json](<../../provenance/scenario-51095e0f7dbedfd19d2cdd95822dffd0eeb316a13fa443f29157411c62bb1753.json>)
- [场景来源 scenario-c11a0c725c2fdffbeec064afe8df14d6bd56a38ad740306031b6dfc9a3ef6b55.json](<../../provenance/scenario-c11a0c725c2fdffbeec064afe8df14d6bd56a38ad740306031b6dfc9a3ef6b55.json>)
- [训练上下文 context-3966992e6c52670ddc536f3fee9315c975a84092e237c0aad0729c37b5ea6326.json](<../../provenance/context-3966992e6c52670ddc536f3fee9315c975a84092e237c0aad0729c37b5ea6326.json>)
- [训练有效配置 config-6b48d9d2d1ffd8449a6cd568173a98e72fcdea717ae8bde3b5f688e59c1ef358.json](<../../provenance/config-6b48d9d2d1ffd8449a6cd568173a98e72fcdea717ae8bde3b5f688e59c1ef358.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_takeoff_anchor_ori.weight": 0.5, "rewards.motion_takeoff_pitch_ang_vel.weight": -0.03}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_takeoff_anchor_ori.weight": 0.5, "rewards.motion_takeoff_pitch_ang_vel.weight": -0.03}`。

## 观察与限制

- fixed-start-nominal-a01完整周期：1.20–1.70s高度RMSE均值2.484cm、范围2.484–2.484cm；全程横滚绝对峰值最大7.884°；髋/大腿/小腿/足部镜像残差峰值{"hip": 5.696843287871253, "thigh": 11.839091322368677, "shank": 29.87913892991539, "foot": 9.67159909115683}度。
- fixed-start-nominal-a01失载后再接触至参考2.0s的落地记录法向力峰值最大2380.912N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-nominal-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-nominal-a01已到达起跳窗口的6段：竖直速度峰值1.233–1.233m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-nominal-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比0.7538、饱和物理样本0；关节速度最大名义限值比1.0011。
- fixed-start-randomized-a01完整周期：1.20–1.70s高度RMSE均值2.873cm、范围1.447–5.264cm；全程横滚绝对峰值最大7.666°；髋/大腿/小腿/足部镜像残差峰值{"hip": 12.504257405540008, "thigh": 21.711748130802235, "shank": 54.57758022814975, "foot": 13.548458860285672}度。
- fixed-start-randomized-a01失载后再接触至参考2.0s的落地记录法向力峰值最大1796.367N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-randomized-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-randomized-a01已到达起跳窗口的6段：竖直速度峰值0.745–1.400m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-randomized-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比1.0000、饱和物理样本1；关节速度最大名义限值比1.4996。
- 复用并校验2026-10-27_20-02-47旧server批次作为描述性对照，见analysis.json baseline_comparison；未重跑旧策略。奖励改变的单次训练比较不证明某项权重的因果效果。
- fixed-start-nominal-a01与上版server同口径完整周期指标对照：{"baseline": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.5143943881939521, "height_rmse_mean_m": 0.03126309626329119, "landing_peak_mean_n": 1191.4149475097656, "landing_peak_worst_n": 1191.4149475097656, "roll_peak_worst_deg": 11.3522863369918, "mirror_peak_worst_deg": {"hip": 7.419929901066454, "thigh": 25.120371538456844, "shank": 30.442877130798852, "foot": 12.199742386018276}, "action_second_difference_mean": 0.13223713771855272}, "current": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.2330572671093543, "height_rmse_mean_m": 0.024836459126038348, "landing_peak_mean_n": 2380.9115600585938, "landing_peak_worst_n": 2380.9115600585938, "roll_peak_worst_deg": 7.883589928674293, "mirror_peak_worst_deg": {"hip": 5.696843287871253, "thigh": 11.839091322368677, "shank": 29.87913892991539, "foot": 9.67159909115683}, "action_second_difference_mean": 0.09686626259386302}}。没有完整周期的窗口为未知，不能据截断数据宣布改善。
- fixed-start-randomized-a01与上版server同口径完整周期指标对照：{"baseline": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.2923824480591874, "height_rmse_mean_m": 0.024767791684818238, "landing_peak_mean_n": 1868.0273406982421, "landing_peak_worst_n": 3248.260009765625, "roll_peak_worst_deg": 11.447698816548595, "mirror_peak_worst_deg": {"hip": 14.577236894111802, "thigh": 20.934000466428405, "shank": 50.298509401819075, "foot": 13.5494321622424}, "action_second_difference_mean": 0.2853401608635332}, "current": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.1145939342634543, "height_rmse_mean_m": 0.028734957943238548, "landing_peak_mean_n": 1445.0841125488282, "landing_peak_worst_n": 1796.366943359375, "roll_peak_worst_deg": 7.665634982318925, "mirror_peak_worst_deg": {"hip": 12.504257405540008, "thigh": 21.711748130802235, "shank": 54.57758022814975, "foot": 13.548458860285672}, "action_second_difference_mean": 0.3246015747637305}}。没有完整周期的窗口为未知，不能据截断数据宣布改善。
- fixed-start-nominal-a01初始化与准备恢复阶段支撑对照（含尾段中已完整观察的准备窗口）：{"baseline": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 5.0, "recovery_single_support_ms_mean": 35.0, "recovery_single_support_ms_worst": 35, "recovery_longest_single_support_ms_worst": 15, "recovery_both_wheels_over_10n_fraction_mean": 0.9, "recovery_both_wheels_over_10n_fraction_worst": 0.9}, "current": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 5.0, "recovery_single_support_ms_mean": 0.0, "recovery_single_support_ms_worst": 0, "recovery_longest_single_support_ms_worst": 0, "recovery_both_wheels_over_10n_fraction_mean": 1.0, "recovery_both_wheels_over_10n_fraction_worst": 1.0}}。单轮支撑按另一轮力范数≤1N定义，双轮支撑另按每轮Fz>10N统计；此力门限统计不含奖励的0.1s接触计时，不能当作奖励原值。
- fixed-start-nominal-a01全采样风险极值：roll=7.883589928674293deg，pitch=8.64100782321002deg，平面位置误差=0.13917023006849885m，镜像峰值={"hip": 5.696843287871253, "thigh": 11.839091322368677, "shank": 29.87913892991539, "foot": 9.67159909115683}deg。包含未完成尾段；不把尾段看作完整成功。
- fixed-start-randomized-a01初始化与准备恢复阶段支撑对照（含尾段中已完整观察的准备窗口）：{"baseline": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 50.0, "recovery_single_support_ms_mean": 82.5, "recovery_single_support_ms_worst": 180, "recovery_longest_single_support_ms_worst": 150, "recovery_both_wheels_over_10n_fraction_mean": 0.9083333333333333, "recovery_both_wheels_over_10n_fraction_worst": 0.8}, "current": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 48.333333333333336, "recovery_single_support_ms_mean": 10.0, "recovery_single_support_ms_worst": 25, "recovery_longest_single_support_ms_worst": 20, "recovery_both_wheels_over_10n_fraction_mean": 0.9891666666666666, "recovery_both_wheels_over_10n_fraction_worst": 0.97}}。单轮支撑按另一轮力范数≤1N定义，双轮支撑另按每轮Fz>10N统计；此力门限统计不含奖励的0.1s接触计时，不能当作奖励原值。
- fixed-start-randomized-a01全采样风险极值：roll=7.665634982318925deg，pitch=22.575527306292216deg，平面位置误差=0.34245517362304756m，镜像峰值={"hip": 12.504257405540008, "thigh": 21.711748130802235, "shank": 54.57758022814975, "foot": 13.548458860285672}deg。包含未完成尾段；不把尾段看作完整成功。
- 训练过程窗口统计见training-windows-20261029-001.json。随机起始相位下的motion_finished训练指标不等同于完整起始周期的跳跃成功率；缺少批准的收敛判据。
- 名义组nominal_start关闭噪声、重置扰动、材质/质量/COM/增益随机化并将执行延迟设为0。随机化组randomized_start仅固定参考起始相位为0，保留训练随机化和0..3步延迟；两组均为确定性策略推理。
- 本轮按训练保存配置评估：4096环境/50000轮、每轮每环境24步；五项跟踪权重0.5/0.5/1/1/1；joint_symmetry=-0.5；hip_pos_zero_l2=-2.0；双轮持续接触权重0.5，0–1.30s，其中1.10–1.30s平滑渐出；前后对齐仍在1.25s结束。当前仓库已关闭两个起跳奖励，评估仅在进程内恢复保存的pitch_ang_vel=-0.03、takeoff_anchor_ori=0.5。训练源自9283cde加保存的三处本地改动，完整重建来源见training-source-reconstruction。对照的上版2026-10-27_20-02-47为8192环境/50000轮、hip=-1、双轮奖励1.0且1.25s硬关闭；参数与样本量均有差异，不作单参数因果结论。
- 5ms物理遥测包含4个足/轮连杆净法向力和4个大腿/小腿对地法向力。八连杆失载并非全机器人腾空的直接证明，base/hip未单独测量；净力不包含摩擦。applied_torque仅为驱动输出，不包含接触被动反力矩。
- 失败终止会截断后续落地过程：不得把未观察到的高度、冲击或接触按零处理，也不能直接以较低冲击证明训练改善。名义重复不是独立种子试验，单环境startup随机化也不覆盖全部物理参数组合。
- 所有模型、训练YAML、既有导出、USD、参考NPZ、相关源码及Isaac Sim持久配置字节与mtime均未变动。实际评估argv含进程内--/app/settings/persistent=false。未启动训练、导出、部署、安装、删除或Git提交。
- 没有用户批准的收敛判据，收敛结论为indeterminate。本批次只支持有限仿真观察；仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-10-27_20-02-47/evaluations/wheel-to-leg-fixed-start-20261028-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [fixed-start-motion-contact.png](<derived/fixed-start-motion-contact.png>)
- [training-windows-20261029-001.json](<../../evidence/training/training-windows-20261029-001.json>)
- [joint-symmetry-comparison.png](<derived/joint-symmetry-comparison.png>)
- [protection-before-20261029-001.json](<../../evidence/training/protection-before-20261029-001.json>)
- [protection-after-20261029-001.json](<../../evidence/training/protection-after-20261029-001.json>)
- [training-origin-20261029-001.json](<../../evidence/training/training-origin-20261029-001.json>)
- [training-source-reconstruction-20261029-001.json](<../../evidence/training/training-source-reconstruction-20261029-001.json>)

## 建议与待授权事项

- 按分段数据审查真实起跳、准备轮子支撑、姿态与落地触地终止；保持现有配置和证据，不自动调整训练参数。

# 评估批次 wheel-to-leg-fixed-start-20261028-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-27_20-02-47`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`fa4c4d97506f7a69b9900d649a22b4c2a164e642eeb74ffb9f10e138d5f4b8ed`；runner：`OnPolicyRunner`。
- [场景来源 scenario-41fd7bec3f18ee0415bb87959778af7ed0d4bc3887fd7ff19889beb48820891d.json](<../../provenance/scenario-41fd7bec3f18ee0415bb87959778af7ed0d4bc3887fd7ff19889beb48820891d.json>)
- [场景来源 scenario-63e0a628f763bf59e4ec31a8c9c3fbf2f35acb5e9ee5bdaaff0f43dc09b0ad2e.json](<../../provenance/scenario-63e0a628f763bf59e4ec31a8c9c3fbf2f35acb5e9ee5bdaaff0f43dc09b0ad2e.json>)
- [训练上下文 context-6a05e0cc7daa75453f770254aa481c33bae94c389b13ccece19bbb406bf66121.json](<../../provenance/context-6a05e0cc7daa75453f770254aa481c33bae94c389b13ccece19bbb406bf66121.json>)
- [训练有效配置 config-ec5d5451c5017ec1117525de1d9f9ce2544e067a340abd40ea3dbc76239fc498.json](<../../provenance/config-ec5d5451c5017ec1117525de1d9f9ce2544e067a340abd40ea3dbc76239fc498.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- fixed-start-nominal-a01完整周期：1.20–1.70s高度RMSE均值3.126cm、范围3.126–3.126cm；全程横滚绝对峰值最大11.352°；髋/大腿/小腿/足部镜像残差峰值{"hip": 7.419929901066454, "thigh": 25.120371538456844, "shank": 30.442877130798852, "foot": 12.199742386018276}度。
- fixed-start-nominal-a01失载后再接触至参考2.0s的落地记录法向力峰值最大1191.415N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-nominal-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-nominal-a01已到达起跳窗口的6段：竖直速度峰值1.514–1.514m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-nominal-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比0.9054、饱和物理样本0；关节速度最大名义限值比0.8562。
- fixed-start-randomized-a01完整周期：1.20–1.70s高度RMSE均值2.477cm、范围1.822–3.430cm；全程横滚绝对峰值最大11.448°；髋/大腿/小腿/足部镜像残差峰值{"hip": 14.577236894111802, "thigh": 20.934000466428405, "shank": 50.298509401819075, "foot": 13.5494321622424}度。
- fixed-start-randomized-a01失载后再接触至参考2.0s的落地记录法向力峰值最大3248.260N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-randomized-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-randomized-a01已到达起跳窗口的6段：竖直速度峰值1.087–1.398m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-randomized-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比1.0000、饱和物理样本1；关节速度最大名义限值比1.0821。
- 复用并校验2026-10-27_11-06-22旧server批次作为描述性对照，见analysis.json baseline_comparison；未重跑旧策略。奖励改变的单次训练比较不证明某项权重的因果效果。
- fixed-start-nominal-a01与上版server同口径完整周期指标对照：{"baseline": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.0874762420524688, "height_rmse_mean_m": 0.03409205937032801, "landing_peak_mean_n": 1626.4193115234375, "landing_peak_worst_n": 1626.4193115234375, "roll_peak_worst_deg": 5.6614494570528, "mirror_peak_worst_deg": {"hip": 7.435131127394553, "thigh": 8.755096987567839, "shank": 29.224227948896008, "foot": 18.015376964264625}, "action_second_difference_mean": 0.12174476083130667}, "current": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.5143943881939521, "height_rmse_mean_m": 0.03126309626329119, "landing_peak_mean_n": 1191.4149475097656, "landing_peak_worst_n": 1191.4149475097656, "roll_peak_worst_deg": 11.3522863369918, "mirror_peak_worst_deg": {"hip": 7.419929901066454, "thigh": 25.120371538456844, "shank": 30.442877130798852, "foot": 12.199742386018276}, "action_second_difference_mean": 0.13223713771855272}}。没有完整周期的窗口为未知，不能据截断数据宣布改善。
- fixed-start-randomized-a01与上版server同口径完整周期指标对照：{"baseline": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.1506624138219383, "height_rmse_mean_m": 0.028548569943858154, "landing_peak_mean_n": 1330.5432434082031, "landing_peak_worst_n": 2182.07666015625, "roll_peak_worst_deg": 12.621062886730003, "mirror_peak_worst_deg": {"hip": 12.728562525430519, "thigh": 16.18038004939629, "shank": 38.709345800238566, "foot": 13.447009206769827}, "action_second_difference_mean": 0.3236594790736601}, "current": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.2923824480591874, "height_rmse_mean_m": 0.024767791684818238, "landing_peak_mean_n": 1868.0273406982421, "landing_peak_worst_n": 3248.260009765625, "roll_peak_worst_deg": 11.447698816548595, "mirror_peak_worst_deg": {"hip": 14.577236894111802, "thigh": 20.934000466428405, "shank": 50.298509401819075, "foot": 13.5494321622424}, "action_second_difference_mean": 0.2853401608635332}}。没有完整周期的窗口为未知，不能据截断数据宣布改善。
- fixed-start-nominal-a01初始化与准备恢复阶段支撑对照（含尾段中已完整观察的准备窗口）：{"baseline": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 5.0, "recovery_single_support_ms_mean": 0.0, "recovery_single_support_ms_worst": 0, "recovery_longest_single_support_ms_worst": 0, "recovery_both_wheels_over_10n_fraction_mean": 1.0, "recovery_both_wheels_over_10n_fraction_worst": 1.0}, "current": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 5.0, "recovery_single_support_ms_mean": 35.0, "recovery_single_support_ms_worst": 35, "recovery_longest_single_support_ms_worst": 15, "recovery_both_wheels_over_10n_fraction_mean": 0.9, "recovery_both_wheels_over_10n_fraction_worst": 0.9}}。单轮支撑按另一轮力范数≤1N定义，双轮支撑另按每轮Fz>10N统计；此力门限统计不含奖励的0.1s接触计时，不能当作奖励原值。
- fixed-start-nominal-a01全采样风险极值：roll=11.3522863369918deg，pitch=13.664742393844683deg，平面位置误差=0.1354230653532587m，镜像峰值={"hip": 7.419929901066454, "thigh": 25.120371538456844, "shank": 30.442877130798852, "foot": 12.199742386018276}deg。包含未完成尾段；不把尾段看作完整成功。
- fixed-start-randomized-a01初始化与准备恢复阶段支撑对照（含尾段中已完整观察的准备窗口）：{"baseline": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 49.166666666666664, "recovery_single_support_ms_mean": 70.0, "recovery_single_support_ms_worst": 275, "recovery_longest_single_support_ms_worst": 120, "recovery_both_wheels_over_10n_fraction_mean": 0.9283333333333332, "recovery_both_wheels_over_10n_fraction_worst": 0.72}, "current": {"observed_preparation_windows": 6, "initial_single_support_ms_mean": 50.0, "recovery_single_support_ms_mean": 82.5, "recovery_single_support_ms_worst": 180, "recovery_longest_single_support_ms_worst": 150, "recovery_both_wheels_over_10n_fraction_mean": 0.9083333333333333, "recovery_both_wheels_over_10n_fraction_worst": 0.8}}。单轮支撑按另一轮力范数≤1N定义，双轮支撑另按每轮Fz>10N统计；此力门限统计不含奖励的0.1s接触计时，不能当作奖励原值。
- fixed-start-randomized-a01全采样风险极值：roll=12.51641733777119deg，pitch=25.774275393187974deg，平面位置误差=0.44912793648064603m，镜像峰值={"hip": 14.577236894111802, "thigh": 20.934000466428405, "shank": 50.298509401819075, "foot": 13.5494321622424}deg。包含未完成尾段；不把尾段看作完整成功。
- 训练过程窗口统计见training-windows-20261028-001.json。随机起始相位下的motion_finished训练指标不等同于完整起始周期的跳跃成功率；缺少批准的收敛判据。
- 名义组nominal_start关闭噪声、重置扰动、材质/质量/COM/增益随机化并将执行延迟设为0。随机化组randomized_start仅固定参考起始相位为0，保留训练随机化和0..3步延迟；两组均为确定性策略推理。
- 本轮按保存配置评估：五项跟踪权重0.5/0.5/1/1/1，joint_symmetry=-0.5，尺度(0.1,0.2,0.2,0.2)rad；wheel_contact_continuous=1.0，0–1.25s，10N门限，0.1s连续接触，硬关闭；训练8192环境/50000轮、每轮每环境24步。anchor_pos高度阈值0.15m、大腿/小腿触地>1N终止保留；评估未覆盖奖励权重。上一版2026-10-27_11-06-22同为8192环境、相同跟踪和对称权重，但双轮接触权重0.5且训练25000轮。因此不能将最终策略差异单独归因于接触权重。
- 5ms物理遥测包含4个足/轮连杆净法向力和4个大腿/小腿对地法向力。八连杆失载并非全机器人腾空的直接证明，base/hip未单独测量；净力不包含摩擦。applied_torque仅为驱动输出，不包含接触被动反力矩。
- 失败终止会截断后续落地过程：不得把未观察到的高度、冲击或接触按零处理，也不能直接以较低冲击证明训练改善。名义重复不是独立种子试验，单环境startup随机化也不覆盖全部物理参数组合。
- 所有模型、训练YAML、既有导出、USD、参考NPZ、相关源码及Isaac Sim持久配置字节与mtime均未变动。实际评估argv含进程内--/app/settings/persistent=false。未启动训练、导出、部署、安装、删除或Git提交。
- 没有用户批准的收敛判据，收敛结论为indeterminate。本批次只支持有限仿真观察；仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-10-27_11-06-22/evaluations/wheel-to-leg-fixed-start-20261027-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [fixed-start-motion-contact.png](<derived/fixed-start-motion-contact.png>)
- [training-windows-20261028-001.json](<../../evidence/training/training-windows-20261028-001.json>)
- [joint-symmetry-comparison.png](<derived/joint-symmetry-comparison.png>)
- [protection-before-20261028-001.json](<../../evidence/training/protection-before-20261028-001.json>)
- [protection-after-20261028-001.json](<../../evidence/training/protection-after-20261028-001.json>)
- [training-origin-20261028-001.json](<../../evidence/training/training-origin-20261028-001.json>)

## 建议与待授权事项

- 按分段数据审查真实起跳、准备轮子支撑、姿态与落地触地终止；保持现有配置和证据，不自动调整训练参数。

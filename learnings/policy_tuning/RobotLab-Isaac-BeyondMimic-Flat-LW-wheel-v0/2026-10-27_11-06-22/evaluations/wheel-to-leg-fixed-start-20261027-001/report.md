# 评估批次 wheel-to-leg-fixed-start-20261027-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-27_11-06-22`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`0e17ec9cc56c5a60902edd4c5a8314a2ba1ab2a2350607f04835ea41b75a8616`；runner：`OnPolicyRunner`。
- [场景来源 scenario-47f215cf6870d1c901ac57dfc064e8d1ff8c7cbe25e700e261e0ecd9ab53d8d8.json](<../../provenance/scenario-47f215cf6870d1c901ac57dfc064e8d1ff8c7cbe25e700e261e0ecd9ab53d8d8.json>)
- [场景来源 scenario-dcf8efeef070ab908246e89c09e1d8d2457e86a39f990f458cbbd431ffb72f47.json](<../../provenance/scenario-dcf8efeef070ab908246e89c09e1d8d2457e86a39f990f458cbbd431ffb72f47.json>)
- [训练上下文 context-75c3eed5811c0dd63b121c17b57f6816f52ddc3a7afd84347a572a3df05daced.json](<../../provenance/context-75c3eed5811c0dd63b121c17b57f6816f52ddc3a7afd84347a572a3df05daced.json>)
- [训练有效配置 config-e61bf66b65afe1884dbd823c6f320556fcdfa4881729378cecc6501d3504aff8.json](<../../provenance/config-e61bf66b65afe1884dbd823c6f320556fcdfa4881729378cecc6501d3504aff8.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- fixed-start-nominal-a01完整周期：1.20–1.70s高度RMSE均值3.409cm、范围3.409–3.409cm；全程横滚绝对峰值最大5.661°；髋/大腿/小腿/足部镜像残差峰值{"hip": 7.435131127394553, "thigh": 8.755096987567839, "shank": 29.224227948896008, "foot": 18.015376964264625}度。
- fixed-start-nominal-a01失载后再接触至参考2.0s的落地记录法向力峰值最大1626.419N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-nominal-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-nominal-a01已到达起跳窗口的6段：竖直速度峰值1.087–1.087m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-nominal-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比0.7341、饱和物理样本0；关节速度最大名义限值比0.6715。
- fixed-start-randomized-a01完整周期：1.20–1.70s高度RMSE均值2.855cm、范围2.233–3.768cm；全程横滚绝对峰值最大12.621°；髋/大腿/小腿/足部镜像残差峰值{"hip": 12.728562525430519, "thigh": 16.18038004939629, "shank": 38.709345800238566, "foot": 13.447009206769827}度。
- fixed-start-randomized-a01失载后再接触至参考2.0s的落地记录法向力峰值最大2182.077N；区别于固定1.60–2.00s窗口，包含可能早于1.60s的首次落地。
- fixed-start-randomized-a01：1000控制步、4000物理步；5个完整参考周期、0个失败终止、另有1个未完成尾段。所有重置均从frame0开始，遥测complete；终止项计数{"motion_finished": 5}。
- fixed-start-randomized-a01已到达起跳窗口的6段：竖直速度峰值1.030–1.244m/s；参考同窗口峰值1.967m/s。分段指标见analysis.json，提前终止后的窗口不是零误差。
- fixed-start-randomized-a01：接触视图交叉误差最大0N；全程驱动力矩最大限幅比1.0000、饱和物理样本2；关节速度最大名义限值比1.0057。
- 复用并校验2026-10-26_13-25-00旧server批次作为描述性对照，见analysis.json baseline_comparison；未重跑旧策略。奖励改变的单次训练比较不证明某项权重的因果效果。
- fixed-start-nominal-a01与上版server同口径完整周期指标对照：{"baseline": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.2507863468719569, "height_rmse_mean_m": 0.03094167871420824, "landing_peak_mean_n": 1858.7762451171875, "landing_peak_worst_n": 1858.7762451171875, "roll_peak_worst_deg": 7.610954924261912, "mirror_peak_worst_deg": {"hip": 9.5940674794539, "thigh": 14.73589962721026, "shank": 37.58683153090365, "foot": 28.70845816655354}, "action_second_difference_mean": 0.22146197873640389}, "current": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.0874762420524688, "height_rmse_mean_m": 0.03409205937032801, "landing_peak_mean_n": 1626.4193115234375, "landing_peak_worst_n": 1626.4193115234375, "roll_peak_worst_deg": 5.6614494570528, "mirror_peak_worst_deg": {"hip": 7.435131127394553, "thigh": 8.755096987567839, "shank": 29.224227948896008, "foot": 18.015376964264625}, "action_second_difference_mean": 0.12174476083130667}}。没有完整周期的窗口为未知，不能据截断数据宣布改善。
- fixed-start-randomized-a01与上版server同口径完整周期指标对照：{"baseline": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.3605086570646274, "height_rmse_mean_m": 0.027706795631607196, "landing_peak_mean_n": 1522.0076110839843, "landing_peak_worst_n": 1801.8585205078125, "roll_peak_worst_deg": 13.681533921543682, "mirror_peak_worst_deg": {"hip": 25.724933780009476, "thigh": 37.583896470551245, "shank": 48.68705689452244, "foot": 48.45415085891959}, "action_second_difference_mean": 0.4781691131280662}, "current": {"complete_cycles": 5, "takeoff_vz_mean_m_s": 1.1506624138219383, "height_rmse_mean_m": 0.028548569943858154, "landing_peak_mean_n": 1330.5432434082031, "landing_peak_worst_n": 2182.07666015625, "roll_peak_worst_deg": 12.621062886730003, "mirror_peak_worst_deg": {"hip": 12.728562525430519, "thigh": 16.18038004939629, "shank": 38.709345800238566, "foot": 13.447009206769827}, "action_second_difference_mean": 0.3236594790736601}}。没有完整周期的窗口为未知，不能据截断数据宣布改善。
- 训练过程窗口统计见training-windows-20261027-001.json。随机起始相位下的motion_finished训练指标不等同于完整起始周期的跳跃成功率；缺少批准的收敛判据。
- 名义组nominal_start关闭噪声、重置扰动、材质/质量/COM/增益随机化并将执行延迟设为0。随机化组randomized_start仅固定参考起始相位为0，保留训练随机化和0..3步延迟；两组均为确定性策略推理。
- 本轮按当前训练保存值评估：五项跟踪权重为基座位置0.5、基座姿态0.5、连杆位置1、连杆姿态1、连杆线速度1；joint_symmetry=-0.5，尺度(0.1,0.2,0.2,0.2)rad；训练8192环境/25000轮，num_steps_per_env=24。anchor_pos高度阈值0.15m和大腿/小腿对地>1N终止保留；评估未覆盖奖励权重。上一版server有五项不同跟踪权重，且4096环境/50000轮，两版总采样量相同但不能归因到单个参数。
- 5ms物理遥测包含4个足/轮连杆净法向力和4个大腿/小腿对地法向力。八连杆失载并非全机器人腾空的直接证明，base/hip未单独测量；净力不包含摩擦。applied_torque仅为驱动输出，不包含接触被动反力矩。
- 失败终止会截断后续落地过程：不得把未观察到的高度、冲击或接触按零处理，也不能直接以较低冲击证明训练改善。名义重复不是独立种子试验，单环境startup随机化也不覆盖全部物理参数组合。
- 所有模型、训练YAML、既有导出、USD、参考NPZ、相关源码及Isaac Sim持久配置字节与mtime均未变动。实际评估argv含进程内--/app/settings/persistent=false。未启动训练、导出、部署、安装、删除或Git提交。
- 没有用户批准的收敛判据，收敛结论为indeterminate。本批次只支持有限仿真观察；仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-10-26_13-25-00/evaluations/wheel-to-leg-fixed-start-20261027-001/manifest.json>)
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

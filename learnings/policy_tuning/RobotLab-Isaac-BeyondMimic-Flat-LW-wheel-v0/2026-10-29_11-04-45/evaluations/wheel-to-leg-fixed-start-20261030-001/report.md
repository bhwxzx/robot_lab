# 评估批次 wheel-to-leg-fixed-start-20261030-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-29_11-04-45`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`47a797c57620e761af2118a38daf838ae4e0a14ac6800faa7f0ed0d392be7650`；runner：`OnPolicyRunner`。
- [场景来源 scenario-5d0ab3b694c43b3fc20bc539a43e38df1e5a076540976def6edc0e3fcbfab95b.json](<../../provenance/scenario-5d0ab3b694c43b3fc20bc539a43e38df1e5a076540976def6edc0e3fcbfab95b.json>)
- [场景来源 scenario-f0c3fca3931067024aa06d3b43951564661f581a2b430207b95a773edb992566.json](<../../provenance/scenario-f0c3fca3931067024aa06d3b43951564661f581a2b430207b95a773edb992566.json>)
- [训练上下文 context-512a7180ffb510b698f2e91b0e30dc43d44165974a3df7dad2f7a3d9d9fc9d1a.json](<../../provenance/context-512a7180ffb510b698f2e91b0e30dc43d44165974a3df7dad2f7a3d9d9fc9d1a.json>)
- [训练有效配置 config-3d5f5125c913763bc6bba441103ba4e859f58b6c1b0eceffe70c276542660583.json](<../../provenance/config-3d5f5125c913763bc6bba441103ba4e859f58b6c1b0eceffe70c276542660583.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.wheel_contact_continuous.weight": 0.5}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.wheel_contact_continuous.weight": 0.5}`。

## 观察与限制

- fixed-start-nominal-a01完整周期统计：{"complete_cycles": 0, "censored_tails": 1, "failed_terminations": 18, "height_rmse_cm": null, "takeoff_vz_peak_m_s": null, "eight_measured_links_unloaded_s": null, "landing_peak_n": null, "prepare_both_wheels_over10N_fraction": null, "prepare_single_support_ms": null, "roll_peak_deg": null, "pitch_peak_deg": null, "takeoff_hip_mirror_peak_deg": null, "mirror_peak_deg": {"hip": null, "thigh": null, "shank": null, "foot": null}, "foot_velocity_peak_all_samples_rad_s": 3.4172911643981934, "drive_saturated_physics_samples_all": 0}
- fixed-start-nominal-a01上一server同口径：{"complete_cycles": 5, "censored_tails": 1, "failed_terminations": 0, "height_rmse_cm": {"mean": 2.483645912603835, "min": 2.483645912603835, "max": 2.483645912603835}, "takeoff_vz_peak_m_s": {"mean": 1.2330572671093543, "min": 1.2330572671093543, "max": 1.2330572671093543}, "eight_measured_links_unloaded_s": {"mean": 0.23, "min": 0.23, "max": 0.23}, "landing_peak_n": {"mean": 2380.9115600585938, "min": 2380.9115600585938, "max": 2380.9115600585938}, "prepare_both_wheels_over10N_fraction": {"mean": 1, "min": 1, "max": 1}, "prepare_single_support_ms": {"mean": 0.0, "min": 0.0, "max": 0.0}, "roll_peak_deg": 7.883589928674293, "pitch_peak_deg": 8.64100782321002, "takeoff_hip_mirror_peak_deg": 5.696522432866867, "mirror_peak_deg": {"hip": 5.696522432866867, "thigh": 11.827176057361589, "shank": 29.878852061970246, "foot": 9.675902110333938}, "foot_velocity_peak_all_samples_rad_s": 10.010662078857422, "drive_saturated_physics_samples_all": 0}
- fixed-start-nominal-a01各周期起跳髋偏移：[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]
- fixed-start-randomized-a01完整周期统计：{"complete_cycles": 0, "censored_tails": 1, "failed_terminations": 29, "height_rmse_cm": null, "takeoff_vz_peak_m_s": null, "eight_measured_links_unloaded_s": null, "landing_peak_n": null, "prepare_both_wheels_over10N_fraction": null, "prepare_single_support_ms": null, "roll_peak_deg": null, "pitch_peak_deg": null, "takeoff_hip_mirror_peak_deg": null, "mirror_peak_deg": {"hip": null, "thigh": null, "shank": null, "foot": null}, "foot_velocity_peak_all_samples_rad_s": 5.559927463531494, "drive_saturated_physics_samples_all": 0}
- fixed-start-randomized-a01上一server同口径：{"complete_cycles": 5, "censored_tails": 1, "failed_terminations": 0, "height_rmse_cm": {"mean": 2.873495794323855, "min": 1.446710971073568, "max": 5.264106030053688}, "takeoff_vz_peak_m_s": {"mean": 1.1145939342634543, "min": 0.7450575054620437, "max": 1.4004596929650235}, "eight_measured_links_unloaded_s": {"mean": 0.217, "min": 0.20500000000000002, "max": 0.24}, "landing_peak_n": {"mean": 1445.0841125488282, "min": 1161.6058044433594, "max": 1796.366943359375}, "prepare_both_wheels_over10N_fraction": {"mean": 0.986, "min": 0.97, "max": 1}, "prepare_single_support_ms": {"mean": 13.0, "min": 0.0, "max": 25.0}, "roll_peak_deg": 7.665634982318925, "pitch_peak_deg": 22.575527306292216, "takeoff_hip_mirror_peak_deg": 9.842563650288456, "mirror_peak_deg": {"hip": 12.503982385814673, "thigh": 21.7132371120413, "shank": 54.5776826809873, "foot": 13.552355483207165}, "foot_velocity_peak_all_samples_rad_s": 14.99648666381836, "drive_saturated_physics_samples_all": 1}
- fixed-start-randomized-a01各周期起跳髋偏移：[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]
- 最新本机server2026-10-29_11-04-45/model_49999.pt；两组frame0、seed42、1环境各1000控制步/4000物理步，Native无视频。
- 保存配置4096环境/50000轮/每环境24步；与上一server训练有效配置仅hip_pos_zero_l2从-2改为-3（另有日志路径变化）。一次训练对照仍不证明因果。
- 当前源码wheel_contact_continuous=0；评估仅进程内恢复保存训练值0.5。两项起跳奖励保存值-0.03/0.5；其余参数和物理随机化按原场景。
- 名义组关闭随机化/噪声/扰动和延迟；随机化组保留训练随机化和0..3物理步延迟，仅固定参考从frame0开始。
- 单seed和一次随机化startup抽样；名义重复周期不是独立种子。完整周期用于均值，尾段只用于已观测窗口和风险极值。
- 137个保护文件哈希/mtime核验：全部未变；没有修改源码、训练YAML、模型、导出、旧证据或Isaac持久设置。
- 力来自5ms物理子步的净法向力，排除摩擦；驱动力矩不包含被动碰撞反力矩。八连杆失载不证明base/hip也已离地。
- 训练随机起始相位motion_finished指标不能代替完整动作成功率；缺少用户批准的收敛判据，收敛为indeterminate。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-10-28_13-14-55/evaluations/wheel-to-leg-fixed-start-20261029-001/manifest.json>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [protection-before-20261030-001.json](<../../evidence/training/protection-before-20261030-001.json>)
- [protection-after-20261030-001.json](<../../evidence/training/protection-after-20261030-001.json>)
- [training-windows-20261030-001.json](<../../evidence/training/training-windows-20261030-001.json>)
- [training-origin-20261030-001.json](<../../evidence/training/training-origin-20261030-001.json>)
- [motion-comparison.png](<derived/motion-comparison.png>)

## 建议与待授权事项

- 先对照起跳双髋共同偏移、足速越限、随机化小腿镜像残差和落地峰值，再提参数方案；不自动修改或启动训练。

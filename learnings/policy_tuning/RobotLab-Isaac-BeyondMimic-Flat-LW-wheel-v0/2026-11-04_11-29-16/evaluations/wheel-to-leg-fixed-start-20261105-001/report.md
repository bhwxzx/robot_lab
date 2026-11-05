# 评估批次 wheel-to-leg-fixed-start-20261105-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-04_11-29-16`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ef7ea23e11b9818c539be9254e664724036650f8bb7144d2eb4c73002ff683fc`；runner：`OnPolicyRunner`。
- [场景来源 scenario-08998eaed8bb99eb5dc0a370a40922d0947b1e5ba0143ac76d59d2708d0dce5e.json](<../../provenance/scenario-08998eaed8bb99eb5dc0a370a40922d0947b1e5ba0143ac76d59d2708d0dce5e.json>)
- [场景来源 scenario-5f0c9fdc115a57c4853da3792401d69b719c984694329783fa77b10b3dfd8c7b.json](<../../provenance/scenario-5f0c9fdc115a57c4853da3792401d69b719c984694329783fa77b10b3dfd8c7b.json>)
- [训练上下文 context-d897c33b7458c024f3ab7d59ef91509532412e5b05dd259dffbb0b5784975956.json](<../../provenance/context-d897c33b7458c024f3ab7d59ef91509532412e5b05dd259dffbb0b5784975956.json>)
- [训练有效配置 config-188a240cb0d5ef049a7b23a33dc8983d916ce0791836e0e894d4e37099be26e0.json](<../../provenance/config-188a240cb0d5ef049a7b23a33dc8983d916ce0791836e0e894d4e37099be26e0.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 本机 server 场次2026-11-04_11-29-16/model_49999.pt；SHA256 ef7ea23e11b9818c539be9254e664724036650f8bb7144d2eb4c73002ff683fc。不是 younghit 的11-33-55场次。
- 用户确认 bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra参数。8192环境、50000轮、24步/env、seed42；训练HEAD7f17bed，保存的git diff干净，相关源码与当前HEAD逐字节相同。
- 训练有效配置与本机11月3日相比，除log_dir外仅新增 leg_symmetry=+0.5/std0.05m 与 feet_distance_penalize=-20/范围0.496–0.536m；横滚奖励仍为+0.5，joint_symmetry=-0.5，hip_pos_zero_l2=-2。agent配置相同。
- Native mean policy、固定参考frame0；nominal去掉随机化，randomized保留训练随机化；各1环境/seed42/1000控制步/4000物理样本，无视频，无网络更新。
- 控制状态为动作后、重置前；phase沿用动作前reference frame/50Hz。轮距为世界XY轮心距离；双轮支撑为两轮Fz同时>10N。双轮窗口0.1≤phase<1.25s；另记录0.1–1.1s支撑比例。分阶段统计含截尾样本；每周期平均仅完整周期。
- 起跳速度采用世界z速度在1.1–1.36s内峰值；落地力采用1.5–2.0s内足/轮/大小腿八连杆正向Fz之和峰值。驱动力矩不包含被动碰撞载荷；配置速度/扭矩用于机械对照，不是硬件验收标准。
- 历史基线复用本机11月3日同场景数据，已校验；单训练seed的前后对比不能证明单个奖励项的因果效果。单startup物性样本不能代表随机化总体成功率。
- 末1000轮motion_finished均值0.996143，采样归一化熵0.850933。随机起始训练指标不是完整动作成功率；无批准收敛标准，convergence=indeterminate。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整/失败/截尾=(5, 0, 1)；历史reset=True；关键指标={"wheel_distance_min_m": 0.516263977301018, "wheel_distance_max_m": 0.5182951279859996, "wheel_distance_outside_fraction": 0.0, "bilateral_support_fraction_0p1_1p25s": 1.0, "roll_abs_peak_0p8_1p4s_deg": 4.750038327537937, "pitch_tracking_rmse_deg": 1.6050062771503881, "hip_mirror_peak_1_1p8s_deg": 4.412038974740927, "ground_thigh_shank_force_peak_n": 0.0, "completed_cycle_means": {"takeoff_vz_peak_m_s": 1.0616788820496188, "height_peak_m": 0.6997802257537842, "landing_peak_1p5_2s_n": 1975.0074462890625}}
- fixed-start-randomized-a01: 完整/失败/截尾=(5, 0, 1)；历史reset=True；关键指标={"wheel_distance_min_m": 0.4677665321277217, "wheel_distance_max_m": 0.5148191417068675, "wheel_distance_outside_fraction": 0.49353448275862066, "bilateral_support_fraction_0p1_1p25s": 0.9755747126436781, "roll_abs_peak_0p8_1p4s_deg": 8.323822127661694, "pitch_tracking_rmse_deg": 4.299342323727862, "hip_mirror_peak_1_1p8s_deg": 8.08900557384653, "ground_thigh_shank_force_peak_n": 0.0, "completed_cycle_means": {"takeoff_vz_peak_m_s": 1.0885213487318217, "height_peak_m": 0.7022641777992249, "landing_peak_1p5_2s_n": 1348.4706085205078}}
- fixed-start-randomized-a01 关节限值事件：{"joint": "left_foot_joint", "velocity_peak_rad_s": 12.507267951965332, "velocity_rms_rad_s": 1.3149150708406894, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 5, "overspeed_total_ms": 25.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.595, "velocity_peak_episode": 3, "applied_torque_at_velocity_peak_nm": 0.5273885726928711, "torque_peak_nm": 27.0, "torque_rms_nm": 4.511570311234135, "effort_limit_nm": 27.0, "saturation_samples": 3, "saturation_total_ms": 15.0}
- fixed-start-randomized-a01 关节限值事件：{"joint": "right_foot_joint", "velocity_peak_rad_s": 10.038241386413574, "velocity_rms_rad_s": 1.1336218119085555, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 2, "overspeed_total_ms": 10.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.555, "velocity_peak_episode": 2, "applied_torque_at_velocity_peak_nm": 0.7763112783432007, "torque_peak_nm": 24.01355743408203, "torque_rms_nm": 2.8583033735883663, "effort_limit_nm": 27.0, "saturation_samples": 0, "saturation_total_ms": 0.0}
- 受保护文件校验：{"checked_at": "2026-11-05T05:43:26.219624+00:00", "checked_files": 148, "changed": [], "stash_unchanged": true}
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../../../2026-11-03_10-50-39/evaluations/wheel-to-leg-fixed-start-20261104-001/manifest.json>)
- [config-diff-20261105-001.json](<../../evidence/training/config-diff-20261105-001.json>)
- [evaluation-request-20261105-001.json](<../../evidence/training/evaluation-request-20261105-001.json>)
- [protection-before-20261105-001.json](<../../evidence/training/protection-before-20261105-001.json>)
- [training-origin-20261105-001.json](<../../evidence/training/training-origin-20261105-001.json>)
- [training-windows-20261105-001.json](<../../evidence/training/training-windows-20261105-001.json>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [phase-comparison.png](<derived/phase-comparison.png>)
- [protection-after-20261105-001.json](<../../evidence/training/protection-after-20261105-001.json>)

## 建议与待授权事项

- 综合轮距、双轮支撑、姿态、起跳和关节限值审阅；后续优化先提出方案。此次未启动训练、导出或提交Git。

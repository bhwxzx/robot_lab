# 评估批次 wheel-to-leg-fixed-start-20261102-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-31_15-48-13`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`d5d75f2da8d11057f00d65f4f73f13dcee9dfb4dec312d362622fc1b0a058059`；runner：`OnPolicyRunner`。
- [场景来源 scenario-bdbd26c52071b3b59457c0ad9eaf2a0df07323bc5fe251dd0cff464e4d526a11.json](<../../provenance/scenario-bdbd26c52071b3b59457c0ad9eaf2a0df07323bc5fe251dd0cff464e4d526a11.json>)
- [场景来源 scenario-c182549559a708c27b617438c550f9c3b918c2b9106ca4ef9bb780ea01d61bcf.json](<../../provenance/scenario-c182549559a708c27b617438c550f9c3b918c2b9106ca4ef9bb780ea01d61bcf.json>)
- [训练上下文 context-60e0625421acf2b392e2fb4747819ca55c61cfdd47f159030da914d0575159d4.json](<../../provenance/context-60e0625421acf2b392e2fb4747819ca55c61cfdd47f159030da914d0575159d4.json>)
- [训练有效配置 config-fec6608572a1beb4da6cba32099c69d5c38db56618098433f023566a1a54ab7f.json](<../../provenance/config-fec6608572a1beb4da6cba32099c69d5c38db56618098433f023566a1a54ab7f.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 本机 server 场次2026-10-31_15-48-13/model_49999.pt；SHA256 d5d75f2da8d11057f00d65f4f73f13dcee9dfb4dec312d362622fc1b0a058059；PPO/OnPolicyRunner，8192环境、50000轮、seed42。
- 用户确认启动命令 bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra参数；训练与评估源码HEAD均为879357b，训练git记录干净。
- 保存配置相对上一版2026-10-30_11-48-12，除log_dir外唯一差异为 same_foot_x_position.ramp_time_s 从0.2变为0.0；hip_pos_zero_l2=-2、joint_symmetry=-0.5。
- Native固定第0帧名义/训练随机化两组，各1环境、1000控制步、seed42、无视频；控制周期0.02s、物理周期0.005s。
- 执行包装器仅给评估进程追加 --kit_args=--/app/settings/persistent=false；相位覆盖hip_pos_zero_l2=-2为与保存配置一致的无效应覆盖，保持与旧场景一致。
- 末5000轮motion_finished均值 0.9951709052562714；此训练随机相位重置指标不是完整动作成功率，也不代表收敛。
- 末1000轮困难采样top1概率均值 0.4984201722145081，top1归一化bin恒0.25；采样仍集中于包含起跳的第二时间段。
- 周期均值仅使用完整动作；风险峰值包含所有采样。名义重复周期不是独立种子，随机化仅一次启动随机化及同一seed的周期重置。
- 起跳速度窗口1.1..1.36s；髋镜像窗口1.0..1.5s；落地峰值取1.1..1.8s最长八连杆失载区间后至2.0s。八连杆失载不覆盖base/hip，不能单独证明整机腾空。
- 实际驱动力矩不包含全部被动碰撞载荷；超速/饱和累计时间与最长连续时间分别记录。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整周期/失败终止/未完成尾段=(5, 0, 1)，遥测complete，历史reset检查=True。
- fixed-start-nominal-a01 完整周期均值={"takeoff_vz_peak_m_s": 1.0554730588994148, "flight_height_rmse_cm": 3.7236710911594946, "prepare_min_height_m": 0.5585359334945679, "landing_min_height_m": 0.456643670797348, "eight_link_unloading_ms": 220.0, "landing_peak_n": 1753.2488098144531}
- fixed-start-nominal-a01 起跳髋峰值={"mirror_error_deg": 6.446576077590628, "left_deg": 3.7099483019471253, "right_deg": 2.7366277756435022, "phase_s": 1.22, "episode_id": 0}
- fixed-start-nominal-a01 风险峰值：roll=6.322722883483669deg，pitch=8.832449622181985deg，水平anchor偏差=8.140813280158627cm，小腿/大腿地面接触=0.0N。
- fixed-start-nominal-a01 起跳前1.0..1.3s指标={"roll_abs_peak_deg": 6.322722883483669, "left_hip_abs_peak_deg": 3.7099483019471253, "right_hip_abs_peak_deg": 2.7366277756435022, "wheel_fore_aft_separation_peak_cm": 2.531874842104606}
- fixed-start-nominal-a01 相对上一版 takeoff_vz_peak_m_s: 1.4769405452466087 -> 1.0554730588994148
- fixed-start-nominal-a01 相对上一版 landing_peak_n: 923.4376831054688 -> 1753.2488098144531
- fixed-start-nominal-a01 相对上一版髋镜像峰值：{'mirror_error_deg': 12.20988201528457, 'left_deg': 8.865978278553827, 'right_deg': 3.343903736730744, 'phase_s': 1.22, 'episode_id': 0} -> {'mirror_error_deg': 6.446576077590628, 'left_deg': 3.7099483019471253, 'right_deg': 2.7366277756435022, 'phase_s': 1.22, 'episode_id': 0}
- fixed-start-randomized-a01: 完整周期/失败终止/未完成尾段=(5, 0, 1)，遥测complete，历史reset检查=True。
- fixed-start-randomized-a01 完整周期均值={"takeoff_vz_peak_m_s": 1.152217122803816, "flight_height_rmse_cm": 2.5757665816210613, "prepare_min_height_m": 0.5498752951622009, "landing_min_height_m": 0.46123196482658385, "eight_link_unloading_ms": 228.0, "landing_peak_n": 1387.382778930664}
- fixed-start-randomized-a01 起跳髋峰值={"mirror_error_deg": 12.945222315762258, "left_deg": 10.176944297278563, "right_deg": 2.768278018483695, "phase_s": 1.4, "episode_id": 3}
- fixed-start-randomized-a01 风险峰值：roll=10.541949700140195deg，pitch=27.48970425517141deg，水平anchor偏差=29.751069797964412cm，小腿/大腿地面接触=0.0N。
- fixed-start-randomized-a01 起跳前1.0..1.3s指标={"roll_abs_peak_deg": 10.47501251159979, "left_hip_abs_peak_deg": 8.52815626843135, "right_hip_abs_peak_deg": 10.215931017060997, "wheel_fore_aft_separation_peak_cm": 3.76464968168459}
- fixed-start-randomized-a01 相对上一版 takeoff_vz_peak_m_s: 1.2847663421410487 -> 1.152217122803816
- fixed-start-randomized-a01 相对上一版 landing_peak_n: 1206.6432800292969 -> 1387.382778930664
- fixed-start-randomized-a01 相对上一版髋镜像峰值：{'mirror_error_deg': 10.878809333705917, 'left_deg': -0.3584508089219286, 'right_deg': -10.520358524783989, 'phase_s': 1.3, 'episode_id': 4} -> {'mirror_error_deg': 12.945222315762258, 'left_deg': 10.176944297278563, 'right_deg': 2.768278018483695, 'phase_s': 1.4, 'episode_id': 3}
- fixed-start-randomized-a01 限值事件 {"joint": "left_foot_joint", "velocity_peak_rad_s": 10.043514251708984, "velocity_rms_rad_s": 1.373972612890914, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 3, "overspeed_total_ms": 15.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.59, "velocity_peak_episode": 3, "torque_peak_nm": 22.680973052978516, "torque_rms_nm": 3.3960328567607263, "effort_limit_nm": 27.0, "saturation_samples": 0, "saturation_total_ms": 0.0, "saturation_longest_ms": 0.0}
- fixed-start-randomized-a01 限值事件 {"joint": "right_foot_joint", "velocity_peak_rad_s": 10.98525619506836, "velocity_rms_rad_s": 1.388460425444481, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 1, "overspeed_total_ms": 5.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.625, "velocity_peak_episode": 4, "torque_peak_nm": 21.565330505371094, "torque_rms_nm": 2.5998268654430947, "effort_limit_nm": 27.0, "saturation_samples": 0, "saturation_total_ms": 0.0, "saturation_longest_ms": 0.0}
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [hip-alignment-comparison.png](<derived/hip-alignment-comparison.png>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [jump-contact-comparison.png](<derived/jump-contact-comparison.png>)
- [protection-after-20261102-001.json](<../../evidence/training/protection-after-20261102-001.json>)
- [protection-before-20261102-001.json](<../../evidence/training/protection-before-20261102-001.json>)
- [training-origin-20261102-001.json](<../../evidence/training/training-origin-20261102-001.json>)
- [training-windows-20261102-001.json](<../../evidence/training/training-windows-20261102-001.json>)
- [manifest.json](<../../../2026-10-30_11-48-12/evaluations/wheel-to-leg-fixed-start-20261031-001/manifest.json>)

## 建议与待授权事项

- 优先结合名义和随机化两组的髋偏斜、机身roll、起跳速度及落地冲击进行判断；不能仅凭训练reward或motion_finished推荐部署。
- 本次比较只覆盖单次训练、单个评估seed；参数效果需更多受控证据确认。后续训练参数改动先提出方案并获批准。
- 原始证据、训练checkpoint及既有导出文件均保持原样；本次未训练、导出、安装或执行Git提交。

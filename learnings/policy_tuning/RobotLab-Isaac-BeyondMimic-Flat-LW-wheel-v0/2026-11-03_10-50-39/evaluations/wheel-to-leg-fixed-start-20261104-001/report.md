# 评估批次 wheel-to-leg-fixed-start-20261104-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-03_10-50-39`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`cc35f8ce85c356521189a51d42a8d29bed50fb49676df7e2c212166180d5f96d`；runner：`OnPolicyRunner`。
- [场景来源 scenario-2246a6b294ca2eed74a87da21412faab348a6a43d3178b231884428059552bdd.json](<../../provenance/scenario-2246a6b294ca2eed74a87da21412faab348a6a43d3178b231884428059552bdd.json>)
- [场景来源 scenario-d38d2dd29cc3266ae9c0b04f7ce31902af97ee70d66c30796f0f837b12614a9f.json](<../../provenance/scenario-d38d2dd29cc3266ae9c0b04f7ce31902af97ee70d66c30796f0f837b12614a9f.json>)
- [训练上下文 context-f8e4e6858efa31e0dc103be07308052ff72badf3a8f6a41f3da1a4296a3deb63.json](<../../provenance/context-f8e4e6858efa31e0dc103be07308052ff72badf3a8f6a41f3da1a4296a3deb63.json>)
- [训练有效配置 config-ee69d7bc94aa67ee7f3020ac3343b379656cfda1e471ddbc8cd6e5abc524e15b.json](<../../provenance/config-ee69d7bc94aa67ee7f3020ac3343b379656cfda1e471ddbc8cd6e5abc524e15b.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 本机 server 场次2026-11-03_10-50-39/model_49999.pt；SHA256 cc35f8ce85c356521189a51d42a8d29bed50fb49676df7e2c212166180d5f96d；PPO/OnPolicyRunner，8192环境、50000轮、seed42。
- 用户确认 bash scripts/start_beyondmimic.sh --type wheel，无额外参数。训练源30485c8加训练时保存的roll奖励补丁；该补丁与当前a4e7603相关文件差异完全一致，其余相关源码逐字节相同。
- 相对本机11月2日版本，除log_dir外唯一保存配置变化是新增 motion_anchor_roll_horizontal，weight=+0.5，std=0.15rad，全程鼓励世界坐标roll=0；agent配置字节相同。anchor高度终止阈值均为0.25m。
- Native固定第0帧名义/训练随机化两组，各1环境、1000控制步、seed42、无视频；控制周期0.02s、物理周期0.005s。均值动作，无网络更新。
- 执行包装器仅给评估进程追加 --kit_args=--/app/settings/persistent=false；hip_pos_zero_l2=-2与保存配置一致，保持旧场景契约。
- 末1000轮采样top1概率均值=0.41685701030492783，归一化熵=0.8451336497664451；top1-bin记录范围=(0.25, 0.5)，第二/三段附近切换，不能将迭代平均值当成精确采样直方图。
- 末5000轮motion_finished均值=0.9961791177749634；随机相位正常结束指标并非完整动作成功率。新增正奖励后总reward不可直接与旧版本比较，收敛判定仍indeterminate。
- 周期均值仅使用完整动作；风险峰值包含所有采样。名义重复周期不是独立种子，随机化仅单个seed；单次训练比较不构成普遍因果结论。
- 起跳速度窗口1.1..1.36s；髋镜像窗口1.0..1.5s；新增侧倾诊断窗口0.8..1.4s。落地峰值取1.1..1.8s最长八连杆失载区间后至2.0s；失载不覆盖base/hip，不能单独证明整机腾空。
- 驱动力矩不包含全部被动碰撞载荷。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整周期/失败终止/未完成尾段=(5, 0, 1)，遥测complete，历史reset检查=True。
- fixed-start-nominal-a01 完整周期均值={"takeoff_vz_peak_m_s": 1.3130842370372833, "flight_height_rmse_cm": 2.6723619156143132, "prepare_min_height_m": 0.5393885374069214, "landing_min_height_m": 0.4596964716911316, "eight_link_unloading_ms": 230.0, "landing_peak_n": 1900.3195190429688}
- fixed-start-nominal-a01 起跳髋峰值={"mirror_error_deg": 7.317076643676171, "left_deg": 10.173893764040507, "right_deg": -2.8568171203643358, "phase_s": 1.24, "episode_id": 0}
- fixed-start-nominal-a01 风险峰值：roll=3.0057576432811666deg，pitch=14.436578799518603deg，水平anchor偏差=7.36541145402507cm，小腿/大腿地面接触=0.0N。
- fixed-start-nominal-a01 起跳侧倾0.8..1.4s指标={"roll_abs_peak_deg": 3.0057576432811666, "roll_abs_p95_deg": 2.967437494834415, "body_x_angular_velocity_abs_peak_rad_s": 1.2315057516098022, "body_x_angular_velocity_rms_rad_s": 0.4869261093399003}；baseline={"roll_abs_peak_deg": 9.78638125084813, "roll_abs_p95_deg": 8.904819694078203, "body_x_angular_velocity_abs_peak_rad_s": 2.197911024093628, "body_x_angular_velocity_rms_rad_s": 0.748532680151954}
- fixed-start-nominal-a01 起跳前1.0..1.3s指标={"roll_abs_peak_deg": 3.0057576432811666, "left_hip_abs_peak_deg": 10.173893764040507, "right_hip_abs_peak_deg": 7.877648198105689, "wheel_fore_aft_separation_peak_cm": 3.947131525958137}
- fixed-start-nominal-a01 相对上一版 takeoff_vz_peak_m_s: 1.2293738156280583 -> 1.3130842370372833
- fixed-start-nominal-a01 相对上一版 landing_peak_n: 2876.2991943359375 -> 1900.3195190429688
- fixed-start-nominal-a01 相对上一版髋镜像峰值：{'mirror_error_deg': 6.512123268621124, 'left_deg': 3.6191699652357783, 'right_deg': 2.892953303385346, 'phase_s': 1.22, 'episode_id': 0} -> {'mirror_error_deg': 7.317076643676171, 'left_deg': 10.173893764040507, 'right_deg': -2.8568171203643358, 'phase_s': 1.24, 'episode_id': 0}
- fixed-start-nominal-a01 限值事件 {"joint": "right_foot_joint", "velocity_peak_rad_s": 10.558598518371582, "velocity_rms_rad_s": 1.0911070530702678, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 6, "overspeed_total_ms": 30.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.59, "velocity_peak_episode": 0, "torque_peak_nm": 20.994991302490234, "torque_rms_nm": 2.6919346960117494, "effort_limit_nm": 27.0, "saturation_samples": 0, "saturation_total_ms": 0.0, "saturation_longest_ms": 0.0}
- fixed-start-randomized-a01: 完整周期/失败终止/未完成尾段=(5, 0, 1)，遥测complete，历史reset检查=True。
- fixed-start-randomized-a01 完整周期均值={"takeoff_vz_peak_m_s": 1.4070783386701964, "flight_height_rmse_cm": 1.8015623046628086, "prepare_min_height_m": 0.5454800724983215, "landing_min_height_m": 0.4651971638202667, "eight_link_unloading_ms": 239.0, "landing_peak_n": 1260.4101470947267}
- fixed-start-randomized-a01 起跳髋峰值={"mirror_error_deg": 8.023833883137826, "left_deg": 13.962445163616609, "right_deg": -5.938611280478783, "phase_s": 1.24, "episode_id": 1}
- fixed-start-randomized-a01 风险峰值：roll=7.066291435347417deg，pitch=15.27962927826456deg，水平anchor偏差=46.897891321148926cm，小腿/大腿地面接触=0.0N。
- fixed-start-randomized-a01 起跳侧倾0.8..1.4s指标={"roll_abs_peak_deg": 7.066291435347417, "roll_abs_p95_deg": 4.9859646336602355, "body_x_angular_velocity_abs_peak_rad_s": 1.6532628536224365, "body_x_angular_velocity_rms_rad_s": 0.5241502820612741}；baseline={"roll_abs_peak_deg": 12.330805058973105, "roll_abs_p95_deg": 9.100918086845523, "body_x_angular_velocity_abs_peak_rad_s": 2.357675075531006, "body_x_angular_velocity_rms_rad_s": 0.6915758704505993}
- fixed-start-randomized-a01 起跳前1.0..1.3s指标={"roll_abs_peak_deg": 6.4813047720613, "left_hip_abs_peak_deg": 14.192463736747836, "right_hip_abs_peak_deg": 11.07894334793868, "wheel_fore_aft_separation_peak_cm": 7.916097135434362}
- fixed-start-randomized-a01 相对上一版 takeoff_vz_peak_m_s: 1.2378190889456966 -> 1.4070783386701964
- fixed-start-randomized-a01 相对上一版 landing_peak_n: 1420.686489868164 -> 1260.4101470947267
- fixed-start-randomized-a01 相对上一版髋镜像峰值：{'mirror_error_deg': 10.235479658795807, 'left_deg': -2.2010794730119727, 'right_deg': -8.034400185783836, 'phase_s': 1.3, 'episode_id': 2} -> {'mirror_error_deg': 8.023833883137826, 'left_deg': 13.962445163616609, 'right_deg': -5.938611280478783, 'phase_s': 1.24, 'episode_id': 1}
- fixed-start-randomized-a01 限值事件 {"joint": "left_shank_joint", "velocity_peak_rad_s": 15.491990089416504, "velocity_rms_rad_s": 3.147653758484174, "nominal_velocity_limit_rad_s": 20.0, "overspeed_samples": 0, "overspeed_total_ms": 0.0, "overspeed_longest_ms": 0.0, "velocity_peak_phase_s": 1.455, "velocity_peak_episode": 3, "torque_peak_nm": 120.0, "torque_rms_nm": 30.798219028342885, "effort_limit_nm": 120.0, "saturation_samples": 1, "saturation_total_ms": 5.0, "saturation_longest_ms": 5.0}
- fixed-start-randomized-a01 限值事件 {"joint": "left_foot_joint", "velocity_peak_rad_s": 11.213696479797363, "velocity_rms_rad_s": 1.2739064813343157, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 3, "overspeed_total_ms": 15.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.625, "velocity_peak_episode": 4, "torque_peak_nm": 23.96525001525879, "torque_rms_nm": 3.5105084573067145, "effort_limit_nm": 27.0, "saturation_samples": 0, "saturation_total_ms": 0.0, "saturation_longest_ms": 0.0}
- fixed-start-randomized-a01 限值事件 {"joint": "right_foot_joint", "velocity_peak_rad_s": 26.407920837402344, "velocity_rms_rad_s": 1.216475034116406, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 3, "overspeed_total_ms": 15.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.62, "velocity_peak_episode": 0, "torque_peak_nm": 27.0, "torque_rms_nm": 2.7853994572177045, "effort_limit_nm": 27.0, "saturation_samples": 1, "saturation_total_ms": 5.0, "saturation_longest_ms": 5.0}
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [hip-alignment-comparison.png](<derived/hip-alignment-comparison.png>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [jump-contact-comparison.png](<derived/jump-contact-comparison.png>)
- [protection-after-20261104-001.json](<../../evidence/training/protection-after-20261104-001.json>)
- [protection-before-20261104-001.json](<../../evidence/training/protection-before-20261104-001.json>)
- [training-origin-20261104-001.json](<../../evidence/training/training-origin-20261104-001.json>)
- [training-windows-20261104-001.json](<../../evidence/training/training-windows-20261104-001.json>)
- [manifest.json](<../../../2026-11-02_14-03-48/evaluations/wheel-to-leg-fixed-start-20261103-001/manifest.json>)

## 建议与待授权事项

- 优先结合名义和随机化两组的髋偏斜、机身roll、起跳速度及落地冲击进行判断；不能仅凭训练reward或motion_finished推荐部署。
- 本次比较只覆盖单次训练、单个评估seed；参数效果需更多受控证据确认。后续训练参数改动先提出方案并获批准。
- 原始证据、训练checkpoint及既有导出文件均保持原样；本次未训练、导出、安装或执行Git提交。

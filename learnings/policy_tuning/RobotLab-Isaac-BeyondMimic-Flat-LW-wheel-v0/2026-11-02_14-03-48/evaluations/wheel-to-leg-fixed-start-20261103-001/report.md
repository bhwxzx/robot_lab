# 评估批次 wheel-to-leg-fixed-start-20261103-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-02_14-03-48`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`6cdf99aac29b275e9ff72d4f519cbbd91da683b748a93b6cf2d9fb142f19e504`；runner：`OnPolicyRunner`。
- [场景来源 scenario-61b703d970f527e7b3bc1de15f1c391e891650b6889b1678f55a4e858437587b.json](<../../provenance/scenario-61b703d970f527e7b3bc1de15f1c391e891650b6889b1678f55a4e858437587b.json>)
- [场景来源 scenario-d6044dfbaeab2d5607ac0cf8584c6ff5fd444a61ab7bc5ba617c36c6e26d7945.json](<../../provenance/scenario-d6044dfbaeab2d5607ac0cf8584c6ff5fd444a61ab7bc5ba617c36c6e26d7945.json>)
- [训练上下文 context-a8719c2e0f75ae5edcc2cd4f526d12a4bfec820a82881353c1324d48b8ef0550.json](<../../provenance/context-a8719c2e0f75ae5edcc2cd4f526d12a4bfec820a82881353c1324d48b8ef0550.json>)
- [训练有效配置 config-9a183e1b81aa1ccf72078917f8cc7bc337b9fc129c12b1f3b717ec59654d383a.json](<../../provenance/config-9a183e1b81aa1ccf72078917f8cc7bc337b9fc129c12b1f3b717ec59654d383a.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 本机 server 场次2026-11-02_14-03-48/model_49999.pt；SHA256 6cdf99aac29b275e9ff72d4f519cbbd91da683b748a93b6cf2d9fb142f19e504；PPO/OnPolicyRunner，8192环境、50000轮、seed42。
- 用户确认启动命令 bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra参数。训练源e731215，评估源30485c8；两者训练相关源码字节相同，后者仅新增另一主机的评估证据。
- 相对本机10月31日版本，奖励和agent配置相同；anchor_pos终止阈值0.15变为0.25m，新增诊断配置默认关闭。运行日/日志路径也不同。正常结束率不能作为相同终止条件下的成功率比较。
- Native固定第0帧名义/训练随机化两组，各1环境、1000控制步、seed42、无视频；控制周期0.02s、物理周期0.005s。均值动作，无网络更新。
- 执行包装器仅给评估进程追加 --kit_args=--/app/settings/persistent=false；hip_pos_zero_l2=-2与保存配置一致，保持旧场景契约。
- 末1000轮采样top1概率均值=0.44568566685914995，归一化熵=0.8464658117890358，top1归一化bin主要0.5（第三段，失败计数帧85..127，约1.70..2.54s）；迭代日志是均值，不能恢复精确四段采样次数。
- 末5000轮motion_finished均值=0.9956921568036079；训练随机相位正常结束指标并非完整动作成功率，收敛判定仍indeterminate。
- 周期均值仅使用完整动作；风险峰值包含所有采样。名义重复周期不是独立种子，随机化仅单个seed。
- 起跳速度窗口1.1..1.36s；髋镜像窗口1.0..1.5s；落地峰值取1.1..1.8s最长八连杆失载区间后至2.0s。八连杆失载不覆盖base/hip，不能单独证明整机腾空。
- 实际驱动力矩不包含全部被动碰撞载荷；超速/饱和累计时间与最长连续时间分别记录。仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整周期/失败终止/未完成尾段=(5, 0, 1)，遥测complete，历史reset检查=True。
- fixed-start-nominal-a01 完整周期均值={"takeoff_vz_peak_m_s": 1.2293738156280583, "flight_height_rmse_cm": 3.9085477399032116, "prepare_min_height_m": 0.5469522476196289, "landing_min_height_m": 0.4692884385585785, "eight_link_unloading_ms": 230.0, "landing_peak_n": 2876.2991943359375}
- fixed-start-nominal-a01 起跳髋峰值={"mirror_error_deg": 6.512123268621124, "left_deg": 3.6191699652357783, "right_deg": 2.892953303385346, "phase_s": 1.22, "episode_id": 0}
- fixed-start-nominal-a01 风险峰值：roll=9.78638125084813deg，pitch=8.551212209118518deg，水平anchor偏差=8.917499339961447cm，小腿/大腿地面接触=0.0N。
- fixed-start-nominal-a01 起跳前1.0..1.3s指标={"roll_abs_peak_deg": 9.78638125084813, "left_hip_abs_peak_deg": 3.670003221246775, "right_hip_abs_peak_deg": 3.6664280441030965, "wheel_fore_aft_separation_peak_cm": 3.2952180409589777}
- fixed-start-nominal-a01 相对上一版 takeoff_vz_peak_m_s: 1.0554730588994148 -> 1.2293738156280583
- fixed-start-nominal-a01 相对上一版 landing_peak_n: 1753.2488098144531 -> 2876.2991943359375
- fixed-start-nominal-a01 相对上一版髋镜像峰值：{'mirror_error_deg': 6.446576077590628, 'left_deg': 3.7099483019471253, 'right_deg': 2.7366277756435022, 'phase_s': 1.22, 'episode_id': 0} -> {'mirror_error_deg': 6.512123268621124, 'left_deg': 3.6191699652357783, 'right_deg': 2.892953303385346, 'phase_s': 1.22, 'episode_id': 0}
- fixed-start-randomized-a01: 完整周期/失败终止/未完成尾段=(5, 0, 1)，遥测complete，历史reset检查=True。
- fixed-start-randomized-a01 完整周期均值={"takeoff_vz_peak_m_s": 1.2378190889456966, "flight_height_rmse_cm": 3.664399783800911, "prepare_min_height_m": 0.540197217464447, "landing_min_height_m": 0.459157258272171, "eight_link_unloading_ms": 227.0, "landing_peak_n": 1420.686489868164}
- fixed-start-randomized-a01 起跳髋峰值={"mirror_error_deg": 10.235479658795807, "left_deg": -2.2010794730119727, "right_deg": -8.034400185783836, "phase_s": 1.3, "episode_id": 2}
- fixed-start-randomized-a01 风险峰值：roll=12.330805058973105deg，pitch=18.297975180237668deg，水平anchor偏差=30.328324272756845cm，小腿/大腿地面接触=0.0N。
- fixed-start-randomized-a01 起跳前1.0..1.3s指标={"roll_abs_peak_deg": 12.330805058973105, "left_hip_abs_peak_deg": 11.410169102870215, "right_hip_abs_peak_deg": 10.67695683320581, "wheel_fore_aft_separation_peak_cm": 3.968314226907971}
- fixed-start-randomized-a01 相对上一版 takeoff_vz_peak_m_s: 1.152217122803816 -> 1.2378190889456966
- fixed-start-randomized-a01 相对上一版 landing_peak_n: 1387.382778930664 -> 1420.686489868164
- fixed-start-randomized-a01 相对上一版髋镜像峰值：{'mirror_error_deg': 12.945222315762258, 'left_deg': 10.176944297278563, 'right_deg': 2.768278018483695, 'phase_s': 1.4, 'episode_id': 3} -> {'mirror_error_deg': 10.235479658795807, 'left_deg': -2.2010794730119727, 'right_deg': -8.034400185783836, 'phase_s': 1.3, 'episode_id': 2}
- fixed-start-randomized-a01 限值事件 {"joint": "left_foot_joint", "velocity_peak_rad_s": 15.239202499389648, "velocity_rms_rad_s": 1.3064826359401909, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 3, "overspeed_total_ms": 15.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.585, "velocity_peak_episode": 5, "torque_peak_nm": 27.0, "torque_rms_nm": 4.258906681799088, "effort_limit_nm": 27.0, "saturation_samples": 1, "saturation_total_ms": 5.0, "saturation_longest_ms": 5.0}
- fixed-start-randomized-a01 限值事件 {"joint": "right_foot_joint", "velocity_peak_rad_s": 10.080896377563477, "velocity_rms_rad_s": 1.2143449736950027, "nominal_velocity_limit_rad_s": 10.0, "overspeed_samples": 2, "overspeed_total_ms": 10.0, "overspeed_longest_ms": 5.0, "velocity_peak_phase_s": 1.64, "velocity_peak_episode": 3, "torque_peak_nm": 20.9532527923584, "torque_rms_nm": 2.5800255819279947, "effort_limit_nm": 27.0, "saturation_samples": 0, "saturation_total_ms": 0.0, "saturation_longest_ms": 0.0}
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [hip-alignment-comparison.png](<derived/hip-alignment-comparison.png>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [jump-contact-comparison.png](<derived/jump-contact-comparison.png>)
- [protection-after-20261103-001.json](<../../evidence/training/protection-after-20261103-001.json>)
- [protection-before-20261103-001.json](<../../evidence/training/protection-before-20261103-001.json>)
- [training-origin-20261103-001.json](<../../evidence/training/training-origin-20261103-001.json>)
- [training-windows-20261103-001.json](<../../evidence/training/training-windows-20261103-001.json>)
- [manifest.json](<../../../2026-10-31_15-48-13/evaluations/wheel-to-leg-fixed-start-20261102-001/manifest.json>)

## 建议与待授权事项

- 优先结合名义和随机化两组的髋偏斜、机身roll、起跳速度及落地冲击进行判断；不能仅凭训练reward或motion_finished推荐部署。
- 本次比较只覆盖单次训练、单个评估seed；参数效果需更多受控证据确认。后续训练参数改动先提出方案并获批准。
- 原始证据、训练checkpoint及既有导出文件均保持原样；本次未训练、导出、安装或执行Git提交。

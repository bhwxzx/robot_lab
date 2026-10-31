# 评估批次 wheel-to-leg-fixed-start-20261031-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-30_11-48-12`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`21388bfa4e76eb15bc2d0321c09461a9ba5cccfe071af35d77fc070448333b03`；runner：`OnPolicyRunner`。
- [场景来源 scenario-4b83d634cc25acb90ef9154fad3a8f41f633600a20ae0670423c9a830d49e519.json](<../../provenance/scenario-4b83d634cc25acb90ef9154fad3a8f41f633600a20ae0670423c9a830d49e519.json>)
- [场景来源 scenario-64a007b317529b885ba732cd0f6d41a2a6de32e8310a1b89961d5723c7fd2962.json](<../../provenance/scenario-64a007b317529b885ba732cd0f6d41a2a6de32e8310a1b89961d5723c7fd2962.json>)
- [训练上下文 context-5f2bbedca2a4511f7541267de8c488a5c86f3fbda3676d6e8716ef5cdf9692a8.json](<../../provenance/context-5f2bbedca2a4511f7541267de8c488a5c86f3fbda3676d6e8716ef5cdf9692a8.json>)
- [训练有效配置 config-753d2d53fb9bb20e652cbde7d5dfb89d982a49e6a0fbcb74f1d7febb64881f7a.json](<../../provenance/config-753d2d53fb9bb20e652cbde7d5dfb89d982a49e6a0fbcb74f1d7febb64881f7a.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.hip_pos_zero_l2.weight": -2.0}`。

## 观察与限制

- 本机server场次2026-10-30_11-48-12/model_49999.pt，SHA256 21388bfa4e76eb15bc2d0321c09461a9ba5cccfe071af35d77fc070448333b03；PPO/OnPolicyRunner，seed42，8192环境、50000轮配置，最终checkpoint迭代49999。
- 训练来源为用户确认的bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra参数；W&B记录训练HEAD feb3583且训练git记录干净。当前HEAD 3ce5db8只有髋归零权重从-2改为-3；本次评估仅进程内覆盖为保存训练值-2。
- 名义/训练随机化两组各固定frame0，1环境、1000控制步、0.02s控制周期，完整窗口每控制步4个0.005s物理样本；无视频。名义关闭噪声/随机化并delay0，随机化保留训练随机化与0..3步延迟。
- 执行包装器仅添加--kit_args=--/app/settings/persistent=false；实际argv保存在每组console.log。完整遥测及物理子步数据保留。
- 训练末5000轮motion_finished均值0.994773，不能等同完整动作成功率；末1000轮困难采样top1均值0.482536、归一化bin恒0.25，即第2/4时间段（重置frame42..84，约0.84..1.68s），不再由末段独占。
- 完整周期均值排除未完成尾段；风险极值保留所有观测。名义重复周期并非独立种子，随机化只含单seed的一次startup抽样。
- 指标窗口：起跳速度1.1..1.36s，腾空高度误差1.2..1.7s；髋镜像峰值1.0..1.5s。落地冲击取1.1..1.8s最长八连杆失载区间后首次再接触至2.0s的总法向Fz峰值；各连杆力范数<=1N定义失载。
- 八连杆失载不覆盖base/hip，不能单独证明整机腾空；驱动力矩不含全部被动碰撞载荷。关节超速与饱和按模型名义限值统计，累计时长与最长连续时长分开记录。
- 历史对照均复用已验证同场景、同seed、同步数原始遥测，并用相同计算函数重算；训练参数和训练步数不同，不构成单参数因果试验，也不是全部历史checkpoint排名。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- fixed-start-nominal-a01: 完整周期5，失败终止0，未完成尾段1；遥测complete，控制/物理样本1000/4000，历史reset检查True。
- fixed-start-nominal-a01 完整周期均值：{"takeoff_vz_peak_m_s": 1.4769405452466087, "flight_height_rmse_cm": 2.647340238633643, "prepare_min_height_m": 0.5419126152992249, "landing_min_height_m": 0.45877113938331604, "eight_link_unloading_ms": 215.0, "landing_peak_n": 923.4376831054688}
- fixed-start-nominal-a01 全采样风险峰值：roll 10.938°，pitch 7.714°，水平anchor误差16.283cm，八连杆总Fz 923.438N；大腿/小腿地面接触0.000N，自碰0.000N。
- fixed-start-nominal-a01 起跳髋峰值：{"mirror_error_deg": 12.20988201528457, "left_deg": 8.865978278553827, "right_deg": 3.343903736730744, "phase_s": 1.22, "episode_id": 0}；全动作镜像峰值：{"hip": 12.20988201528457, "thigh": 25.39374516423155, "shank": 41.73126054567316, "foot": 10.424439616948561}
- fixed-start-nominal-a01 left_foot_joint: 速度峰值7.875rad/s（相位1.620s），力矩峰值17.933Nm；超速0样本、累计0.0ms/最长连续0.0ms；饱和0样本、累计0.0ms/最长连续0.0ms。
- fixed-start-nominal-a01 right_foot_joint: 速度峰值7.889rad/s（相位1.575s），力矩峰值20.791Nm；超速0样本、累计0.0ms/最长连续0.0ms；饱和0样本、累计0.0ms/最长连续0.0ms。
- fixed-start-randomized-a01: 完整周期5，失败终止0，未完成尾段1；遥测complete，控制/物理样本1000/4000，历史reset检查True。
- fixed-start-randomized-a01 完整周期均值：{"takeoff_vz_peak_m_s": 1.2847663421410487, "flight_height_rmse_cm": 2.288449989413636, "prepare_min_height_m": 0.5748575687408447, "landing_min_height_m": 0.4555229783058167, "eight_link_unloading_ms": 232.0, "landing_peak_n": 1206.6432800292969}
- fixed-start-randomized-a01 全采样风险峰值：roll 11.703°，pitch 18.165°，水平anchor误差49.725cm，八连杆总Fz 1735.309N；大腿/小腿地面接触0.000N，自碰0.000N。
- fixed-start-randomized-a01 起跳髋峰值：{"mirror_error_deg": 10.878809333705917, "left_deg": -0.3584508089219286, "right_deg": -10.520358524783989, "phase_s": 1.3, "episode_id": 4}；全动作镜像峰值：{"hip": 11.45529744337696, "thigh": 27.004585942710644, "shank": 37.82257380255935, "foot": 17.120037348045273}
- fixed-start-randomized-a01 left_foot_joint: 速度峰值10.004rad/s（相位1.600s），力矩峰值23.984Nm；超速1样本、累计5.0ms/最长连续5.0ms；饱和0样本、累计0.0ms/最长连续0.0ms。
- fixed-start-randomized-a01 right_foot_joint: 速度峰值10.321rad/s（相位1.655s），力矩峰值20.351Nm；超速3样本、累计15.0ms/最长连续5.0ms；饱和0样本、累计0.0ms/最长连续0.0ms。
- 对照previous_working/fixed-start-nominal-a01：完整周期/失败终止 5/0 → 5/0；起跳速度 1.2330572671093543 → 1.4769405452466087 m/s；平均落地峰值 2380.9115600585938 → 923.4376831054688 N；起跳髋镜像峰值 5.696843287871253 → 12.20988201528457 deg。
- 对照previous_working/fixed-start-randomized-a01：完整周期/失败终止 5/0 → 5/0；起跳速度 1.1145939342634543 → 1.2847663421410487 m/s；平均落地峰值 1445.0841125488282 → 1206.6432800292969 N；起跳髋镜像峰值 9.843984581432343 → 10.878809333705917 deg。
- 对照previous_run/fixed-start-nominal-a01：完整周期/失败终止 0/18 → 5/0；起跳速度 None → 1.4769405452466087 m/s；平均落地峰值 None → 923.4376831054688 N；起跳髋镜像峰值 11.15621968132138 → 12.20988201528457 deg。
- 对照previous_run/fixed-start-randomized-a01：完整周期/失败终止 0/29 → 5/0；起跳速度 None → 1.2847663421410487 m/s；平均落地峰值 None → 1206.6432800292969 N；起跳髋镜像峰值 None → 10.878809333705917 deg。
- 对照A_8192_model25000/fixed-start-nominal-a01：完整周期/失败终止 5/0 → 5/0；起跳速度 1.3504987004288715 → 1.4769405452466087 m/s；平均落地峰值 1122.9378662109375 → 923.4376831054688 N；起跳髋镜像峰值 7.161418340038873 → 12.20988201528457 deg。
- 对照A_8192_model25000/fixed-start-randomized-a01：完整周期/失败终止 5/0 → 5/0；起跳速度 1.2411225543134612 → 1.2847663421410487 m/s；平均落地峰值 1426.7475006103516 → 1206.6432800292969 N；起跳髋镜像峰值 9.353627745196526 → 10.878809333705917 deg。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [jump-contact-comparison.png](<derived/jump-contact-comparison.png>)
- [hip-shank-comparison.png](<derived/hip-shank-comparison.png>)
- [protection-after-20261031-001.json](<../../evidence/training/protection-after-20261031-001.json>)
- [protection-before-20261031-001.json](<../../evidence/training/protection-before-20261031-001.json>)
- [training-origin-20261031-001.json](<../../evidence/training/training-origin-20261031-001.json>)
- [training-windows-20261031-001.json](<../../evidence/training/training-windows-20261031-001.json>)

## 建议与待授权事项

- 依据完整动作回放与物理子步峰值评估髋偏斜、落地和足部限值；保留历史原始证据。
- 训练环境数及多个奖励参数共同变化，不能将改善或退化独立归因于一个权重；参数修改须先提出具体方案并获批准。
- 单seed测试不代表普遍鲁棒性或收敛；如需扩大测试，下一步采用明确有限的随机种子集合及同一动作窗口。

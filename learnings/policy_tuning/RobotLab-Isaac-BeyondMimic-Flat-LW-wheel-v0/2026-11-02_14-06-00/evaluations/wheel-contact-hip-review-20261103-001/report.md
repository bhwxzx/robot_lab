# 评估批次 wheel-contact-hip-review-20261103-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-02_14-06-00`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<../wheel-to-leg-fixed-start-20261103-001/raw/fixed-start-nominal-a01/result.json>) / [日志](<../wheel-to-leg-fixed-start-20261103-001/raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<../wheel-to-leg-fixed-start-20261103-001/raw/fixed-start-randomized-a01/result.json>) / [日志](<../wheel-to-leg-fixed-start-20261103-001/raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`493ebd7fbecb527f763de79d45cf884b742bd001cea28d8ff5130b260783165a`；runner：`OnPolicyRunner`。
- [场景来源 scenario-bd7cec1cba2fd2b3be0d88477cbb53e894f6aa31b85eadf2d61f5ef667daf7d6.json](<../../provenance/scenario-bd7cec1cba2fd2b3be0d88477cbb53e894f6aa31b85eadf2d61f5ef667daf7d6.json>)
- [场景来源 scenario-df7f38a2edb6ae94a9eaedef0593de7a7a7f8ff57baccbf5f07fe155c3059154.json](<../../provenance/scenario-df7f38a2edb6ae94a9eaedef0593de7a7a7f8ff57baccbf5f07fe155c3059154.json>)
- [训练上下文 context-51a50409569313b0f83f235be701404c6ab754a9989edfca970d9134feecc7c2.json](<../../provenance/context-51a50409569313b0f83f235be701404c6ab754a9989edfca970d9134feecc7c2.json>)
- [训练有效配置 config-4728f2258c6b63e7673cdf93612581c7d5a9376cf4af83912adfc8497915ce9d.json](<../../provenance/config-4728f2258c6b63e7673cdf93612581c7d5a9376cf4af83912adfc8497915ce9d.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.undesired_contacts.weight": -1.0, "terminations.thigh_shank_ground_contact": null}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.undesired_contacts.weight": -1.0, "terminations.thigh_shank_ground_contact": null}`。

## 观察与限制

- 本批次只分析已完成两组Native评估的原始遥测，未追加仿真；引用原始result/telemetry，不复制或改写旧证据。
- 结论：两组均完成动作序列，但每个观测周期都发生大腿触地；起跳高度不足，1.5s后的髋突变仍明显。1.0–1.5s的局部指标改善不能代表过渡动作已解决。
- 本文相位数值采用记录的frame_index/50；图中控制采样加20ms、物理采样按子步加5ms，以对齐物理步末。接触力为仿真物理步平均法向接触力，不可直接视作实物瞬时峰值。
- 大腿触地终止已关闭，所以motion_finished不能用来判定无碰撞完成。时间上相邻支持碰撞与突变相关，不能单凭本测试证明唯一因果。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- nominal: 大腿接触出现在6个观测周期（含5个完整周期和1个未完整尾段）；峰值5.220kN，连杆right_thigh_link，参考帧相位1.520s。
- nominal: 扩大到1.0–1.8s窗口，髋镜像误差峰值8.475°；右髋20ms最大变化3.858°，相位1.520→1.540s。
- nominal: 起跳窗口机身最高高度0.621m，参考0.714m；全程高度误差峰值22.17cm。八连杆最长同时失载0.035s，不涵盖base/hip接触，不能独立证明整体腾空。
- randomized: 大腿接触出现在6个观测周期（含5个完整周期和1个未完整尾段）；峰值6.759kN，连杆left_thigh_link，参考帧相位1.540s。
- randomized: 扩大到1.0–1.8s窗口，髋镜像误差峰值8.786°；右髋20ms最大变化3.111°，相位1.540→1.560s。
- randomized: 起跳窗口机身最高高度0.626m，参考0.714m；全程高度误差峰值21.74cm。八连杆最长同时失载0.105s，不涵盖base/hip接触，不能独立证明整体腾空。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [manifest.json](<../wheel-to-leg-fixed-start-20261103-001/manifest.json>)
- [contact-hip-analysis.json](<derived/contact-hip-analysis.json>)
- [contact-hip-alignment.png](<derived/contact-hip-alignment.png>)

## 建议与待授权事项

- 本版本不能仅凭较高motion_finished和采样熵认定为优化成功。优先把无大腿触地、起跳高度跟踪和髋突变同时作为后续对照的观察项。
- 若优化，先提出单变量对照方案并获用户批准；本轮未改配置、启动训练、导出或提交。

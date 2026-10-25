# 评估批次 wheel-to-leg-local-20261025-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-24_13-38-14`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`5231c5bc2e4aaaa68ebb9dc70c40244bfc2d6ca45acb25cbeefaba9b44012786`；runner：`OnPolicyRunner`。
- [场景来源 scenario-8fc9e35ec5d093718e004cc0885bbcb07fa5c010f50e83a7ebc83a45571e0ce4.json](<../../provenance/scenario-8fc9e35ec5d093718e004cc0885bbcb07fa5c010f50e83a7ebc83a45571e0ce4.json>)
- [场景来源 scenario-9aa949190d3e83617bf7d19983e3cd43431f545f6a30b5d2125d89e349aef18e.json](<../../provenance/scenario-9aa949190d3e83617bf7d19983e3cd43431f545f6a30b5d2125d89e349aef18e.json>)
- [训练上下文 context-ded4329745b9badc2acf12c3b568fe018c26d7b93ec111f2853d0d53455bafe2.json](<../../provenance/context-ded4329745b9badc2acf12c3b568fe018c26d7b93ec111f2853d0d53455bafe2.json>)
- [训练有效配置 config-a51ce69f71fc7e034235acac754799f62b525ed0caf065f2b49375b43db59ce2.json](<../../provenance/config-a51ce69f71fc7e034235acac754799f62b525ed0caf065f2b49375b43db59ce2.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03"}`。

## 观察与限制

- 本机 younghit 场次 2026-10-24_13-38-14/model_49999.pt，SHA-256 5231c5bc2e4aaaa68ebb9dc70c40244bfc2d6ca45acb25cbeefaba9b44012786。用户确认 bash scripts/start_beyondmimic.sh --type wheel、无额外覆盖。
- 有效训练配置：10/23 NPZ SHA-256 8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03；anchor_pos 为基座高度误差阈值 0.15 m；action_rate_l2=-0.2，action_smoothness=-0.075；基座位置/连杆位置/连杆线速度跟踪权重 0.5/1/1。腿/轮/足延迟仍为 0..3 物理步。
- 两组 Native、seed 42、单环境各 1000 控制步/4000 物理步、无视频；遥测 complete，无缺失必需信号；接触传感器独立视图交叉校验误差 0 N。590 维分项 10 帧历史输入、10 维动作；reset 历史一致性通过运行时断言。
- play-ground-a01 从 frame 0 完成 5 周期，无非 motion_finished 终止，另有 1 个未完成尾段。5 个完整周期全部出现大腿触地；motion_finished 不能当作正确跳跃成功。
- 固定 5 周期：起跳窗口 1.10–1.36 s 向上速度峰值 1.117–1.238 m/s，均值 1.168，参考峰值 1.967；1.30 s 向上速度均值 1.089 m/s（参考 1.827）；1.20–1.30 s 速度 RMSE 0.503 m/s；1.20–1.70 s 高度 RMSE 5.91 cm。
- 固定 5 周期：记录八连杆连续失载 250–275 ms，均值 262 ms（参考计划腾空 300 ms），1 N 与 10 N 阈值敏感性结果一致；失载从参考时间 1.345–1.355 s 才开始。八连杆不含 base/hip，不能单独证明全身腾空。
- 固定 5 周期：下蹲最低基座高度均值 0.539 m（参考 0.490）；计划飞行窗口内基座最高约 0.701–0.710 m（参考 0.714），但落地 1.60–2.00 s 最低仅 0.362–0.385 m（参考 0.473）。接近最高高度不能替代全段跟踪。
- 固定准备阶段 0.02–1.30 s 双轮同时 ≥1 N 的比例均值 79.8%；俯仰绝对峰值最多 10.25°、横滚 4.09°，髋镜像残差峰值最多 13.51°。飞行窗口俯仰绝对峰值约 8.33–11.31°；尚有支撑和姿态偏差。
- 固定 5 周期首次大腿对地接触在参考 1.645–1.700 s；全程单大腿/小腿对地峰值 5.635 kN（left_thigh_link，1.650 s），记录对地 Fz 合力峰值 5.658 kN。准备与落地的左右小腿峰速、驱动力矩详见 joint-peak-rms.csv 和 landing-and-push-mechanics.json。
- 固定场景未见驱动力矩达到限幅；左/右小腿全程峰值 90.91/92.80 Nm。左/右足关节速度峰值 10.71/11.05 rad/s，均有 1 个物理样本超过保存配置名义 10 rad/s。
- training-random-phase-a01 完成 12 个随机起始片段，无非 motion_finished 终止，另有 1 个尾段；无 frame 0 完整周期，4/12 已完成片段出现大腿触地。只有 2 段覆盖完整起跳及落地观察窗口，不能把 12 片段当 12 次完整跳跃。
- 上述随机场景 2 个覆盖片段：起跳峰速 1.095–1.251 m/s，均值 1.173；八连杆失载 235–280 ms；高度 RMSE 均值 6.12 cm。全程左大腿对地峰值及记录合力峰值 7.362 kN（参考 1.630 s）；左足达到 27 Nm 限幅 15 个物理样本，累计 75 ms，不代表单次连续 75 ms。
- 随机场景左/右小腿驱动力矩峰值 113.41/107.20 Nm；左/右足关节峰速 10.25/10.27 rad/s。碰撞被动关节反力矩不在 applied_torque 中，不能从驱动力矩未限幅推断碰撞负载很小。
- 与本机 2026-10-23_19-58-38 同场景重算：固定起跳峰速均值 0.587→1.168 m/s，失载 167→262 ms，高度 RMSE 9.97→5.91 cm；大腿触地仍 5/5；单大腿/小腿对地峰值 4.270→5.635 kN，随机合力峰值 5.916→7.362 kN。起跳改善伴随更重落地。
- 同上对照：固定一阶动作差分平方和均值 0.497→0.444，二阶差分 0.1614→0.1605，按每周期计算且排除 reset 跨界。动作变化略减，髋镜像残差峰值均值却从 5.94° 增至 13.03°。
- 对照有效配置除日志目录外，同时改变 action_rate -0.15→-0.2、anchor_pos 0.25→0.15；同机同 seed 单次训练和有限回放不支持归因于某一个参数。完整配置差异保存在 training-origin 和 analysis JSON。
- 固定周期最大高度误差约 0.091–0.111 m，低于 0.15，因此没有触发 anchor_pos 终止。此项只检查高度，不能保证足端落地正确，也不会单独排除大腿触地。
- 训练 console 确认至 iteration 49999。末尾两个 1000 轮窗口：mean_reward 7.3230→7.4375，anchor 线速度误差 0.11635→0.11467；最新 motion_finished 均值 99.508%，主采样 bin 归一化索引 0.25，对应 4 bins 的索引 1，概率约 58.2%。这些是活动与表现描述，不是批准的收敛判据。
- 参考轨迹 README 记录源规划 IPOPT=-5（超时），严格节点可行性验收未通过；这是参考来源限制，不能据此单独认定本策略偏差的原因。
- 所有检查点、保存 env/agent YAML、既有导出、用户配置修改、参考轨迹和 Isaac Sim 持久设置的字节及 mtime 均未改变。评估 CLI 显式 --/app/settings/persistent=false。本轮未训练、修改代码、导出、部署、安装删除或提交推送。
- 5 ms 接触力是物理步平均法向力；单 seed、无视频、有限仿真不能证明普遍成功率或实物效果。缺少批准的收敛判据，收敛结论 indeterminate。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [jump-contact-play-ground-a01.png](<derived/jump-contact-play-ground-a01.png>)
- [jump-contact-training-random-phase-a01.png](<derived/jump-contact-training-random-phase-a01.png>)
- [analysis.json](<derived/analysis.json>)
- [joint-peak-rms.csv](<derived/joint-peak-rms.csv>)
- [protection-after-20261025-001.json](<../../evidence/training/protection-after-20261025-001.json>)
- [training-origin-20261025-001.json](<../../evidence/training/training-origin-20261025-001.json>)
- [training-tail-20261025-001.json](<../../evidence/training/training-tail-20261025-001.json>)
- [landing-and-push-mechanics.json](<derived/landing-and-push-mechanics.json>)
- [manifest.json](<../../../2026-10-23_19-58-38/evaluations/wheel-to-leg-local-20261024-001/manifest.json>)

## 建议与待授权事项

- 下一步优先核验落地足端几何、左右蹬伸时序与大腿触地，而不是仅用最高高度或 motion_finished 作为目标动作成功证据。
- 若扩大验证，使用新批次增加 seed；保留当前全部证据。本轮不自动改权重或进一步收紧阈值。

# 评估批次 commandamp-50000-inrange-assessment-20261127

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-CommandAmp-v0`；训练：`2026-11-26_21-23-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| straight-a01 | completed | straight，960 / 16 / 42 | complete | [结果](<../commandamp-50000-velocity-20261127-001/raw/straight-a01/result.json>) |
| straight-training-a01 | completed | straight-training，960 / 16 / 42 | complete | [结果](<../commandamp-50000-velocity-20261127-001/raw/straight-training-a01/result.json>) |
| stand-boundary-a01 | completed | stand-boundary，960 / 16 / 42 | complete | [结果](<../commandamp-50000-velocity-20261127-001/raw/stand-boundary-a01/result.json>) |
| turn-inrange-a02 | completed | turn-inrange，960 / 16 / 42 | complete | [结果](<../commandamp-50000-inrange-20261127-002/raw/turn-inrange-a02/result.json>) |
| hold-stand-a02 | completed | hold-stand，700 / 16 / 42 | complete | [结果](<../commandamp-50000-inrange-20261127-002/raw/hold-stand-a02/result.json>) |
| hold-left-a02 | completed | hold-left，700 / 16 / 42 | complete | [结果](<../commandamp-50000-inrange-20261127-002/raw/hold-left-a02/result.json>) |
| hold-right-a02 | completed | hold-right，700 / 16 / 42 | complete | [结果](<../commandamp-50000-inrange-20261127-002/raw/hold-right-a02/result.json>) |
| hold-forward-a03 | completed | hold-forward，500 / 16 / 42 | complete | [结果](<../commandamp-50000-speed-holds-20261127-003/raw/hold-forward-a03/result.json>) |
| hold-reverse-a03 | completed | hold-reverse，500 / 16 / 42 | complete | [结果](<../commandamp-50000-speed-holds-20261127-003/raw/hold-reverse-a03/result.json>) |

## 策略与场景

- checkpoint SHA-256：`ee5d545435abc93210daf4cf4a58577c983748092f771870a1829086624f6ab6`；runner：`OnPolicyRunnerAmp`。
- [场景来源 scenario-315686eec97294206886089039c2965e91ace5ed51b6f771d8c4426d452c20c9.json](<../../provenance/scenario-315686eec97294206886089039c2965e91ace5ed51b6f771d8c4426d452c20c9.json>)
- [场景来源 scenario-381b0fe3f8de8974770538867c6dc223ee28e4225e159664625aa91de5e6d67a.json](<../../provenance/scenario-381b0fe3f8de8974770538867c6dc223ee28e4225e159664625aa91de5e6d67a.json>)
- [场景来源 scenario-7496084bf093f403caa64ac5af5c81eb1ab91e49505c1bff82d04c46c2d89564.json](<../../provenance/scenario-7496084bf093f403caa64ac5af5c81eb1ab91e49505c1bff82d04c46c2d89564.json>)
- [场景来源 scenario-8b55a688d926dd35ab2f7344dd26ef98c3ef5d9bea0e16c09a070575f51dd3c5.json](<../../provenance/scenario-8b55a688d926dd35ab2f7344dd26ef98c3ef5d9bea0e16c09a070575f51dd3c5.json>)
- [场景来源 scenario-9713845ac5d96f31efc5a2ab93592f32eeabb6a751e080cdc3369569b2a3781a.json](<../../provenance/scenario-9713845ac5d96f31efc5a2ab93592f32eeabb6a751e080cdc3369569b2a3781a.json>)
- [场景来源 scenario-9c0787902291c7cf423e9743dde1053738ab7ebdde0adcbce181be565216d500.json](<../../provenance/scenario-9c0787902291c7cf423e9743dde1053738ab7ebdde0adcbce181be565216d500.json>)
- [场景来源 scenario-a7d4b3f6adaf57e6e54514d842e0bef5096686c627c7990f932d6ff875dab9b4.json](<../../provenance/scenario-a7d4b3f6adaf57e6e54514d842e0bef5096686c627c7990f932d6ff875dab9b4.json>)
- [场景来源 scenario-c827e4416b15f2ce27ea9e2c7fed21da95c5f6c01aa0ad507af543b41df66ee3.json](<../../provenance/scenario-c827e4416b15f2ce27ea9e2c7fed21da95c5f6c01aa0ad507af543b41df66ee3.json>)
- [场景来源 scenario-de4c7d6568ee73c3382309390c9144e17d685ea0707b01227444be8fabffda94.json](<../../provenance/scenario-de4c7d6568ee73c3382309390c9144e17d685ea0707b01227444be8fabffda94.json>)
- [训练上下文 context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json](<../../provenance/context-b22e8c643f8c955047f7c23ab89f996282ef7dc5455869046c98606e95abfb91.json>)
- [训练有效配置 config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json](<../../provenance/config-9e79991ae8b2a0be75322c5cfaf90f2e751bc82883ea82397dac2145c28b518f.json>)

- `straight-a01` 指标：`{"max_joint_velocity_utilization": 1.4040807723999023, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.23866226044483452, "tracking_yaw_rmse": 0.1337179698862979}`
- `straight-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 159, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 319, "start_step": 160}, {"command": [0.8, 0, 0], "end_step": 479, "start_step": 320}, {"command": [0, 0, 0], "end_step": 639, "start_step": 480}, {"command": [-0.4, 0, 0], "end_step": 799, "start_step": 640}, {"command": [-0.8, 0, 0], "end_step": 959, "start_step": 800}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `straight-training-a01` 指标：`{"max_joint_velocity_utilization": 1.2904970169067382, "max_tilt": 0.33941224217414856, "termination_rate": 0.0, "tracking_xy_rmse": 0.2647174363124286, "tracking_yaw_rmse": 0.16880731509146157}`
- `straight-training-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 159, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 319, "start_step": 160}, {"command": [0.8, 0, 0], "end_step": 479, "start_step": 320}, {"command": [0, 0, 0], "end_step": 639, "start_step": 480}, {"command": [-0.4, 0, 0], "end_step": 799, "start_step": 640}, {"command": [-0.8, 0, 0], "end_step": 959, "start_step": 800}]`；训练配置覆盖：`{}`。
- `stand-boundary-a01` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.15549172357729388, "tracking_yaw_rmse": 0.10189587696910887}`
- `stand-boundary-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 159, "start_step": 0}, {"command": [0.1, 0, 0], "end_step": 319, "start_step": 160}, {"command": [0.12, 0, 0], "end_step": 479, "start_step": 320}, {"command": [0, 0, 0.1], "end_step": 639, "start_step": 480}, {"command": [0, 0, 0.12], "end_step": 799, "start_step": 640}, {"command": [0, 0, 0], "end_step": 959, "start_step": 800}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `turn-inrange-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.2095124872654573, "tracking_yaw_rmse": 0.2720334647325634}`
- `turn-inrange-a02` 命令调度：`[{"command": [0, 0, 0], "end_step": 159, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 319, "start_step": 160}, {"command": [0, 0, -0.35], "end_step": 479, "start_step": 320}, {"command": [0.4, 0, 0.35], "end_step": 639, "start_step": 480}, {"command": [0.4, 0, -0.35], "end_step": 799, "start_step": 640}, {"command": [0, 0, 0], "end_step": 959, "start_step": 800}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-stand-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.05568058214501297, "tracking_yaw_rmse": 0.06504584778359235}`
- `hold-stand-a02` 命令调度：`[{"command": [0, 0, 0], "end_step": 699, "start_step": 0}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-left-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.2358314506947755, "tracking_yaw_rmse": 0.30804655098704325}`
- `hold-left-a02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.35], "end_step": 699, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-right-a02` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.2358535962585587, "tracking_yaw_rmse": 0.30249685053185055}`
- `hold-right-a02` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.35], "end_step": 699, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-forward-a03` 指标：`{"max_joint_velocity_utilization": 1.1436590194702148, "max_tilt": 0.2962498366832733, "termination_rate": 0.0, "tracking_xy_rmse": 0.24635692731855519, "tracking_yaw_rmse": 0.15376188756363124}`
- `hold-forward-a03` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。
- `hold-reverse-a03` 指标：`{"max_joint_velocity_utilization": 1.1314328193664551, "max_tilt": 0.24799273908138275, "termination_rate": 0.0, "tracking_xy_rmse": 0.29361125656938547, "tracking_yaw_rmse": 0.19170117053370633}`
- `hold-reverse-a03` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.8, 0, 0], "end_step": 499, "start_step": 100}]`；训练配置覆盖：`{"events.push_robot_hard": null, "events.randomize_apply_external_force_torque": null, "events.randomize_push_robot": null, "observations.policy.enable_corruption": false}`。

## 观察与限制

- 已核验 2026-11-26_21-23-35 的原生 model_50000.pt、训练配置、TensorBoard 和全部受评用例。训练完成 50,000 轮，seed 42。本报告采用训练指令范围内的 9 个工况，每个工况 16 个环境，共 111,040 个控制步样本，提前终止 0 次。最长单次测试为 19.2 s。
- 所有工况采用训练中的 DWAQ 微起伏平地、质量与驱动参数随机化及执行器延时。straight-training 保留训练观测噪声和周期推扰，其他用例关闭这两项。随机种子一致，可作配对比较。逐段均值、世界位移、航向和关节步态取环境 0，工况级 RMSE 与终止计数覆盖全部 16 个环境。
- 最初 1,000 轮至最后 1,000 轮的任务回报均值为 144.85 → 245.76，加权风格回报 3.41 → 9.54，混合回报 46.87 → 83.27。平均回合长度从 789.5 增至 997.3/1,000 控制步，非法接触终止由 31.80% 降至 0.60%。混合回报为 0.3 × 任务回报 + 已加权风格回报。
- 第 40,000–44,999 轮与第 45,000–49,999 轮的混合回报均值为 83.06 和 83.21，增幅 0.18%。末 5,000 轮专家和策略的判别器得分均值分别为 0.604 和 -0.604。训练记录各 52 项标量均有 50,000 个有限值。
- 持续 +0.8 和 −0.8 m/s 的配对测试均先静站 2 s，再保持速度 8 s。最后 6 s 的纵向速度均值分别为 +0.784、-0.534 m/s，达到目标幅值的 98% 和 67%。同一窗口倒退产生 9.0° 的航向变化，正向为 -0.2°。
- 持续原地转向的目标为 ±0.35 rad/s，最后 10 s 机身偏航角速度均值为 +0.033 和 -0.031 rad/s。按世界系四元数展开后的航向变化为 +24.8° 和 -23.4°，同一采样时间跨度的目标为 ±200.1°。水平净位移为 0.66 和 0.68 m。全 16 环境全工况的偏航 RMSE 分别为 0.308、0.302 rad/s。
- 弧线段指令为纵向 +0.4 m/s 与偏航 ±0.35 rad/s。两段最后 2 s 的纵向速度为 +0.427、+0.439 m/s，对应偏航只有 −0.001、−0.031 rad/s。单独切换至纵向 0.10 和 0.12 m/s 时实测为 +0.116、+0.183 m/s，单独切换至偏航 0.10 和 0.12 rad/s 时实测为 −0.010、−0.015 rad/s。站立专家门限对策略输出没有硬限幅作用。
- 全程零指令 14 s 的测试中，环境 0 舍弃头 2 s 后仍有 0.68 m 水平净漂移。16 环境全程平面速度 RMSE 为 0.056 m/s。直线换指令工况中，从 +0.8 切回零后的最后 2 s 净位移仅 0.013 m，表明停止表现受初始状态和前序动作影响。
- 稳态正向和倒退时，根据双侧大腿角差拟合得到的频率分别为 1.25、1.26 Hz，实测左右相位差为 177°、176°。匹配速度的专家巡航段为 1.25 Hz、约 180°。本次步态证据来自关节角，未采集足底接触力。
- 偏航奖励使用 exp(−误差² / 0.5²)。指令为 0.35 rad/s、实际角速度为零时，仍可得到该项最高分的约 61%。这是回报较高与实际转向不足可以并存的一项解释，尚未通过消融分离其贡献。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [training-curves-model-50000.png](<../../evidence/training/training-curves-model-50000.png>)
- [native-inrange-response.png](<../../evidence/analysis/commandamp-50000-inrange/native-inrange-response.png>)
- [training-curves-500iter.csv](<../../evidence/training/training-curves-500iter.csv>)
- [training-windows-model-50000.json](<../../evidence/training/training-windows-model-50000.json>)
- [native-segment-tracking.csv](<../../evidence/analysis/commandamp-50000-inrange/native-segment-tracking.csv>)
- [native-case-metrics.json](<../../evidence/analysis/commandamp-50000-inrange/native-case-metrics.json>)
- [native-expert-gait.csv](<../../evidence/analysis/commandamp-50000-inrange/native-expert-gait.csv>)

## 建议与待授权事项

- 若用于下一版策略比较，优先使用本报告的持续左右转、持续静站和倒退工况，分别比较实际航向、净漂移及纵向速度。

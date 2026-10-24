# 评估批次 wheel-to-leg-reference-control-20261024-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-22_11-01-24`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-20000-a01 | failed | play-ground，1000 / 1 / 42 | — | — / [日志](<raw/play-ground-20000-a01/console.log>) |

## 策略与场景


- `play-ground-20000-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `play-ground-20000-a01` 未完成原因：evaluation exit=1; see console log

## 观察与限制

- 按用户问题进行参考轨迹对照；两个训练运行保存的 agent.yaml 完全相同，env.yaml 仅 motion_file 和日志目录不同；腿部 delay 均为 0–3。使用各自 model_20000.pt 匹配训练轮数。
- 新旧均采用固定 frame-zero 开始、1000 控制步、1 环境、seed 42，无观测噪声和事件随机化；保留执行器延迟；物理步 5 ms；不录视频、不训练、不修改代码。
- play-ground-20000-a01 未完成：evaluation exit=1; see console log
- 新旧参考目标不同，不能将更低跟踪误差直接视作更强起跳；单 seed 不能证明普遍成功率，未提供批准的收敛判据。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 结合相同训练轮数的起跳、姿态、接触和冲击共同判断；本次仅做有限测试，不调整奖励参数。

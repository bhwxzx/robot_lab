# 评估批次 wheel-to-leg-latest-20261023-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-22_17-40-29`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| play-ground-a01 | completed | play-ground，1000 / 1 / 42 | complete | [结果](<raw/play-ground-a01/result.json>) / [日志](<raw/play-ground-a01/console.log>) |
| training-random-phase-a01 | completed | training-random-phase，1000 / 1 / 42 | complete | [结果](<raw/training-random-phase-a01/result.json>) / [日志](<raw/training-random-phase-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`9b42606f73222fd18d17949be0aa829fde37758ec2c99f23774722bd8cf0b952`；runner：`OnPolicyRunner`。
- [场景来源 scenario-82f5ae48c3949ffe984156750a659482caa34c7777e2cbb575dda0721da4cf62.json](<../../provenance/scenario-82f5ae48c3949ffe984156750a659482caa34c7777e2cbb575dda0721da4cf62.json>)
- [场景来源 scenario-84f30ca999aeae8839d5db6e3657e31b6c280aad4393d8ded73690496c03851b.json](<../../provenance/scenario-84f30ca999aeae8839d5db6e3657e31b6c280aad4393d8ded73690496c03851b.json>)
- [训练上下文 context-4642d359340c5b75a11c4e1a35489f11aed8711d6946d77e24490829c794656a.json](<../../provenance/context-4642d359340c5b75a11c4e1a35489f11aed8711d6946d77e24490829c794656a.json>)
- [训练有效配置 config-139b35afadcf0265f8b113ccd586f88cf1a212d7adf644125b614c524989be61.json](<../../provenance/config-139b35afadcf0265f8b113ccd586f88cf1a212d7adf644125b614c524989be61.json>)

- `play-ground-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `play-ground-a01` 命令调度：`[]`；训练配置覆盖：`{"commands.motion.joint_position_range": [0, 0], "commands.motion.pose_range": {}, "commands.motion.velocity_range": {}, "evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62", "events.add_joint_default_pos": null, "events.base_com": null, "events.physics_material": null, "events.push_robot": null, "events.randomize_actuator_gains": null, "events.randomize_rigid_body_mass_base": null, "events.randomize_rigid_body_mass_others": null, "observations.policy.enable_corruption": false, "rewards.wheel_contact_continuous.weight": 0.5}`。
- `training-random-phase-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `training-random-phase-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_phase", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "4d17c9a09f1fd6201bf14d72a468eec9c8a58000b6f67777d93b964c387f5d62"}`。

## 观察与限制

- play-ground-a01：{"完成周期": 5, "失败周期": 0, "起跳峰值vz": {"mean": 1.7475657741760515, "min": 1.650631080355652, "max": 1.7835173369458388}, "速度RMSE_1p20_1p30": {"mean": 0.32650976304408535, "min": 0.30058714266572273, "max": 0.3898081330860635}, "高度RMSE_1p20_1p70": {"mean": 0.02129859648546669, "min": 0.018352966808621096, "max": 0.02630216898611576}, "连续无支撑接触时间": {"mean": 0.281, "min": 0.275, "max": 0.28500000000000003}, "大腿对地接触周期数": 0, "统计周期范围": "complete_frame_zero"}
- training-random-phase-a01：{"完成周期": 12, "失败周期": 0, "起跳峰值vz": {"mean": 1.5969317879394822, "min": 1.5124474376646584, "max": 1.6866518858906971}, "速度RMSE_1p20_1p30": {"mean": 0.4166750115601586, "min": 0.34536045560514583, "max": 0.44733976847417994}, "高度RMSE_1p20_1p70": {"mean": 0.022394767169312702, "min": 0.008557199268290712, "max": 0.036555199370383165}, "连续无支撑接触时间": {"mean": 0.24166666666666667, "min": 0.1, "max": 0.32}, "大腿对地接触周期数": 2, "统计周期范围": "completed_partial_phase"}
- 全部评估子进程显式传入 --/app/settings/persistent=false；退出后用户配置字节及 mtime 不变，checkpoint、有效 YAML、已有导出文件哈希不变。
- 本次训练源自 W&B 记录的旧 HEAD 加保存的补丁；已验证相关源码与当前提交一致。
- 接触阈值 1 N 用于测量定义，不是用户批准的收敛判据。完整轨迹完成率不等同于成功跳跃。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [training-origin-20261023-001.json](<../../evidence/training/training-origin-20261023-001.json>)
- [jump-analysis-20261023-001.json](<../../evidence/training/jump-analysis-20261023-001.json>)
- [jump-comparison-20261023-001.png](<../../evidence/training/jump-comparison-20261023-001.png>)
- [training-tail-20261023-001.json](<../../evidence/training/training-tail-20261023-001.json>)

## 建议与待授权事项

- 按起跳速度、高度、接触力与动作/力矩指标共同审查策略；不得仅凭总奖励或 motion_finished 判定跳跃成功。
- 本次不更改训练参数、不启动训练；缺少本运行批准的判据，不作收敛认证。

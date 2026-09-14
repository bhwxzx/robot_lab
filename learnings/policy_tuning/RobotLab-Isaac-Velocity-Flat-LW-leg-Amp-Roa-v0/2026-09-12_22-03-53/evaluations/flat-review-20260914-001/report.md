# 评估批次 flat-review-20260914-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-12_22-03-53`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，1000 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [视频](<raw/stand-a01/video.mp4>) / [日志](<raw/stand-a01/console.log>) |
| forward-stop-a01 | completed | forward-stop，1000 / 1 / 42 | complete | [结果](<raw/forward-stop-a01/result.json>) / [视频](<raw/forward-stop-a01/video.mp4>) / [日志](<raw/forward-stop-a01/console.log>) |
| backward-stop-a01 | completed | backward-stop，1000 / 1 / 42 | complete | [结果](<raw/backward-stop-a01/result.json>) / [视频](<raw/backward-stop-a01/video.mp4>) / [日志](<raw/backward-stop-a01/console.log>) |
| turn-left-a01 | completed | turn-left，1000 / 1 / 42 | complete | [结果](<raw/turn-left-a01/result.json>) / [视频](<raw/turn-left-a01/video.mp4>) / [日志](<raw/turn-left-a01/console.log>) |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [视频](<raw/turn-right-a01/video.mp4>) / [日志](<raw/turn-right-a01/console.log>) |
| moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-a01/result.json>) / [视频](<raw/moving-turn-a01/video.mp4>) / [日志](<raw/moving-turn-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`c085221208af7328c74da8c260f198822b604c0388a9609c13eb0ec515971536`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-716b586a20272b3c16a97ba57cfdfe5005b847f8cfe913ab90143ceaa320e0d5.json](<../../provenance/scenario-716b586a20272b3c16a97ba57cfdfe5005b847f8cfe913ab90143ceaa320e0d5.json>)
- [场景来源 scenario-937dd0b73fad6ec11e50ebea47df01851c515b204e4ee296efeadab0ae67b6a1.json](<../../provenance/scenario-937dd0b73fad6ec11e50ebea47df01851c515b204e4ee296efeadab0ae67b6a1.json>)
- [场景来源 scenario-9ee50bc9db0f1975317a4a27c0c33598da6e4d303ed5f3f44744fbcc4912f827.json](<../../provenance/scenario-9ee50bc9db0f1975317a4a27c0c33598da6e4d303ed5f3f44744fbcc4912f827.json>)
- [场景来源 scenario-a22c39f78d6b15ffc3ea0b88a1651619d2a833a9c5b49654f59efb7697492480.json](<../../provenance/scenario-a22c39f78d6b15ffc3ea0b88a1651619d2a833a9c5b49654f59efb7697492480.json>)
- [场景来源 scenario-a295626e0ce0c0db86f50100857a2c251acdb6b21cbf9fda888001599be2263a.json](<../../provenance/scenario-a295626e0ce0c0db86f50100857a2c251acdb6b21cbf9fda888001599be2263a.json>)
- [场景来源 scenario-f76ca5d665bcff47c2c7fded714a9bdd59271078290e37f0687ffb26062fb6dd.json](<../../provenance/scenario-f76ca5d665bcff47c2c7fded714a9bdd59271078290e37f0687ffb26062fb6dd.json>)
- [训练上下文 context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json](<../../provenance/context-23d26bb9c9cb723f9847ef908f7d24a54ba254da12fd648da317ec9dc73549e5.json>)
- [训练有效配置 config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json](<../../provenance/config-4dcda3327a07455d3febf0ff28b54880e56ee007150dfb322674284ee62c0343.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 1.0100210189819336, "max_tilt": 0.11787909269332886, "termination_rate": 0.0, "tracking_xy_rmse": 0.054951060484968006, "tracking_yaw_rmse": 0.05316272396608753}`
- `stand-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 999, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `forward-stop-a01` 指标：`{"max_joint_velocity_utilization": 1.0100210189819336, "max_tilt": 0.1318841278553009, "termination_rate": 0.0, "tracking_xy_rmse": 0.1529673271747042, "tracking_yaw_rmse": 0.08455420727666997}`
- `forward-stop-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `backward-stop-a01` 指标：`{"max_joint_velocity_utilization": 1.053335475921631, "max_tilt": 0.11599422246217728, "termination_rate": 0.0, "tracking_xy_rmse": 0.18278165481759892, "tracking_yaw_rmse": 0.09566568666329384}`
- `backward-stop-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 1.0100210189819336, "max_tilt": 0.13624727725982666, "termination_rate": 0.0, "tracking_xy_rmse": 0.07211328471164344, "tracking_yaw_rmse": 0.21937641277498623}`
- `turn-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0100210189819336, "max_tilt": 0.1723351776599884, "termination_rate": 0.0, "tracking_xy_rmse": 0.08295850740893866, "tracking_yaw_rmse": 0.2114919429701807}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.0420910835266113, "max_tilt": 0.16113121807575226, "termination_rate": 0.0, "tracking_xy_rmse": 0.14348089768519628, "tracking_yaw_rmse": 0.31015766588153704}`
- `moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

# 评估批次 flat-review-20260918-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-17_23-13-31`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，1000 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [视频](<raw/stand-a01/video.mp4>) / [日志](<raw/stand-a01/console.log>) |
| forward-stop-a01 | completed | forward-stop，1000 / 1 / 42 | complete | [结果](<raw/forward-stop-a01/result.json>) / [视频](<raw/forward-stop-a01/video.mp4>) / [日志](<raw/forward-stop-a01/console.log>) |
| backward-stop-a01 | completed | backward-stop，1000 / 1 / 42 | complete | [结果](<raw/backward-stop-a01/result.json>) / [视频](<raw/backward-stop-a01/video.mp4>) / [日志](<raw/backward-stop-a01/console.log>) |
| turn-left-a01 | completed | turn-left，1000 / 1 / 42 | complete | [结果](<raw/turn-left-a01/result.json>) / [视频](<raw/turn-left-a01/video.mp4>) / [日志](<raw/turn-left-a01/console.log>) |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [视频](<raw/turn-right-a01/video.mp4>) / [日志](<raw/turn-right-a01/console.log>) |
| moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-a01/result.json>) / [视频](<raw/moving-turn-a01/video.mp4>) / [日志](<raw/moving-turn-a01/console.log>) |
| moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/moving-turn-right-a01/result.json>) / [视频](<raw/moving-turn-right-a01/video.mp4>) / [日志](<raw/moving-turn-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ac3ade5cb64737ceb3daf8f7253476b533fa3309d86164ebcf9f4caeb6e331ab`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-7781b4579903ea1f6b78d7e882a180913c2d980703acf49d8677594b99b3d903.json](<../../provenance/scenario-7781b4579903ea1f6b78d7e882a180913c2d980703acf49d8677594b99b3d903.json>)
- [场景来源 scenario-ab5361aea6c67cfac5e2e47628ed647bca6837e19d3a2621f57f8d4cac32f96b.json](<../../provenance/scenario-ab5361aea6c67cfac5e2e47628ed647bca6837e19d3a2621f57f8d4cac32f96b.json>)
- [场景来源 scenario-b885b88102d1fdbf267ffe342bdce0db620e90251ef8796ea7a62c3b7c855044.json](<../../provenance/scenario-b885b88102d1fdbf267ffe342bdce0db620e90251ef8796ea7a62c3b7c855044.json>)
- [场景来源 scenario-cf471e97000cab13c5d77b09c06f1311adf25bfe306ce741aabcf3c4758a52ea.json](<../../provenance/scenario-cf471e97000cab13c5d77b09c06f1311adf25bfe306ce741aabcf3c4758a52ea.json>)
- [场景来源 scenario-e1d5d700b1677ab0766b55f9147c6b8ac4525670866d7c8b648bb02cb9585599.json](<../../provenance/scenario-e1d5d700b1677ab0766b55f9147c6b8ac4525670866d7c8b648bb02cb9585599.json>)
- [场景来源 scenario-e29fc721144e438654faf9805b5c200c2280de564d135108acb7c29547300376.json](<../../provenance/scenario-e29fc721144e438654faf9805b5c200c2280de564d135108acb7c29547300376.json>)
- [场景来源 scenario-f36b61ba7a19a011a610822b9049f573c15a691d2b76e659e1234154a5616652.json](<../../provenance/scenario-f36b61ba7a19a011a610822b9049f573c15a691d2b76e659e1234154a5616652.json>)
- [训练上下文 context-3e2b1131a0506080679af2bf9a632fbcbcd31d87f9630dc424066d02446720a2.json](<../../provenance/context-3e2b1131a0506080679af2bf9a632fbcbcd31d87f9630dc424066d02446720a2.json>)
- [训练有效配置 config-ae9fa62a220fa36ed27740376d3bb85527a5563bc8c5a940a66b15ce667b9fda.json](<../../provenance/config-ae9fa62a220fa36ed27740376d3bb85527a5563bc8c5a940a66b15ce667b9fda.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 0.49009881019592283, "max_tilt": 0.10077513009309769, "termination_rate": 0.0, "tracking_xy_rmse": 0.056052241329684296, "tracking_yaw_rmse": 0.04381330091122299}`
- `stand-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 999, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `forward-stop-a01` 指标：`{"max_joint_velocity_utilization": 0.9714858055114746, "max_tilt": 0.13263240456581116, "termination_rate": 0.0, "tracking_xy_rmse": 0.16067604510989242, "tracking_yaw_rmse": 0.11190751551765867}`
- `forward-stop-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `backward-stop-a01` 指标：`{"max_joint_velocity_utilization": 0.8314857482910156, "max_tilt": 0.10143233835697174, "termination_rate": 0.0, "tracking_xy_rmse": 0.18974697284511435, "tracking_yaw_rmse": 0.10216096170869508}`
- `backward-stop-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 0.866032600402832, "max_tilt": 0.16349226236343384, "termination_rate": 0.0, "tracking_xy_rmse": 0.09280551174356912, "tracking_yaw_rmse": 0.20636763790462936}`
- `turn-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9031131744384766, "max_tilt": 0.16574470698833466, "termination_rate": 0.0, "tracking_xy_rmse": 0.08914145297068868, "tracking_yaw_rmse": 0.2138803964470651}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.0871654510498048, "max_tilt": 0.2047736793756485, "termination_rate": 0.0, "tracking_xy_rmse": 0.1677619678603589, "tracking_yaw_rmse": 0.30550446267317216}`
- `moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9200326919555664, "max_tilt": 0.23819610476493835, "termination_rate": 0.0, "tracking_xy_rmse": 0.15955782487131503, "tracking_yaw_rmse": 0.2808907496501408}`
- `moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

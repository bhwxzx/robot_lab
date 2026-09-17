# 评估批次 flat-review-20260917-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-16_15-45-12`。

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

- checkpoint SHA-256：`f8956bb1f9b43bdc638a52dbaf0a3301b022a4f967b581f2ed40663eb797d16a`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-0b97a3651bbce5c2c7f312e7b2ca6bc7663f4b839dce8027ab3e26ab6e401e6d.json](<../../provenance/scenario-0b97a3651bbce5c2c7f312e7b2ca6bc7663f4b839dce8027ab3e26ab6e401e6d.json>)
- [场景来源 scenario-1911f1c1407eddc68d7bde35061bfa8f38edc47a873e11568dbbe48013315687.json](<../../provenance/scenario-1911f1c1407eddc68d7bde35061bfa8f38edc47a873e11568dbbe48013315687.json>)
- [场景来源 scenario-26e715c74e04fa0add452336afb014abd9e42bb4db32a1b221156418ce3d0a70.json](<../../provenance/scenario-26e715c74e04fa0add452336afb014abd9e42bb4db32a1b221156418ce3d0a70.json>)
- [场景来源 scenario-29095609e0468a970dafadeb7afa14a03d38cb6856cbe96bc0d9ba876573b6ad.json](<../../provenance/scenario-29095609e0468a970dafadeb7afa14a03d38cb6856cbe96bc0d9ba876573b6ad.json>)
- [场景来源 scenario-4f845b1a502c8a012bfe6a176e285d16fea27d9268ee73575a90678c75e6ce1a.json](<../../provenance/scenario-4f845b1a502c8a012bfe6a176e285d16fea27d9268ee73575a90678c75e6ce1a.json>)
- [场景来源 scenario-cab8b9dedac5226dee846ed01a77a9bc931c8c91933dc16d5ff8ac2221435eea.json](<../../provenance/scenario-cab8b9dedac5226dee846ed01a77a9bc931c8c91933dc16d5ff8ac2221435eea.json>)
- [场景来源 scenario-cd3923048ed0801056d4c5878feed668021a61378216cf43558af17597b6beda.json](<../../provenance/scenario-cd3923048ed0801056d4c5878feed668021a61378216cf43558af17597b6beda.json>)
- [训练上下文 context-382b5ee84e17093ab3736ad910bff68706c9048456b9de6d2520f55513184e3a.json](<../../provenance/context-382b5ee84e17093ab3736ad910bff68706c9048456b9de6d2520f55513184e3a.json>)
- [训练有效配置 config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json](<../../provenance/config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 0.7051561355590821, "max_tilt": 0.10502362996339798, "termination_rate": 0.0, "tracking_xy_rmse": 0.055324405897655984, "tracking_yaw_rmse": 0.05279093582591673}`
- `stand-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 999, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `forward-stop-a01` 指标：`{"max_joint_velocity_utilization": 0.8499949455261231, "max_tilt": 0.20048734545707703, "termination_rate": 0.0, "tracking_xy_rmse": 0.13899547255489725, "tracking_yaw_rmse": 0.10808067555500263}`
- `forward-stop-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `backward-stop-a01` 指标：`{"max_joint_velocity_utilization": 0.8620853424072266, "max_tilt": 0.17043747007846832, "termination_rate": 0.0, "tracking_xy_rmse": 0.17988112711989462, "tracking_yaw_rmse": 0.1001308630478157}`
- `backward-stop-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 0.9658340454101563, "max_tilt": 0.21359477937221527, "termination_rate": 0.0, "tracking_xy_rmse": 0.08085014610398314, "tracking_yaw_rmse": 0.19237814659298008}`
- `turn-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.8831339836120605, "max_tilt": 0.14131884276866913, "termination_rate": 0.0, "tracking_xy_rmse": 0.08447078528810596, "tracking_yaw_rmse": 0.22185146309793413}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 0.980260181427002, "max_tilt": 0.27148768305778503, "termination_rate": 0.0, "tracking_xy_rmse": 0.14735341351149123, "tracking_yaw_rmse": 0.29786919320572086}`
- `moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。
- `moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.040437602996826, "max_tilt": 0.23030702769756317, "termination_rate": 0.0, "tracking_xy_rmse": 0.1436742459412322, "tracking_yaw_rmse": 0.2840038647909703}`
- `moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

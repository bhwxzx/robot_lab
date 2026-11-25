# 评估批次 flat-leg-roa-server-49999-20261125-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0`；训练：`2026-11-24_14-51-42`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-a01 | completed | stand，1000 / 1 / 42 | complete | [结果](<raw/stand-a01/result.json>) / [日志](<raw/stand-a01/console.log>) |
| forward-a01 | completed | forward，1000 / 1 / 42 | complete | [结果](<raw/forward-a01/result.json>) / [日志](<raw/forward-a01/console.log>) |
| backward-a01 | completed | backward，1000 / 1 / 42 | complete | [结果](<raw/backward-a01/result.json>) / [日志](<raw/backward-a01/console.log>) |
| turn-left-a01 | completed | turn-left，1000 / 1 / 42 | complete | [结果](<raw/turn-left-a01/result.json>) / [日志](<raw/turn-left-a01/console.log>) |
| turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/turn-right-a01/result.json>) / [日志](<raw/turn-right-a01/console.log>) |
| forward-left-a01 | completed | forward-left，1000 / 1 / 42 | complete | [结果](<raw/forward-left-a01/result.json>) / [日志](<raw/forward-left-a01/console.log>) |
| forward-right-a01 | completed | forward-right，1000 / 1 / 42 | complete | [结果](<raw/forward-right-a01/result.json>) / [日志](<raw/forward-right-a01/console.log>) |
| forward-030-a01 | completed | forward-030，1000 / 1 / 42 | complete | [结果](<raw/forward-030-a01/result.json>) / [日志](<raw/forward-030-a01/console.log>) |
| turn-left-extended-a01 | completed | turn-left-extended，1000 / 1 / 42 | complete | [结果](<raw/turn-left-extended-a01/result.json>) / [日志](<raw/turn-left-extended-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`dd24cbd6c1baf05e56645535f7b8ac4c2e0fffb8b3d5d3797383faaf34b61650`；runner：`OnPolicyRunnerROA`。
- [场景来源 scenario-262996e2cbc4c666a82e3e7f65c773ce1aa806089dbd39a920a0d0b9f776547e.json](<../../provenance/scenario-262996e2cbc4c666a82e3e7f65c773ce1aa806089dbd39a920a0d0b9f776547e.json>)
- [场景来源 scenario-27e8a47aacfa034914bfb354281da572f9df734eef5801c26e1de698d37bc400.json](<../../provenance/scenario-27e8a47aacfa034914bfb354281da572f9df734eef5801c26e1de698d37bc400.json>)
- [场景来源 scenario-3ab7f333f58574faf7a6f4e1c457348aed9c2f12e500fb0f32453bfab47368ad.json](<../../provenance/scenario-3ab7f333f58574faf7a6f4e1c457348aed9c2f12e500fb0f32453bfab47368ad.json>)
- [场景来源 scenario-4130e782f32c19b427419c2d12e1b047172f3d4122330aee307f711fc14e55c1.json](<../../provenance/scenario-4130e782f32c19b427419c2d12e1b047172f3d4122330aee307f711fc14e55c1.json>)
- [场景来源 scenario-63b55e869fc043c8dc58d10981760731069a706de43b9d5f66ddfb5c1ba66baf.json](<../../provenance/scenario-63b55e869fc043c8dc58d10981760731069a706de43b9d5f66ddfb5c1ba66baf.json>)
- [场景来源 scenario-b4d0f3da42295967fe15ccef417eeab5ab8aff86e15bdb614a8991796d542b14.json](<../../provenance/scenario-b4d0f3da42295967fe15ccef417eeab5ab8aff86e15bdb614a8991796d542b14.json>)
- [场景来源 scenario-c3aa88d44fcb56e284fa5b58697268d348ae760f7d87d56a080ce498f32a9993.json](<../../provenance/scenario-c3aa88d44fcb56e284fa5b58697268d348ae760f7d87d56a080ce498f32a9993.json>)
- [场景来源 scenario-ce0948b0984dfa3f19e57d554123bdb590534e61248c17869b21a1e917c740d8.json](<../../provenance/scenario-ce0948b0984dfa3f19e57d554123bdb590534e61248c17869b21a1e917c740d8.json>)
- [场景来源 scenario-f56d20bfaf7efda50650f0b1fe58d5906485e7f50e39434d8f22322863f5af83.json](<../../provenance/scenario-f56d20bfaf7efda50650f0b1fe58d5906485e7f50e39434d8f22322863f5af83.json>)
- [训练上下文 context-46f1813cdaad954a30cec57e0d1e3379e01013506ca3f2a3d3ba9395819cf52d.json](<../../provenance/context-46f1813cdaad954a30cec57e0d1e3379e01013506ca3f2a3d3ba9395819cf52d.json>)
- [训练有效配置 config-4bae6b94de31c2d9886f91e98b1469651f9909fbfe916912c30e3dd34c6fbc40.json](<../../provenance/config-4bae6b94de31c2d9886f91e98b1469651f9909fbfe916912c30e3dd34c6fbc40.json>)

- `stand-a01` 指标：`{"max_joint_velocity_utilization": 0.8747810363769531, "max_tilt": 0.10511477291584015, "termination_rate": 0.0, "tracking_xy_rmse": 0.06053503933917121, "tracking_yaw_rmse": 0.06726968079535299}`
- `stand-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 999, "start_step": 0}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-a01` 指标：`{"max_joint_velocity_utilization": 1.1073890686035157, "max_tilt": 0.23785176873207092, "termination_rate": 0.0, "tracking_xy_rmse": 0.18369420648308973, "tracking_yaw_rmse": 0.12048239663787635}`
- `forward-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `backward-a01` 指标：`{"max_joint_velocity_utilization": 1.017801284790039, "max_tilt": 0.24352295696735382, "termination_rate": 0.0, "tracking_xy_rmse": 0.19915169074321273, "tracking_yaw_rmse": 0.1576554907068147}`
- `backward-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [-0.4, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-left-a01` 指标：`{"max_joint_velocity_utilization": 0.9946378707885742, "max_tilt": 0.2712899446487427, "termination_rate": 0.0, "tracking_xy_rmse": 0.1733348683795981, "tracking_yaw_rmse": 0.3735797755438007}`
- `turn-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9997956275939941, "max_tilt": 0.30527463555336, "termination_rate": 0.0, "tracking_xy_rmse": 0.21633065373920185, "tracking_yaw_rmse": 0.389851014682669}`
- `turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-left-a01` 指标：`{"max_joint_velocity_utilization": 1.2038436889648438, "max_tilt": 0.3204881250858307, "termination_rate": 0.0, "tracking_xy_rmse": 0.1825583358413968, "tracking_yaw_rmse": 0.38957311420431706}`
- `forward-left-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-right-a01` 指标：`{"max_joint_velocity_utilization": 1.3230411529541015, "max_tilt": 0.2960810959339142, "termination_rate": 0.0, "tracking_xy_rmse": 0.1973771630098542, "tracking_yaw_rmse": 0.3872877278446326}`
- `forward-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `forward-030-a01` 指标：`{"max_joint_velocity_utilization": 1.2024067878723144, "max_tilt": 0.2690912187099457, "termination_rate": 0.0, "tracking_xy_rmse": 0.19520887449755078, "tracking_yaw_rmse": 0.13301690090712603}`
- `forward-030-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.3, 0, 0], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_foot_ground": true, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `turn-left-extended-a01` 指标：`{"max_joint_velocity_utilization": 1.0112947463989257, "max_tilt": 0.2712899446487427, "termination_rate": 0.0, "tracking_xy_rmse": 0.18656926522231596, "tracking_yaw_rmse": 0.40757417669679386}`
- `turn-left-extended-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, 0.5], "end_step": 849, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 850}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "evaluation.roa_mode": "student", "events.randomize_push_robot": null, "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

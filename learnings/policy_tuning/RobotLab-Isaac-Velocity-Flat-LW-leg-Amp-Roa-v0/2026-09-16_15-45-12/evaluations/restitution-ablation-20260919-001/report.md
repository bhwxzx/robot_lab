# 评估批次 restitution-ablation-20260919-001

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-16_15-45-12`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| r050-turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/r050-turn-right-a01/result.json>) / [日志](<raw/r050-turn-right-a01/console.log>) |
| r050-moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/r050-moving-turn-a01/result.json>) / [日志](<raw/r050-moving-turn-a01/console.log>) |
| r050-moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/r050-moving-turn-right-a01/result.json>) / [日志](<raw/r050-moving-turn-right-a01/console.log>) |
| r010-turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/r010-turn-right-a01/result.json>) / [日志](<raw/r010-turn-right-a01/console.log>) |
| r010-moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/r010-moving-turn-a01/result.json>) / [日志](<raw/r010-moving-turn-a01/console.log>) |
| r010-moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/r010-moving-turn-right-a01/result.json>) / [日志](<raw/r010-moving-turn-right-a01/console.log>) |
| r000-turn-right-a01 | completed | turn-right，1000 / 1 / 42 | complete | [结果](<raw/r000-turn-right-a01/result.json>) / [日志](<raw/r000-turn-right-a01/console.log>) |
| r000-moving-turn-a01 | completed | moving-turn，1000 / 1 / 42 | complete | [结果](<raw/r000-moving-turn-a01/result.json>) / [日志](<raw/r000-moving-turn-a01/console.log>) |
| r000-moving-turn-right-a01 | completed | moving-turn-right，1000 / 1 / 42 | complete | [结果](<raw/r000-moving-turn-right-a01/result.json>) / [日志](<raw/r000-moving-turn-right-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`f8956bb1f9b43bdc638a52dbaf0a3301b022a4f967b581f2ed40663eb797d16a`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-12c610400c42638ae7e9594a522fe442cf71cda463e9d99461d52c48e9661b04.json](<../../provenance/scenario-12c610400c42638ae7e9594a522fe442cf71cda463e9d99461d52c48e9661b04.json>)
- [场景来源 scenario-6b344cccc90f6405d7c1554adcd06798e2be28f46f59da172612052c09508e76.json](<../../provenance/scenario-6b344cccc90f6405d7c1554adcd06798e2be28f46f59da172612052c09508e76.json>)
- [场景来源 scenario-6b963df21422f50b7927d87c708a20c0009c3e9dd464cbd762a28212fbba5ac0.json](<../../provenance/scenario-6b963df21422f50b7927d87c708a20c0009c3e9dd464cbd762a28212fbba5ac0.json>)
- [场景来源 scenario-7a44d2f0a048d6b68470400254293bbc0f486a83cf7f2847887ea3b77841f5f9.json](<../../provenance/scenario-7a44d2f0a048d6b68470400254293bbc0f486a83cf7f2847887ea3b77841f5f9.json>)
- [场景来源 scenario-7a5d0031b0d1f112268fb1288f193ebc679925493a319fb416015127d598ebcf.json](<../../provenance/scenario-7a5d0031b0d1f112268fb1288f193ebc679925493a319fb416015127d598ebcf.json>)
- [场景来源 scenario-9312f68d3dddf309be2e466a8b84f03c53897bdf175c237942e6ae09db75c313.json](<../../provenance/scenario-9312f68d3dddf309be2e466a8b84f03c53897bdf175c237942e6ae09db75c313.json>)
- [场景来源 scenario-efa056f81b849891212bc9c667132818796ea5de188480a297bb7a392b0c4dd1.json](<../../provenance/scenario-efa056f81b849891212bc9c667132818796ea5de188480a297bb7a392b0c4dd1.json>)
- [场景来源 scenario-f29a76583fd74785a459d44ca6d78bb23eba1ff4f9a858d42596d9e273addf14.json](<../../provenance/scenario-f29a76583fd74785a459d44ca6d78bb23eba1ff4f9a858d42596d9e273addf14.json>)
- [场景来源 scenario-f40431961c38274ba12c618b83c730e775fb211dfdcc10f0104dd947c01ba4e4.json](<../../provenance/scenario-f40431961c38274ba12c618b83c730e775fb211dfdcc10f0104dd947c01ba4e4.json>)
- [训练上下文 context-8cf8f15f947d54068f7288642447bb4246948d3a2ca91dac15d47ec25f5645ac.json](<../../provenance/context-8cf8f15f947d54068f7288642447bb4246948d3a2ca91dac15d47ec25f5645ac.json>)
- [训练有效配置 config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json](<../../provenance/config-636927f3c2285b54bfb199c38b5fecb129ae5862a1aec5c24e96166adcc15c9c.json>)

- `r050-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9991875648498535, "max_tilt": 0.16323137283325195, "termination_rate": 0.0, "tracking_xy_rmse": 0.07330808600269538, "tracking_yaw_rmse": 0.31113644714044714}`
- `r050-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.5], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r050-moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.1428607940673827, "max_tilt": 0.22102390229701996, "termination_rate": 0.0, "tracking_xy_rmse": 0.16845860880395996, "tracking_yaw_rmse": 0.3533195764190554}`
- `r050-moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.5], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r050-moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.5107056617736816, "max_tilt": 0.1936391443014145, "termination_rate": 0.0, "tracking_xy_rmse": 0.14564678165816497, "tracking_yaw_rmse": 0.34032158842736016}`
- `r050-moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.5], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r010-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0053484916687012, "max_tilt": 0.13953936100006104, "termination_rate": 0.0, "tracking_xy_rmse": 0.08012538323065341, "tracking_yaw_rmse": 0.2942197589791842}`
- `r010-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.1], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r010-moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.0845368385314942, "max_tilt": 0.21254663169384003, "termination_rate": 0.0, "tracking_xy_rmse": 0.17504251474354915, "tracking_yaw_rmse": 0.345739985380839}`
- `r010-moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.1], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r010-moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0053484916687012, "max_tilt": 0.16805315017700195, "termination_rate": 0.0, "tracking_xy_rmse": 0.1454697755321891, "tracking_yaw_rmse": 0.3365928040146954}`
- `r010-moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.1], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r000-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 0.9990490913391114, "max_tilt": 0.1630970686674118, "termination_rate": 0.0, "tracking_xy_rmse": 0.07891643059767585, "tracking_yaw_rmse": 0.28637273064028135}`
- `r000-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r000-moving-turn-a01` 指标：`{"max_joint_velocity_utilization": 1.0833328247070313, "max_tilt": 0.3425676226615906, "termination_rate": 0.0, "tracking_xy_rmse": 0.1795289103546894, "tracking_yaw_rmse": 0.3753127120488349}`
- `r000-moving-turn-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, 0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。
- `r000-moving-turn-right-a01` 指标：`{"max_joint_velocity_utilization": 1.0015192985534669, "max_tilt": 0.18584758043289185, "termination_rate": 0.0, "tracking_xy_rmse": 0.14642216246565032, "tracking_yaw_rmse": 0.3460227614819285}`
- `r000-moving-turn-right-a01` 命令调度：`[{"command": [0, 0, 0], "end_step": 99, "start_step": 0}, {"command": [0.4, 0, -0.5], "end_step": 749, "start_step": 100}, {"command": [0, 0, 0], "end_step": 999, "start_step": 750}]`；训练配置覆盖：`{"curriculum.terrain_levels": null, "episode_length_s": 65.0, "events.randomize_push_robot": null, "events.randomize_rigid_body_material.params.restitution_range": [0.0, 0.0], "scene.terrain.terrain_generator": null, "scene.terrain.terrain_type": "plane"}`。

## 观察与限制

- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据


## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

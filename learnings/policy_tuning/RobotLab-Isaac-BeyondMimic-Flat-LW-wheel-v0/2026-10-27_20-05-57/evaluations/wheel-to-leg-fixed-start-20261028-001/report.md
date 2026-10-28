# 评估批次 wheel-to-leg-fixed-start-20261028-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-10-27_20-05-57`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`ea5d1de64ee3a8a10c6456d92fdccfe5f0bbe58ec301b5cf7015555fa6b2f687`；runner：`OnPolicyRunner`。
- [场景来源 scenario-44e40da1d1cb6c4ba7cd0ecfb76c6f3a31a508e515d915b6c2724256101dd345.json](<../../provenance/scenario-44e40da1d1cb6c4ba7cd0ecfb76c6f3a31a508e515d915b6c2724256101dd345.json>)
- [场景来源 scenario-e21adffa3db48b2d5a0e38b6aad87a1711a13a06ae851c65a9abd8fa13b499e6.json](<../../provenance/scenario-e21adffa3db48b2d5a0e38b6aad87a1711a13a06ae851c65a9abd8fa13b499e6.json>)
- [训练上下文 context-be95f56e871c647facf81a8c723b58f904941f8481339431c185b534461f980f.json](<../../provenance/context-be95f56e871c647facf81a8c723b58f904941f8481339431c185b534461f980f.json>)
- [训练有效配置 config-c5ad2f171f7a2e6b22f2b564a1efde7958601489baf04d89df18b53ba92a3bba.json](<../../provenance/config-c5ad2f171f7a2e6b22f2b564a1efde7958601489baf04d89df18b53ba92a3bba.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_takeoff_anchor_ori.weight": 0, "rewards.motion_takeoff_pitch_ang_vel.weight": 0, "rewards.wheel_contact_continuous.weight": 1}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.motion_takeoff_anchor_ori.weight": 0, "rewards.motion_takeoff_pitch_ang_vel.weight": 0, "rewards.wheel_contact_continuous.weight": 1}`。

## 观察与限制

- 本机younghit场次2026-10-27_20-05-57/model_49999.pt；与服务器20-02-47是不同训练，不能混用结论。
- 两组均固定frame0、seed42、1环境、1000控制步、4000物理步预算、Native无视频。名义组关闭训练随机化，随机化组保留训练随机化和0..3物理步延迟。
- 为匹配保存训练YAML，仅在评估内存中恢复takeoff_pitch_ang_vel=0、takeoff_anchor_ori=0、wheel_contact_continuous=1。没有修改源码或持久配置。
- 实际argv通过既有execute接口追加--kit_args=--/app/settings/persistent=false，完整命令见每组console.log。
- 训练日志最后100轮motion_finished均值99.4816%来自自适应随机起始相位训练，不是固定frame0完整动作成功率。
- 单seed、每组一个startup物性抽样。无批准收敛或硬件验收标准；驱动力矩不含全部接触被动载荷，八连杆失载不覆盖base/hip。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [training-origin-20261028-001.json](<../../evidence/training/training-origin-20261028-001.json>)
- [training-windows-20261028-001.json](<../../evidence/training/training-windows-20261028-001.json>)
- [protection-before-20261028-001.json](<../../evidence/training/protection-before-20261028-001.json>)

## 建议与待授权事项

- 按批次报告检查失败项和动作表现；复测使用新的 batch_id 与 attempt_id。

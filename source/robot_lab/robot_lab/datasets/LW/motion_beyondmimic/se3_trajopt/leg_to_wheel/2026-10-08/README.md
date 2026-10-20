# LW 腿式→轮式规划轨迹（2026-10-08）

`leg_to_wheel_transform_60hz.csv` 逐字节复制自 `se3_trajopt` 的单次运行 `trajopt_solutions_batch/leg_to_wheel_test/current_weights_rerun_20261008_210153/leg_to_wheel_transform_60hz.csv`。求解于 2026-10-08T21:10:13.462427+08:00 结束。本目录与已有日期版本并列，供区分不同规划结果。

- CSV SHA-256：`ef3ced90bda48a949c21e20cdc4e35cd105198746773471c424af171c1d7f4ad`；来源 `run.json` SHA-256：`967bb9774baf65db4e41eec6185636f600b27953500d4d405bf82856ece42c1a`。
- 本目录的 `audit.json` 逐字节复制自该运行，SHA-256：`12fb1b7df9412e784a2d11fa599c21fc56d8866e387c27cf7c26c91d04a7080e`。测试脚本快照 SHA-256：`30c75a7e1841ed6abf9f36e74e35667dbaa659a8875e32378b03e38b2b70573f`；URDF 快照 SHA-256：`1067866957241f3f87c04cd231c7d3638213eb26844672572aa463ad0605f3f1`。
- 原始完整求解解保留在本地源运行目录的 `trajopt_solutions_batch/leg_to_wheel/leg_to_wheel_08102026_211013.json`，SHA-256：`5304d308191f50397b457a227b757d63f0febce7e81f93725177710b9ed0984e`。其中含全部节点的 q/v/a 和接触力；本目录 CSV 只含位姿。
- CSV 无表头，204 行 × 17 列，60 Hz；203 个间隔共 3.383333333 秒。保留首末帧，直接导出全部求解节点，**未插值**；相对保存节点的最大十进制舍入误差 `4.997765986080912e-07`。列顺序：`x, y, z, qx, qy, qz, qw, right_hip_joint, left_hip_joint, right_thigh_joint, left_thigh_joint, right_shank_joint, left_shank_joint, right_foot_joint, left_foot_joint, right_wheel_joint, left_wheel_joint`。
- 位置采用世界系米，四元数顺序 xyzw，关节角为弧度；导入时按上述关节名称映射。
- 有效配置：`DT=1/60 s`，足端支撑/腾空/轮式阶段时长 `1.30/0.30/1.80 s`，`wheel_contact=False`；IPOPT `max_iter=500`、`tol=1e-3`、墙钟上限 600 秒。腾空为 18 个保存节点，从 `1.30 s` 到落地首节点 `1.60 s`，规划腾空时长 0.30 秒。
- 在保存节点上，右/左小腿关节速度的全程峰值分别为 `9.5301997044/9.5252732524 rad/s`，均发生在落地后一帧 `1.616667 s`；模型 URDF 速度上限为 `20 rad/s`。规划落地总竖向接触力峰值为 **687.054880 N**。这只是优化器计划载荷，不是实测碰撞冲击。
- IPOPT 因 500 次迭代上限退出，状态 `-1`，**未正式收敛**；最终对偶不可行度为 `0.0073272688929464135`。独立重建的最大原始节点约束违反量为 `1.0499326492663386e-06`，位于第 76 节点的左小腿力矩上限，略高于既有 `1e-6` 验收线。落地速度下降不代表落地冲击也下降。
- CSV 读取、按关节名称映射及全部 204 帧正运动学检查通过。未完成节点间连续动力学、碰撞、RL 跟踪或实机验收，不能视为已验证的部署参考轨迹。

# LW 腿式→轮式规划轨迹（2026-09-29）

`leg_to_wheel_transform_60hz.csv` 是 `se3_trajopt` 测试运行 `current_parameters_rerun_20260929_202110` 的原版导出，逐字节复制自 `trajopt_solutions_batch/leg_to_wheel_test/current_parameters_rerun_20260929_202110/leg_to_wheel_transform_60hz.csv`。运行结束于 2026-09-29T20:29:45.979226+08:00。

- CSV SHA-256：`33475c628ffc96d9c3c742bc46495eb9f45951444221ffeccd6280fc80351bf9`；来源 `run.json` SHA-256：`6e65a1b7e25d26b42d3b5ea52a11d2b03c73915341639501619ebf8291ba7793`。
- 无表头，171 行 × 17 列，60 Hz；170 个间隔共 2.833333333 秒。保留首末帧，直接导出全部求解节点，**未插值**。
- 列顺序：`x, y, z, qx, qy, qz, qw, right_hip_joint, left_hip_joint, right_thigh_joint, left_thigh_joint, right_shank_joint, left_shank_joint, right_foot_joint, left_foot_joint, right_wheel_joint, left_wheel_joint`。位置为世界系米，四元数顺序为 xyzw，关节角为弧度。
- 保存节点上的计划落地总竖向力峰值为 **480.316644 N**。这是优化结果中的计划载荷，不是实测碰撞峰值。
- IPOPT 达到 500 次迭代上限（状态 -1），未正式收敛；独立离散约束最大残差为 `2.128765942188693e-06`，高于原运行记录的 `1e-6` 验收项。
- 原运行记录确认 CSV 与保存节点一致、关节位置及保存节点速度限位检查通过；未完成连续动力学、碰撞、RL 或实机验收。CSV 只有位置，不含速度、加速度和接触力，不能视为已验证的部署参考轨迹。

# LW 轮式→腿式规划轨迹（2026-10-23）

用户于 2026-10-23T16:39:12.873389+08:00 将 `current_script_rerun_20261023_161936` 指定为新的比较基线，取代 2026-10-23 16:00 版。本目录新增归档，历史版本保留。

**状态：IPOPT=-5，达到墙钟上限，未收敛；最大原始违反量 8.8552074089420785e-05，严格 1e-6 节点验收失败。该记录是用户指定的比较基线，不表示通过严格可行性验收。**

- 来源：`/home/lfr/se3_trajopt/trajopt_solutions_batch/wheel_to_leg_test/current_script_rerun_20261023_161936`。435 次迭代，求解耗时 600.302031040 秒，`EXIT: Maximum wallclock time exceeded.`；变量界最大违反量 0。
- `wheel_to_leg_transform_60hz.csv` 逐字节复制自源运行，无表头、204 行 × 17 列、原始 60 Hz；保留首末帧，直接导出全部原始求解节点，**无插值**。203 个时间间隔共 3.383333333 秒。
- CSV SHA-256：`791632a4bbd9d19da77fed85b0d291e65a992f941cf7a8484f15e2bb9519df1c`。
- `audit.json` 逐字节复制自源运行，SHA-256：`f874943200c973861c220f942691c5dbca64ef19b32900be211fd517663d63e0`。其中 `baseline_run` 是求解时的旧比较基线（16:00），当前指定基线是本轮（16:19）；保留原审计与 run.json 的历史语义。
- 源 run.json SHA-256：`a6cb4845a51f1b94d441787c667ed47e8c7fb635f63b37bf7562c1641237c731`；执行脚本快照 SHA-256：`7ab9bcd7dee86dafc970d0a6594bbc71b2c6df124d9cc9015ed2689b7e768449`；LW_Wheel URDF 快照 SHA-256：`88d82a1ac19bfd91c7eeee05080ad97688eabb36a35700e7ea043d698847b512`。
- 完整原始解保存在本地 `/home/lfr/se3_trajopt/trajopt_solutions_batch/wheel_to_leg_test/current_script_rerun_20261023_161936/trajopt_solutions_batch/wheel_to_leg/wheel_to_leg_23102026_163033.json`，SHA-256：`62ecdc5f3121d9876e96318e1a0f32036a338d71aeb54f366f17f13aad059fdd`。该 JSON 含全部节点 q/v/a 与接触力；归档 CSV 仅含位姿。配置、求解向量、残差、依赖快照和 REPORT.md 保留在本地源运行目录。
- 列顺序：`x, y, z, qx, qy, qz, qw, right_hip_joint, left_hip_joint, right_thigh_joint, left_thigh_joint, right_shank_joint, left_shank_joint, right_foot_joint, left_foot_joint, right_wheel_joint, left_wheel_joint`。位置为世界系米，四元数 xyzw，关节角为弧度；导入按关节名称映射。
- 轮支撑/腾空/足支撑为 1.30/0.30/1.80 秒，节点数 78/18/108，DT=1/60 秒；触地节点 96、t=1.6 秒。末 0.5 秒（节点 173—203）全部 16 维速度为零。
- 轮支撑 max_delta_force=400，足支撑=100；准备阶段世界竖直力逐点权重 2.5e-4，落地等向力逐点权重 2e-3。准备关节固定初态参考，腾空固定终态参考。
- 准备大腿/小腿姿态权重 3/3、腾空 0.1/3、落地 1/1；全关节速度/加速度权重 0.1/5e-4。BasePositionCost Z 权重准备/腾空/落地为 10/0/30，该代价采用 SE(3) 对数坐标；下述高度均按世界系基座位置计算。
- 最大违反项：节点 76、t=1.266666667 秒、左轮 c_friction_ids 分量 5（接触力模长上限），原始违反量 8.8552074089420785e-05。严格门槛保持 1e-6。
- 右/左小腿峰速 9.590566124/9.585746251 rad/s，峰加速度 264.054484185/263.953343248 rad/s²，峰力矩 87.569629575/87.523334010 Nm。节点关节速度、力矩无超限。
- 双轮下蹲最低基座高度 0.489992133540 m；触地 0.563547438341 m，落地最低 0.472799294858 m；计划双足世界 Fz 峰值 909.911614471 N。
- 归档时重新核验全部节点六位小数导出逐字节一致、CSV 形状、60 Hz、末段零速度及原始残差。全部 204 帧正运动学相对原始解最大位置误差 1.64522379e-06 m，通过 1e-5 m 检查。CSV 未重采样或修改。
- 以上为离散节点与导出检查；计划节点力不等同真实碰撞冲击，未作节点间连续动力学、闭环跟踪或实机验收。

# LW 双足 AMP 专家库

版本 `lw_grid153_125hz_tracking_001`，来自 OCS2 / Pinocchio 参考轨迹生成器。训练入口为 `motion_amp_expert/motion_*.txt`。本目录包含 152 条完整的起步、巡航、制动、停稳轨迹和 1 条静站轨迹，共 109,353 帧。

TXT 文件使用 JSON 格式。头部每个字段单独一行，`Frames` 中每行对应一个 0.02 s 采样时刻，22 个数值按原始精度保存。

| 项目 | 数据 |
| --- | --- |
| 速度网格 | `vx` 从 −0.8 到 +0.8 m/s，间隔 0.1 m/s。`vy = 0`。`wz` 取 0、±0.10、±0.15、±0.25、±0.35 rad/s，全部组合共 153 点 |
| 步态与采样 | 同一只脚每 0.80 s 落地一次，步态周期频率为 1.25 Hz。文件帧间隔为 0.02 s（50 Hz） |
| 每帧 22 维 | 8 个相对标称关节角、8 个相对标称关节速度、机身坐标系下的左足和右足连杆原点各 3 维。关节按髋、股、胫、踝排列，每类均为先右后左 |
| 时长与边界 | 运动轨迹 11.2–16.8 s，使用 `Clamp`。静站轨迹 0.8 s，共 41 帧，使用 `Wrap` |
| 指令与权重 | 每个文件保留 `Command` 和 `MotionWeight`。运动文件还保留 `WalkStart`、`CruiseStart`、`BrakeStart`、`WalkStop`，单位为秒 |

普通 AMP 加载器按 `MotionWeight` 选取文件。实现按速度指令选取邻近专家时，可用 `Command` 建立速度网格索引。

## 在训练机使用

以下示例在 `robot_lab` 根目录运行，假定 `LW_datasets` 与其同级。第一次获取时运行 `git clone https://github.com/bhwxzx/LW_datasets.git ../LW_datasets`。已有仓库可以更新并校验归档。

```bash
git -C ../LW_datasets pull --ff-only origin master
amp_release="$(realpath ../LW_datasets/amp/LW_Leg/lw_grid153_125hz_tracking_001)"
(cd "$amp_release" && sha256sum --check --quiet SHA256SUMS)
```

在装有 `robot_lab` 的 Isaac Lab 训练环境运行下列普通 AMP 示例。命令显式选择本版专家，把策略与价值网络的时钟设为 1.25 Hz，并将 AMP 的 10 帧历史、足端次序和训练指令范围与数据对应起来。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-v0 --headless \
  "agent.amp_motion_files=[$amp_release/motion_amp_expert/motion_*.txt]" \
  agent.amp_history_length=10 \
  env.observations.amp.history_length=10 \
  env.observations.amp.flatten_history_dim=False \
  'env.observations.amp.end_effort_pos.params.asset_cfg.body_names=[left_foot_link,right_foot_link]' \
  env.observations.amp.end_effort_pos.params.asset_cfg.preserve_order=True \
  env.observations.policy.gait_phase.params.gait_freq=1.25 \
  env.observations.critic.gait_phase.params.gait_freq=1.25 \
  'env.commands.base_velocity.ranges.lin_vel_x=[-0.8,0.8]' \
  'env.commands.base_velocity.ranges.lin_vel_y=[0.0,0.0]' \
  'env.commands.base_velocity.ranges.ang_vel_z=[-0.35,0.35]'
```

## 数据核验

- [速度点与采样权重](amp_walk_125hz.json) 给出每个指令对应的参数组与权重。152 条运动轨迹均通过完整参考轨迹验收，零指令通过静止性检查。[整库验收](release_verification.json)
- 导出轨迹的巡航平均纵向速度最大绝对误差为 0.00995 m/s，偏航角速度最大绝对误差为 0.00957 rad/s。[逐点速度记录](tracking_by_command.csv) · [高度、步宽与步频核验](height_verification.json)
- 训练加载器以 0.02 s 间隔构造 10 帧、220 维专家观测。[生成时的加载器核验](training_loader_verification.json) · [归档文件 SHA-256 清单](SHA256SUMS)

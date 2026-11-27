# LW 双足 AMP 专家库 · 巡航速度校准版

`lw_grid153_125hz_tracking_002_heading_velocity` 包含 153 档指令。按最终导出的 50 Hz 专家帧测速，每档的纵向平均速度误差均小于 0.01 m/s，偏航平均角速度误差均小于 0.01 rad/s。相对上一版，32 档由 OCS2 重新求解，其余 121 档专家文件逐字节一致。[逐指令测速表](amp_tracking_by_command.csv) · [验收摘要](release_verification.json) · [上一版](../lw_grid153_125hz_tracking_001_heading_velocity/)

## 速度跟踪

运动轨迹取 `[CruiseStart, BrakeStart)`，对区间内的每个导出帧等权求均值，再减去文件头的 `Command`。纵向取 `Frames` 第 0 列，偏航取第 5 列。静站另取全片 41 帧。严格验收用两项误差的绝对值，等于阈值也算超限。`vy` 取第 1 列，保留在表中供审阅。

| 巡航逐帧均值指标 | 上一版最大绝对误差 | 本版最大绝对误差 | 本版最大值指令 `(vx, wz)` | 达到或超过 0.01 的档数，上一版 → 本版 |
| --- | ---: | ---: | --- | ---: |
| 纵向速度 | 0.02362477 m/s | 0.00974501 m/s | `(+0.3, +0.25)` | 25 → 0 |
| 偏航角速度 | 0.02086439 rad/s | 0.00938150 rad/s | `(-0.2, -0.15)` | 11 → 0 |

152 条运动轨迹合计 41,280 个巡航帧。`first_frame` 和 `end_frame_exclusive` 可直接用作切片端点。`frame_mean` 为逐帧均值，`signed_error` 和 `abs_error` 分别为测量均值减指令的有符号误差及其绝对值。[逐指令测速表](amp_tracking_by_command.csv)

测速表也单列连续十帧的统计。每个窗口步进一帧，全部十帧位于同一测量区间。运动轨迹共 39,912 个窗口，静站另有 32 个。`window10_mean` 为各窗口均值的平均值，`window10_p10` 和 `window10_p90` 为窗口均值的线性分位数。按十帧窗口均值单独统计，纵向有 6 档、偏航有 0 档绝对误差大于 0.01。阈值验收采用上表的逐帧均值。[测速脚本](amp_tracking_by_command.py)

本次重求解逐档调整规划前馈或 MPC 参数。速度指令、起停时刻、步频、机身高度目标及 `MotionWeight` 与上一版逐档对应。两张调参表分别记录本次实际导出帧的优化前后误差和历史生成器的调参结果。[本版 32 档调参记录](frame_tracking_refinement.csv) · [历史生成器调参记录](tracking_by_command.csv)

在 `vx=−0.8 m/s、wz=±0.35 rad/s` 两档，OCS2 巡航反馈对支撑脚趾、脚跟绕脚掌中心转动时的切向速度增加了软惩罚，用来控制换脚前的足滑。两档都按同一套全程步宽、足滑、关节速度和估计力矩限值验收。对其中的正向偏航档和另一条倒走轨迹独立重求解，均复现了原始 200 Hz 状态和两种导出帧。[验收摘要](release_verification.json)

## 步态和数据格式

`vx` 取 −0.8 至 +0.8 m/s，间隔 0.1 m/s。`wz` 取 0、±0.10、±0.15、±0.25、±0.35 rad/s。17 × 9 档一一配对，目标 `vy=0`。152 个非零指令各有一条起步、巡航、制动和停稳的完整 `Clamp` 轨迹，零指令为 0.80 s 的 `Wrap` 静站。运动轨迹同脚步态周期为 0.80 s（1.25 Hz）。[指令及采样权重](amp_walk_125hz.json)

每条轨迹均通过原有参考验收，包括接触时序、支撑足滑动、关节速度和估计力矩。针对全部 109,353 个训练帧，以原始广义坐标和广义速度用 Pinocchio 独立重建基座速度，并通过 MuJoCo 正运动学核对回放与专家帧。200 Hz 求解状态的全库平均步宽为 0.41889 m，最小步宽为 0.38006 m。巡航摆步峰值为 0.09968 至 0.09989 m，起停摆步最小峰值为 0.05627 m。[验收摘要](release_verification.json) · [步宽、抬脚和机身高度](height_verification.json)

训练文件是 `motion_amp_expert/motion_*.txt`，采用 JSON，每行一个完整的 28 维专家帧。下表使用从零开始的 Python 切片。

| 切片 | 内容 | 坐标和单位 |
| --- | --- | --- |
| `[0:3]` | 基座线速度 `vx, vy, vz` | `base_link` 连杆原点，航向系，m/s |
| `[3:6]` | 机身物理角速度 `wx, wy, wz` | 航向系，rad/s |
| `[6:14]` | 8 个相对标称零位的关节角 | 髋、股、胫、踝，每种先右后左，rad |
| `[14:22]` | 8 个关节速度 | 与关节角同序，rad/s |
| `[22:25]`、`[25:28]` | 左、右足连杆原点相对基座的位置 | 含俯仰及横滚的完整机身系，m |

航向系 x 轴沿每帧机身前向在水平面上的投影，y 轴水平向左，z 轴与世界竖直向上对齐。线速度和角速度均用该帧的实际航向将世界系旋至航向系。因此 `wz` 是机身物理角速度在竖直轴上的投影。文件头记录 `RootVelocityFrame="heading"` 和 `RootVelocityPoint="base_link"`。

## 训练机拉取和复算

以下命令从 `robot_lab` 根目录执行，假定两个仓库是同级目录。

```bash
git -C ../LW_datasets pull --ff-only origin master
amp_release="$(realpath ../LW_datasets/amp/LW_Leg/lw_grid153_125hz_tracking_002_heading_velocity)"
(cd "$amp_release" && sha256sum --quiet --check SHA256SUMS)
python3 "$amp_release/amp_tracking_by_command.py" "$amp_release"
```

将训练配置 `agent.amp_motion_files` 指向 `$amp_release/motion_amp_expert/motion_*.txt`。在线 AMP 观测需与每个专家帧的 28 维顺序、速度取点和坐标系一致。按从旧到新拼接十帧得到 280 维，首尾相隔 0.18 s。

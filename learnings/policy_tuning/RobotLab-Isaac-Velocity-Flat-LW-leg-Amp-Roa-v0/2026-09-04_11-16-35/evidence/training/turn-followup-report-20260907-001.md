# LW_Leg Flat AMP-ROA 纯转向补测

**结论：补测复现了“行走中可以转向，纯转向基本停止”。先行走不能维持后续原地转向；低前后速度下也可能进入接近站立状态，并且左右方向存在差异。**

## 测试条件

- Native `model_50000.pt`，训练目录 `2026-09-04_11-16-35`，主机 younghit，seed 42，单环境。
- 共六组：左右方向各包含直接纯转向、行走后纯转向、前后速度逐步降低。共 10000 步 / 200 秒仿真。
- 每组先站立 10 秒；每个后续指令段保持 10 秒。wz 分别固定 +0.5 与 -0.5 rad/s。
- 指令序列：direct 为 vx=0；walking 为 vx=0.4→0；ramp 为 vx=0.4→0.2→0.1→0。
- 关闭随机推扰，评估回合上限统一设为65秒，保留其他训练环境随机化、观测噪声、物理与控制设置。所有六组都没有终止或重置。
- 六组开头10秒的完整遥测逐样本完全相同；walking 与 ramp 的前20秒条件相同，其共享阶段不是独立重复试验。
- 所有结果与视频/遥测哈希已通过 bundle 重验证。必需信号、可用性与可选姿态信号均完整。
- 检查点 SHA-256：`59ab5245acb17de8ea1b79e924c2f0189765be3c51c85f437b22b8e2423b0e19`。

## 稳态片段结果

每段取最后250个采样点，即最后约5秒。净朝向变化由世界系根姿态四元数（wxyz）计算并展开连续角度；端点相隔4.98秒。正号左转，负号右转。转向指令若被完整跟踪，这一窗口对应约±142.7°。

|方向|过程|vx 指令 (m/s)|平均实际 vx (m/s)|净朝向变化 (°)|
|---|---|---:|---:|---:|
|left|direct|0.0|-0.0007|+0.226|
|left|walking|0.4|0.3555|+8.360|
|left|walking|0.0|0.0026|+0.357|
|left|ramp|0.4|0.3555|+8.360|
|left|ramp|0.2|0.0008|+0.180|
|left|ramp|0.1|-0.0000|+0.280|
|left|ramp|0.0|-0.0041|+0.455|
|right|direct|0.0|0.0001|+0.037|
|right|walking|0.4|0.4282|-18.930|
|right|walking|0.0|-0.0004|+0.110|
|right|ramp|0.4|0.4282|-18.930|
|right|ramp|0.2|0.2249|-10.168|
|right|ramp|0.1|-0.0016|+0.058|
|right|ramp|0.0|-0.0044|+0.058|

## 解释

1. 直接纯转向：左右方向最后5秒净朝向变化均不足0.3°，没有形成持续转向。
2. 行走后纯转向：vx=0.4阶段有明确的指令方向朝向变化；切到vx=0后只剩短暂过渡，末5秒变化不足0.4°。因此本次不支持“只要先走起来就能持续原地转向”。
3. 降速过程：左转在vx=0.2时平均实际vx已接近零，末5秒净转向只有0.18°；右转在vx=0.2时仍前行约0.225m/s、净右转10.17°，降到vx=0.1后也接近停止。
4. 左转腿部关节速度RMS从vx=0.4时约1.578rad/s降到vx=0.2时约0.147rad/s；右转对应1.636→1.315rad/s，并在vx=0.1时降到约0.145rad/s。与抽帧中运动减弱的现象一致。
5. 组合指令虽然能转向，但转向强度仍偏弱：vx=0.4时左、右净朝向变化分别约8.36°和18.93°，约为这一窗口期望转向量的5.9%和13.3%。这与“能转向”同时成立。

上述速度点只限定了本随机种子、本减速路径下的观测差异，不能解释成固定的控制阈值，也不能证明某个奖励项是原因。没有测试负vx后退组合或其他随机种子。

## 对此前表述的补充

此前平均机身局部z轴角速度的报告不能单独否定组合指令下的实际转向。本轮加入连续世界系朝向变化和视频复核，明确记录了组合指令下的转向。局部轴角速度与世界系航向角变化是不同观测量，详细JSON保留两者。

## 关节与视频证据

|试验|速度超限采样数|最大速度/限值|扭矩超限采样数|最大扭矩/限值|
|---|---:|---:|---:|---:|
|left direct|0|0.960|0|1.000|
|left walking|0|0.960|0|1.000|
|left ramp|0|0.960|0|1.000|
|right direct|0|0.960|0|1.000|
|right walking|0|0.960|0|1.000|
|right ramp|0|0.975|0|1.000|

- 本轮六组没有记录到关节速度或实际扭矩超过配置上限；部分扭矩达到配置上限。此前启动即施加转向指令时的单点超速证据仍保留，本轮先站立10秒，条件不同，不能据此删除或否定此前峰值。
- 已人工检查六组代表帧和左转降速各段多帧，机器人在抽查帧中可见；本次无重置出镜问题。抽帧检查不等同于逐帧审查。
- direct视频各999帧、walking各1499帧、ramp各2500帧，均50fps；部分视频比遥测短一个采样周期，视觉时间定位允许约一帧误差。
- 现有脚本没有足端接触力/离地高度遥测，因此不能给出接触率、离地次数等定量结论，也未用零值补缺。

## 建议与范围

下一步优先核查纯转向和低速组合指令的训练覆盖，以及AMP直行专家参考对这些动作的影响；这些是待验证的方向。若要改采样、奖励或添加足端接触遥测，应先形成具体代码修改方案并获得用户批准。本轮没有修改源代码、训练配置或启动训练。

本次为单检查点、单种子的行为诊断，置信度中等；没有批准的收敛数值判据，正式收敛状态仍为 indeterminate。没有选择最终检查点、导出或归档，仿真评估不代表硬件就绪。

## 查看证据

- [朝向曲线](turn-followup-heading-20260907-001.png)
- [六组视频抽帧](turn-followup-video-review-20260907-001.jpg)
- [完整数字结果](summary-turn-followup-summary-20260907-001.json)

- [left direct 视频](../play/turn-followup-left-direct-20260907-001/video.mp4) · [遥测](../play/turn-followup-left-direct-20260907-001/telemetry.json) · [结果](../play/turn-followup-left-direct-20260907-001/result.json)
- [left walking 视频](../play/turn-followup-left-walking-20260907-001/video.mp4) · [遥测](../play/turn-followup-left-walking-20260907-001/telemetry.json) · [结果](../play/turn-followup-left-walking-20260907-001/result.json)
- [left ramp 视频](../play/turn-followup-left-ramp-20260907-001/video.mp4) · [遥测](../play/turn-followup-left-ramp-20260907-001/telemetry.json) · [结果](../play/turn-followup-left-ramp-20260907-001/result.json)
- [right direct 视频](../play/turn-followup-right-direct-20260907-001/video.mp4) · [遥测](../play/turn-followup-right-direct-20260907-001/telemetry.json) · [结果](../play/turn-followup-right-direct-20260907-001/result.json)
- [right walking 视频](../play/turn-followup-right-walking-20260907-001/video.mp4) · [遥测](../play/turn-followup-right-walking-20260907-001/telemetry.json) · [结果](../play/turn-followup-right-walking-20260907-001/result.json)
- [right ramp 视频](../play/turn-followup-right-ramp-20260907-001/video.mp4) · [遥测](../play/turn-followup-right-ramp-20260907-001/telemetry.json) · [结果](../play/turn-followup-right-ramp-20260907-001/result.json)

# 2026-10-21_10-48-07

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [wheel-to-leg-knee-ground-20261022-005](<evaluations/wheel-to-leg-knee-ground-20261022-005/report.md>)：1 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/wheel-to-leg-knee-ground-20261022-005/manifest.json>)。
- [wheel-to-leg-zero-contact-review-20261022-001](<evaluations/wheel-to-leg-zero-contact-review-20261022-001/report.md>)：4 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/wheel-to-leg-zero-contact-review-20261022-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [events](<events>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 已有诊断图与数据

- [接触奖励关闭前后的跳跃诊断](evidence/training/jump-diagnostics-20261022-001.json)
- [接触奖励关闭前后的曲线](evidence/training/contact-reward-ablation-20261022-001.png)
- [训练末段日志摘要](evidence/training/summary-20261022-001.json)
- [TensorBoard 最终迭代核验](evidence/training/tensorboard-tail-20261022-001.json)
- [起跳速度跟踪数据](evidence/training/takeoff-velocity-validation-20261022-001.json)
- [起跳速度跟踪曲线](evidence/training/takeoff-velocity-validation-20261022-001.png)

## 膝部附近对地接触核验

两次策略在相同播放条件下各完成 1000 步，采样间隔 5 ms，均为 5/5 完整周期出现大腿连杆对地接触；小腿连杆未检出对地接触。场景碰撞体限于机器人与地面，对地法向力由总法向接触力减去机器人自接触力得到，独立接触力读数逐步交叉核验。新增观测前后的基座位置轨迹逐帧完全一致。STL 几何最低点表明接触位置在大腿末端的膝部区域；该位置是几何推断，GPU 接口未提供实测接触点。

- [两次策略的接触力与膝部高度曲线](evidence/training/knee-ground-contact-comparison-20261022-001.png)
- [逐周期统计、来源哈希与几何核验](evidence/training/knee-ground-contact-comparison-20261022-001.json)

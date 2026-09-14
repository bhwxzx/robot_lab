# 2026-09-11_16-56-27

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [flat-review-20260912-001](<evaluations/flat-review-20260912-001/report.md>)：6 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/flat-review-20260912-001/manifest.json>)。
- [flat-review-smoke-20260912-001](<evaluations/flat-review-smoke-20260912-001/report.md>)：1 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/flat-review-smoke-20260912-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [events](<events>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 本轮分窗口分析

- [Native 分窗口指标、视频抽查与限制](evidence/assessment/native-review-20260912.json)
- [训练末段统计](evidence/training/training-summary-20260912.json)
- [与上一版训练的有效配置差异](evidence/training/effective-config-comparison-20260912.json)

- [零指令与纯转向覆盖、奖励语义排查](evidence/assessment/zero-turn-reward-audit-20260912.json)

## 策略导出与归档

- [用户选择记录](evidence/checkpoint_selection/selection-user-current-20260912-001.json)。
- [JIT/ONNX 导出与重置前后一致性证据](evidence/export/static-20260912-002/receipt.json)：16 样本，static batch 1，obs[1,410] → actions[1,10]。
- [归档回执](evidence/checkpoint_selection/archive-receipt-20260912-001.json)：`policy_storage/LW/leg_loco/2026-09-11-16-56-27`，目录时间为训练开始时间。
- [归档清单与策略说明来源](evidence/checkpoint_selection/archive-manifest-user-current-20260912-001.json)。
- 首次导出因配置覆盖类型不匹配而退出：[失败日志](evidence/export/static-20260912-001/console.log)；未发布模型，后续新 attempt 已完成。
- [提交与远端核验](evidence/checkpoint_selection/archive-git-receipt-20260912-001.json)：`153c380` 已推送到 `origin/master`，存储仓库干净。

## Sim2Sim 返回结果

- [接收端完整性、6000 次模型重放及独立统计核验](evidence/feedback/sim2sim-receiver-analysis-20260912-001.json)：测试 `20260912-162601-flat-20260911`，6 组完成，闭环 reset 未执行；保留原始报告，另行记录角速度坐标口径修正。
- [相位 A/B 接收端核验：10000 次 ONNX 重放、输入分岔与分窗口统计](evidence/feedback/sim2sim-phase-ab-receiver-analysis-20260912-001.json)：测试 `20260912-202155-leg-phase-ab`；单周期提前已去除，残余 float32 时钟误差约 7.18 微秒；直行停车改善、前进转向后停车退化，纯转向仍弱。

# 2026-09-16_15-12-04

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [native-assess-20260917-001](<evaluations/native-assess-20260917-001/report.md>)：4 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/native-assess-20260917-001/manifest.json>)。
- [roa-noise-ablation-20260917-001](<evaluations/roa-noise-ablation-20260917-001/report.md>)：3 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/roa-noise-ablation-20260917-001/manifest.json>)。
- [roa-true-velocity-20260917-001](<evaluations/roa-true-velocity-20260917-001/report.md>)：1 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/roa-true-velocity-20260917-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [events](<events>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 训练与对比分析

- [新旧策略对比图](<evidence/analysis/comparison-20260917-001.png>)
- [可复核对比数据、窗口定义与配置差异](<evidence/analysis/comparison-20260917-001.json>)
- [训练 TensorBoard 统计](<evidence/training/tensorboard-review-20260917-001.json>)
- [训练来源及最终检查点核实](<evidence/training/preflight-20260917-001.json>)
- [收敛判据检查](<evidence/training/assessment-20260917-001.json>)：缺少本次训练的已批准判据，收敛状态为 indeterminate；该脚本提前返回的 Play available=false 不表示评估未完成，完成状态以批次 manifest 为准。

## 站立噪声消融

- [三项噪声消融对比图与速度估计曲线](<evidence/analysis/noise-ablation-20260917-001.png>)
- [可复核统计、速度误差与采样时序](<evidence/analysis/noise-ablation-20260917-001.json>)
- [测试范围、已有遥测功能与验证记录](<evidence/training/preflight-roa-noise-ablation-20260917-001.json>)

仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

## 真实速度替换消融

- [闭环对比图与同观测配对动作](<evidence/analysis/true-velocity-20260917-001.png>)
- [可复核指标、速度误差和配对动作统计](<evidence/analysis/true-velocity-20260917-001.json>)
- [测试范围与输入替换说明](<evidence/training/preflight-roa-true-velocity-20260917-001.json>)

## 导出与归档

- [用户选定检查点](<evidence/checkpoint_selection/selection-deploy-20260917-001.json>)
- [JIT / ONNX 导出与 8 时刻重置一致性回执](<evidence/export/deploy-20260917-001/receipt.json>)
- [归档回执](<evidence/export/deploy-20260917-001/archive-receipt.json>)
- 归档目录：`/home/server/liufengrong/policy_storage/LW/wheel_loco/2026-09-16-15-12-04`；原始 student 策略，速度估计开启。
- policy_storage 提交：`2830956`。
- [提交与推送记录](<evidence/export/deploy-20260917-001/git-publication.json>)：已推送 `origin/master`。

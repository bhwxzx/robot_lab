# 2026-09-16_15-45-12

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [flat-review-20260917-001](<evaluations/flat-review-20260917-001/report.md>)：7 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/flat-review-20260917-001/manifest.json>)。
- [flat-review-smoke-20260917-001](<evaluations/flat-review-smoke-20260917-001/report.md>)：1 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/flat-review-smoke-20260917-001/manifest.json>)。
- [matched-plane-20260917-001](<evaluations/matched-plane-20260917-001/report.md>)：3 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/matched-plane-20260917-001/manifest.json>)。
- [restitution-ablation-20260919-001](<evaluations/restitution-ablation-20260919-001/report.md>)：9 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/restitution-ablation-20260919-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [events](<events>)
- [evidence/checkpoint_selection](<evidence/checkpoint_selection>)
- [evidence/export](<evidence/export>)
- [evidence/source](<evidence/source>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 遥测分析与核验

- [原七场景姿态与跟踪分析](<evidence/assessment/posture-and-tracking-20260917-001.json>)
- [原七场景对比图](<evidence/assessment/posture-and-tracking-20260917-001.png>)
- [七场景录像抽帧](<evidence/assessment/seven-scenario-video-review-20260917-001.jpg>)
- [原七场景完整性检查](<evidence/assessment/verification-20260917-001.json>)
- [训练配置与完成状态](<evidence/training/configuration-and-completion-20260917-001.json>)
- [训练末段 TensorBoard 窗口](<evidence/training/tensorboard-windows-20260917-001.json>)
- [同平面新旧策略指标与奖励核算](<evidence/assessment/matched-plane-comparison-20260917-001.json>)
- [同平面新旧策略曲线](<evidence/assessment/matched-plane-comparison-20260917-001.png>)
- [同平面对照完整性检查](<evidence/assessment/matched-plane-verification-20260917-001.json>)

## 已导出归档并推送

- 选定 checkpoint：model_50000.pt。
- [导出与重置边界校验](<evidence/export/export-50000-20260917-001/receipt.json>)。
- [归档清单](<evidence/export/export-50000-20260917-001/archive-input.json>)。
- [归档回执](<evidence/export/export-50000-20260917-001/archive-receipt.json>)。
- [推送复核](<evidence/export/export-50000-20260917-001/push-verification.json>)。
- 归档位置：/home/young/liufengrong/policy_storage/LW/leg_loco/2026-09-16-15-45-12/；远端 master 提交 724c8a0c5afb15af377e23ca1514ceaf79a8429b。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

## 恢复系数单变量对照（2026-09-19）

- [九场景批次报告](<evaluations/restitution-ablation-20260919-001/report.md>)：有速度估计版，三种恢复系数 × 三个转向场景，共 180 秒。
- [逐窗口指标与差异核算](<evidence/assessment/restitution-comparison-20260919-001.json>)
- [指标对比图](<evidence/assessment/restitution-metrics-20260919-001.png>)
- [姿态与转速时序图](<evidence/assessment/restitution-traces-20260919-001.png>)
- [收尾校验](<evidence/assessment/restitution-verification-20260919-001.json>)

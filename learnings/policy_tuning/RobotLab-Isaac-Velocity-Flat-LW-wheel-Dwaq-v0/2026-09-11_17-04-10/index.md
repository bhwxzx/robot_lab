# 2026-09-11_17-04-10

此页可重新生成，仅用于导航；证据以各批次 manifest 及其校验链为准。

## 评估批次

- [native-assess-20260912-001](<evaluations/native-assess-20260912-001/report.md>)：3 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/native-assess-20260912-001/manifest.json>)。
- [zero-standing-plane-20260912-001](<evaluations/zero-standing-plane-20260912-001/report.md>)：2 完成 / 0 失败 / 0 未运行；[manifest](<evaluations/zero-standing-plane-20260912-001/manifest.json>)。

## 原始记录与生命周期

- [provenance](<provenance>)
- [events](<events>)
- [evidence/checkpoint_selection](<evidence/checkpoint_selection>)
- [evidence/export](<evidence/export>)
- [evidence/source](<evidence/source>)

旧版证据保持原位；本索引不会移动、压缩或重写已有结果。

## 训练检查与派生图

- [平地零指令站立位置轨迹图](evaluations/zero-standing-plane-20260912-001/standing_drift.png)：由该批次完整遥测派生，PNG 元数据记录源 manifest 哈希；原始结果以批次校验链为准。
- [TensorBoard 完整进度与窗口统计](evidence/training/tensorboard-assess-20260912-001.json)
- [有效配置差异](evidence/training/effective-config-diff-20260912-001.json)
- [主评估判据结果](evidence/training/assessment-post-play-20260912-001.json)
- [使用原训练 PID 1104501 的状态采集](evidence/health/health-assess-20260912-002.json)。health-assess-20260912-001.json 输入 PID 错误，保留审计但不用于结论。

## 策略导出与归档

- 用户选择最新 `model_49999.pt`：[选择记录](evidence/checkpoint_selection/selection-user-latest-20260912-001.json)。
- [JIT / ONNX 导出一致性凭据](evidence/export/user-latest-20260912-001/receipt.json)。
- [归档凭据](evidence/checkpoint_selection/archive-receipt-user-latest-20260912-001.json)：`policy_storage/LW/wheel_loco/2026-09-11-17-04-10`（按训练开始时间更名，原始归档凭据保留历史路径）。
- [提交与远端核验](evidence/checkpoint_selection/git-publish-user-latest-20260912-001.json)：`a9fd8651d9d8403aed914275d519ebb1c181db8b` 已在远端 `master`。
- [归档目录更正记录](evidence/checkpoint_selection/archive-path-correction-20260912-001.json)：按训练开始时间命名，模型内容不变。

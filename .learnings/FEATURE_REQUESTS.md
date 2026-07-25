# Feature Requests

## [FEAT-20260725-001] closed_loop_policy_evaluation

**Logged**: 2026-07-25T14:27:38+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
最终策略不能只依据训练曲线选出；训练 skill 必须通过 Play、Native 与导出产物闭环仿真、压力场景和实际运动视频审阅来决定策略是否通过。

### User Context
奖励和损失曲线无法显示身体晃动、抖振、拖脚、异常转向、扰动恢复或部署产物输入不一致。用户希望在实物部署前用多种模拟实物场景实际观察机器人运动效果，并让失败结果在现有授权范围内参与淘汰或重新调优。

### Complexity Estimate
complex

### Suggested Implementation
在版本化会话契约中加入评估权限、候选产物、场景矩阵、指标门槛、动作一致性和视频审阅要求；提供 RSL-RL Native/JIT/ONNX 闭环评测入口、结果汇总与晋级验证，并让训练排名在评估通过前保持 `final_selection=null`。

### Metadata
- Frequency: first_time
- Related Features: monitor-tune-isaaclab-training, algorithm-profiles, bounded-tuning

### Resolution
- **Resolved**: 2026-07-25T14:27:38+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现版本 3 会话契约、评估矩阵、RSL-RL 闭环执行器、硬指标与视频审阅门槛、结果晋级验证，以及训练排名与最终选择分离。

---

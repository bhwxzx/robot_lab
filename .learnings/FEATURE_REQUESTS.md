# Feature Requests

## [FEAT-20260726-003] staged_multiseed_tuning_execution

**Logged**: 2026-07-26T15:05:43+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
自动调优 Skill 应采用不同 seed，并完成第一轮稳健性优化：低成本筛选、
多 seed 确认、自动执行和恢复、有效配置核验、学习质量异常检测及稳健排名。

### User Context
单 seed 容易把初始化或采样偶然性当成参数改进，但对所有候选直接进行完整
多 seed 训练成本过高。自动执行还必须保持单 GPU、不可重复 run/W&B ID、
只管理自身进程组，并且不能让未授权配置或旧重试产物混入结果。

### Complexity Estimate
complex

### Suggested Implementation
增加版本 6 会话：固定 screening/confirmation seed、top-k 分阶段计划、完全
绑定会话的原子执行状态和 attempt 隔离目录；用 exact argv、GPU 空闲检查与
锁、PID/进程组/start-ticks/argv 身份保护启动和恢复。执行后核验完整有效
配置，只允许审批 override；按连续窗口检测异常。最终按每 seed 约束、
同 seed baseline 配对改进、标准差/范围/95% t 区间、最低改进和 Pareto
前沿排名，同时保持版本 3–5 兼容。

### Metadata
- Frequency: first_time
- Related Features: bounded_tuning, closed_loop_policy_evaluation, hardware_feedback_driven_retuning

### Resolution
- **Resolved**: 2026-07-26T15:05:43+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现版本 6 契约、分阶段 seed 计划、可恢复单任务执行器、有效配置差异门禁、连续窗口异常检测、稳健排名、文档与合成测试；未增加 MuJoCo，也未启动真实训练或写入真实策略仓库。

---

## [FEAT-20260726-002] hardware_feedback_driven_retuning

**Logged**: 2026-07-26T13:30:03+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
训练 Skill 应能根据用户提供的实物部署反馈调整后续调优方案，同时继续由
用户选择是否调优、具体参数和范围。

### User Context
仿真闭环和视频评估仍不能替代机器人实物表现。站立晃动、抖振、跟踪延迟、
打滑、转向或恢复异常需要反馈到下一轮仿真与训练，但部署张量、历史帧、
时序、标定和硬件问题不能被误判成奖励参数问题。

### Complexity Estimate
complex

### Suggested Implementation
用版本 5 会话单独授权实物反馈处理，将反馈绑定到归档清单、JIT/ONNX
哈希、部署配置、测试包线、时间段、视频和遥测；先执行安全、部署和硬件
根因分类，再生成不可执行的调优建议。只列出原会话已授权的参数选项，并在
任何新训练前要求用户重新选择参数、范围、指标、预算并批准新会话。

### Metadata
- Frequency: first_time
- Related Features: qualified_policy_storage_archive, closed_loop_policy_evaluation

### Resolution
- **Resolved**: 2026-07-26T13:30:03+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现版本 5 反馈授权、归档策略和部署配置哈希校验、通用症状与安全事件契约、证据置信度、部署/硬件/训练根因分流，以及 proposal-only 和待批准草案模式。严重事件停止实物测试，草案不自动选择参数且不能执行。

---

## [FEAT-20260726-001] qualified_policy_storage_archive

**Logged**: 2026-07-26T11:20:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
调优并通过最终闭环评估、可以尝试实物部署的策略，应能自动复制到
`/home/young/liufengrong/policy_storage`，同时保存 ONNX、PT 和策略说明。

### User Context
现有策略仓库按机器人、策略类别和时间戳保存部署产物，但依赖人工复制，
容易遗漏格式、说明、来源哈希或仿真与实物验证边界。用户希望训练 Skill
在最终候选通过后完成一致、可审计的自动归档。

### Complexity Estimate
complex

### Suggested Implementation
扩展版本化会话授权，要求 JIT 与 ONNX 都作为必需闭环评估产物；只读检查
目标策略仓库，在工作区干净且哈希、指标、视频审查和最终排名全部一致时，
原子创建时间戳目录并生成 `策略说明.txt` 与结构化清单。归档不自动提交或
推送，也不把仿真合格描述为硬件就绪。

### Metadata
- Frequency: first_time
- Related Features: closed_loop_policy_evaluation, final_policy_promotion

### Resolution
- **Resolved**: 2026-07-26T13:20:00+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现版本 4 归档授权、真实仓库只读检查、JIT/ONNX 双格式与哈希门禁、最终排名重算、原子时间戳归档、中文策略说明、结构化清单，以及禁止自动 Git 操作和硬件就绪声明的边界。

---

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

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
## [FEAT-20260728-003] idempotent_campaign_controller

**Logged**: 2026-07-28T12:40:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tuning

### Requested Capability
为自动调优 Skill 增加默认只读预演、显式授权后可执行的幂等 Campaign
Controller，把单机执行器、多阶段晋级和双机 Git 邮箱的阶段边界自动串联。

### User Context
训练、续训、晋级和双机通信的底层原语已经具备，但仍需人工判断并依次调用
多个命令。重复调用、进程中断、部分远端结果或计划文件不一致时容易产生操作
错误，用户希望配置后由控制器持续推进，同时保持参数、归档和实物测试权限
边界不变。

### Complexity Estimate
complex

### Suggested Implementation
新增会话级 `campaign_controller` 授权和独立控制器脚本。`shadow` 只输出下一
动作；`execute` 每次最多启动一个 child 或提交一个不可变阶段决策。单机复用
执行器哈希链；双机邮箱发布完整 JSON 计划快照并保持 worker/job/checkpoint
绑定。训练排序后停止，不自动进入评估、归档或实物测试。

### Resolution
**Resolved**: 2026-07-28T13:24:03+08:00

已实现版本 6/7 `campaign_controller` 授权、默认只读 shadow、显式 execute、
单机和双机角色、原子状态与哈希链、一次一个状态转换、自动 child
启动/reconcile、adaptive/multi-fidelity 计划晋级、Git 邮箱内容寻址计划
快照、worker claim/prepare/result 和最终训练排序停止。旧无快照 campaign
继续保留手动工作流。新增 6 项控制器测试，完整回归 86 项通过；未启动真实
训练、未操作真实远端、未归档策略或进入实物测试。

### Metadata
- Frequency: recurring
- Related Features: synchronous_multi_fidelity_training, distributed_git_mailbox

---
## [FEAT-20260728-002] synchronous_multi_fidelity_training

**Logged**: 2026-07-28T12:08:28+08:00
**Priority**: high
**Status**: resolved
**Area**: tuning

### Requested Capability
为自动调优 Skill 增加同步多阶段训练预算：所有候选先以较小预算训练，只让
满足安全约束且有竞争力的候选从精确 checkpoint 继续到后续预算，减少明显
落后候选占用单机或双机算力。

### User Context
现有自适应搜索能决定下一轮是否继续，却仍会让同一轮的每个候选跑完整训练。
用户采用固定单 seed，主要比较奖励权重和参数差异，适合在相同进度屏障上进行
保守晋级，同时保持最终实物部署为权威。

### Complexity Estimate
complex

### Suggested Implementation
新增可选 `multi_fidelity` 会话契约、确定性 rung 计划和连续落后保护；每个
rung 必须覆盖同期 baseline，先应用硬约束，再用批准目标和 margin 决定晋级。
RSL-RL 适配器绑定预算 Hydra 路径、父 checkpoint 哈希和续训路径；单机执行器
采用 append-only 计划，双机 Git 邮箱保持 trial 到 worker 的亲和性并只交换
checkpoint 元数据。

### Resolution
**Resolved**: 2026-07-28T12:26:12+08:00

已实现版本 6/7 固定单 seed 同步多阶段契约、确定性 rung 屏障、硬约束即时
淘汰、至少两阶段且连续落后后才进行性能剪枝、baseline 全程保留，以及最终
零任务决策。RSL-RL 适配器校验父 checkpoint 路径、哈希和 step 后续训；
单机执行器支持 append-only `adopt-plan`，Git 邮箱保持同 worker 亲和并只
交换 checkpoint 元数据。新增独立参考文档与 5 项端到端测试；完整 Skill
回归共 80 项测试通过。

### Metadata
- Frequency: recurring
- Related Features: history_quality_and_adaptive_early_stop, distributed_git_mailbox

---
## [FEAT-20260728-001] history_quality_and_adaptive_early_stop

**Logged**: 2026-07-28T11:55:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tuning

### Requested Capability
继续优化自动调优 Skill，使历史记录只有在任务、算法、观测和奖励配置可比且
训练质量足够时才影响采样；同时在继续训练已无明显收益时自动停止后续轮次，
并解释每个候选的来源。

### User Context
现有实现限制了历史 run 和点数，但旧源码或不同训练上下文仍可能误导首轮；
固定最大轮数也可能在已无有效提升时继续消耗双机训练预算。

### Complexity Estimate
complex

### Suggested Implementation
为本地 W&B 索引加入源码策略、四项上下文指纹、最终进度、尾部统计和稳定性
门；为计划加入利用/探索来源、anchor 距离与非重复证明。每轮用批准的原始
目标指标、最小提升、patience 和可行候选数生成确定性 continue/stop 决策；
单机执行器与 Git 邮箱对 stop 决策均不得创建新任务。

### Metadata
- Frequency: recurring
- Related Features: bounded_local_history_adaptive_tuning, distributed_git_mailbox

### Resolution
- **Resolved**: 2026-07-28T12:00:12+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现来源提交与四项上下文指纹策略、进度/点数/稳定性质量门、可信先验 schema 2、候选来源和距离证据、确定性早停决策，以及单机执行器和双机 Git 邮箱的零新任务停止语义；技能校验、语法检查和 75 项单元测试全部通过，未启动真实训练或写入策略仓库。

---

## [FEAT-20260726-006] transactional_policy_evaluation_execution

**Logged**: 2026-07-26T18:52:49+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
把 Native/JIT/ONNX 策略评估从计划、人工启动和结果汇总升级为可恢复的
自动执行流水线，并自动提供运动风险视频时间段和两层部署一致性证据。

### User Context
训练与多 seed 调优已经具备事务化执行，但最终 Play/部署产物验证仍存在
人工断点。策略不能只靠训练曲线或单步 action 一致性晋级，还需要同场景
闭环行为差异、完整视频和实际视觉审核。

### Complexity Estimate
complex

### Suggested Implementation
新增一次只执行一个矩阵单元的评估状态机，使用 attempt 隔离、共享 GPU
锁、精确进程身份、超时升级、启动回执和 hash-chain journal；完成后重新
核验 checkpoint/artifact，原子晋升规范结果与视频。评估器记录峰值 step
及视频审阅窗口，并在相同 scenario/seed 下比较部署产物与 Native 的闭环
指标差异。结果汇总绑定执行状态中的结果与视频哈希。

### Metadata
- Frequency: recurring
- Related Features: closed_loop_policy_evaluation, live_transactional_training_supervision

### Resolution
- **Resolved**: 2026-07-26T18:52:49+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现事务化评估执行、训练/评估共享 GPU 锁、闭环 parity、运动证据窗口、状态绑定汇总及崩溃、超时、篡改、视频缺失和恢复测试；真实 Isaac Sim 评估仍受空闲 GPU 门禁约束。

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
## [FEAT-20260726-004] concrete_rsl_rl_tuning_executor

**Logged**: 2026-07-26T17:05:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
把自动调优 Skill 从合成执行契约推进为可调用真实 RSL-RL 训练入口的执行器，
同时增加超时、资源门禁、并发状态保护、可恢复日志和故障测试。

### User Context
第一轮已经具备分阶段多 seed、稳健排名和仿真评估，但还需要把审批参数安全
映射到真实训练命令，并在长期自动运行时避免重复启动、无界等待、磁盘耗尽、
GPU 过热或状态文件损坏。

### Complexity Estimate
complex

### Suggested Implementation
增加 RSL-RL adapter contract，将已审批参数映射到 Hydra 路径并生成唯一 seed
和 run name；从真实 stdout 定位日志目录，读取 dump 配置、提取指标和
checkpoint 哈希。执行器增加状态锁、总/单次超时、SIGTERM 到精确 SIGKILL
升级、磁盘/GPU 健康门禁和 hash-chain 状态日志，并用合成 RSL-RL 与故障
注入测试覆盖。

### Metadata
- Frequency: first_time
- Related Features: robust_multiseed_automated_tuning, training_watchdog

### Resolution
- **Resolved**: 2026-07-26T17:05:00+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现真实 RSL-RL adapter、运行身份配置归一化、一次性 baseline 配置引导、终端/checkpoint 回执、状态锁、资源与时间限制、优雅终止和 hash-chain 恢复；真实 GPU smoke 因现有训练占用而未启动。

---
## [FEAT-20260726-005] live_transactional_training_supervision

**Logged**: 2026-07-26T18:15:00+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
继续优化自动调优 Skill，使真实 RSL-RL 训练在运行期间即可被监督和安全
停止，同时强化调度器崩溃恢复并记录可复现实验环境。

### User Context
训练结束后的指标检查无法及时阻止 NaN、质量崩塌或资源浪费；启动进程与
写状态之间的故障窗口也可能留下重复任务或不可复用 attempt。正式自动调优
前需要在线证据、事务化状态和可复现清单。

### Complexity Estimate
complex

### Suggested Implementation
将日志解析器改为流式状态机，在完整 iteration 和非有限指标时刷新原子摘要；
加入 warm-up 门槛与 TensorBoard 第二证据。用 reserve-attempt、launch receipt
和 hash-chain 状态构成两阶段启动，支持截断尾记录恢复与孤儿阻断。每次
trial 记录 Git、命令、运行库、GPU 和审批输入文件哈希。

### Metadata
- Frequency: recurring
- Related Features: concrete_rsl_rl_tuning_executor, training_watchdog

### Resolution
- **Resolved**: 2026-07-26T18:15:00+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现流式摘要、实时非有限检测、warm-up、TensorBoard 辅助证据、事务化 attempt、launch receipt 恢复、截断 journal 尾修复和可复现清单，并通过慢训练、崩塌、NaN、启动失败和恢复测试。真实 PPO smoke 因现有 GPU 训练占用而延期。

---
## [FEAT-20260727-001] approval_gated_first_run_configuration

**Logged**: 2026-07-27T16:07:08+08:00
**Priority**: high
**Status**: resolved
**Area**: config

### Requested Capability
训练监督与自动调优 Skill 首次运行时，应引导并配置本机运行目录、Conda、
GPU、策略仓库，以及双机互通所需的专用 Git coordination 仓库、机器身份和
worker 分支。

### User Context
双机通过 Git 邮箱协作需要两边使用一致的远端、机器表和分支，同时每台机器
又有不同的源码、状态、评估和反馈路径。靠临时手工填写容易把运行文件放入
源码仓库、泄露凭据、覆盖既有配置，或在未验证环境时直接开始训练。

### Complexity Estimate
complex

### Suggested Implementation
增加哈希绑定的 `plan/apply/verify` 首次运行工具。先生成零执行计划并展示
精确 SHA-256，批准后只创建本地目录、写入新配置和回执、克隆已有私有 HTTPS
远端；禁止 push、凭据落盘、覆盖、reset、stash、删除和安装。最后只读验证
Git 根目录/远端、Conda、GPU、目录、policy storage，并让双机分别绑定
`local_machine_id`。

### Metadata
- Frequency: first_time
- Related Features: distributed_git_mailbox, fixed_single_seed, training_watchdog

### Resolution
- **Resolved**: 2026-07-27T16:07:08+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现首次运行计划、精确哈希批准、幂等应用、环境/漂移验证、双机参考流程及安全测试；未创建或写入任何真实远端。

---
## [FEAT-20260727-002] shared_policy_storage_archive_lease

**Logged**: 2026-07-27T20:07:52+08:00
**Priority**: high
**Status**: resolved
**Area**: infra

### Requested Capability
两台调优电脑共用一个 `policy_storage` Git 仓库时，Skill 应避免并发归档、
相互覆盖或基于过期分支写入，同时继续保存 JIT、ONNX 和策略说明。

### User Context
双机按不同奖励权重和参数组合并行训练后，都可能产生可归档候选。本地文件锁
只能保护单个 clone，不能协调另一台电脑；策略二进制又不应进入元数据邮箱。

### Complexity Estimate
complex

### Suggested Implementation
在版本 7 会话中绑定共享策略远端、分支、授权 worker 和各机本地 clone。
通过现有 Git 邮箱发布哈希绑定的 request，由协调机授予全局唯一 lease；
worker 在远端 HEAD 一致且工作区干净时归档。策略仓库的 commit/push 保持
独立批准，完成后发布远端提交证据，再由协调机复核并 release。失败接管只能
显式 revoke，不允许按超时或静默自动推断。

### Metadata
- Frequency: recurring
- Related Features: qualified_policy_storage_archive, distributed_git_mailbox

### Resolution
- **Resolved**: 2026-07-27T20:07:52+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现首次配置远端绑定、request/grant/prepare/complete/release/revoke、归档器租约校验、不可变事件校验、远端提交复核、文档与端到端测试；未操作真实远端或真实策略仓库。

---
## [FEAT-20260727-003] bounded_local_history_adaptive_tuning

**Logged**: 2026-07-27T22:10:00+08:00
**Priority**: high
**Status**: resolved
**Area**: tuning

### Requested Capability
自动调优应从本机已有 W&B 训练记录中提取少量先验，决定第一轮如何采样，并在
每轮结果完成后自动生成下一轮候选；双机模式下通过 Git 邮箱合并两边的历史
摘要和发布新增任务。

### User Context
固定同一 seed 时，主要实验价值来自不同奖励权重和参数组合。完全静态均匀
采样会忽略已有实验，但读取过多历史又会增加成本、放大旧配置偏差，并压缩新
区域探索。

### Complexity Estimate
complex

### Suggested Implementation
只读本地 W&B 二进制记录，不访问云端；按会话精确映射参数与指标，限制时间、
run 数和每 run 保留点数。历史只影响首轮一部分候选，其余保持确定性探索；
后续轮次仅使用全部已完成 trial 的约束与目标，生成可重建的只追加计划。双机
仅交换哈希绑定 JSON 元数据。

### Metadata
- Frequency: recurring
- Related Features: fixed_single_seed, distributed_git_mailbox

### Resolution
- **Resolved**: 2026-07-27T22:10:00+08:00
- **Commit/PR**: N/A
- **Notes**: 已实现本地历史索引、全局有界合并、首轮先验与探索混合、后续轮次扩展、单机计划接纳、双机元数据交换及防篡改测试。

---

# 评估批次 isaac-model-input-20260911-002

任务：`RobotLab-Isaac-Velocity-Flat-LW-leg-Amp-Roa-v0`；训练：`2026-09-04_11-16-35`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| stand-trace-a02 | completed | stand-model-input-trace，500 / 1 / 42 | complete | [结果](<raw/stand-trace-a02/result.json>) / [日志](<raw/stand-trace-a02/console.log>) |

## 策略与场景

- checkpoint SHA-256：`59ab5245acb17de8ea1b79e924c2f0189765be3c51c85f437b22b8e2423b0e19`；runner：`OnPolicyRunnerAmpROA`。
- [场景来源 scenario-2988af16942c77c32872922e00e664ddb84274fcfffa1d76f03cd8363add2760.json](<../../provenance/scenario-2988af16942c77c32872922e00e664ddb84274fcfffa1d76f03cd8363add2760.json>)
- [训练上下文 context-09384ce54946455e5b0d358798a4a4ecffd22ed40dcfb90259037663e8388882.json](<../../provenance/context-09384ce54946455e5b0d358798a4a4ecffd22ed40dcfb90259037663e8388882.json>)
- [训练有效配置 config-60b130fed4816625417deec5d99ba9ddd74eb87f4922dbcbe81bd84bc7920a74.json](<../../provenance/config-60b130fed4816625417deec5d99ba9ddd74eb87f4922dbcbe81bd84bc7920a74.json>)

- `stand-trace-a02` 指标：`{"max_joint_velocity_utilization": 0.9597506523132324, "max_tilt": 0.08984678983688354, "termination_rate": 0.0, "tracking_xy_rmse": 0.0793526608989271, "tracking_yaw_rmse": 0.06699987027339803}`
- `stand-trace-a02` 命令调度：`[{"command": [0, 0, 0], "end_step": 499, "start_step": 0}]`；训练配置覆盖：`{"episode_length_s": 65.0, "events.randomize_push_robot": null}`。

## 观察与限制

- 本批次直接确认两列实际模型输入。model_50000.pt，Native，1 环境、seed 42、500 步 × 0.02 s = 10 s，全程零指令；关闭随机推力，其余训练配置中的随机化和观测噪声保留。遥测 complete、无缺失信号，500 个推理入口样本连续，episode_step 0..499，无终止或超时。
- 采集方法：在 Isaac 和模型导入完成后安装只读 sys.setprofile，在 ActorCriticROA.act_inference 真实调用入口复制 obs[policy]，并记录 joint_pos_rel_without_wheel 返回值、当时的 q/default_q、解析后的关节 ID、_process_policy_obs 返回值和 Native 动作。未修改函数参数、张量、源码或模型。精确启动代码和 argv 分别见 trace_launch.json 与 console.log；模型输入 gzip 为标准 500 行 JSONL。
- 实际 policy 输入 shape=[1,10,41]。每帧 41 维按 base_ang_vel(3)、projected_gravity(3)、velocity_commands(3)、joint_pos(10)、joint_vel(10)、actions(10)、gait_phase(2) 排列。10 帧展平为 410 维；以下索引全部从 0 开始。当前帧为最后 41 维。500 次推理各有 2 次 _process_policy_obs 返回，其 current_frame 和 flat_history 分别与捕获输入最后 41 维和全部 410 维逐元素完全相同，最大误差 0；actor_obs_normalizer 实际类型为 Identity。
- 现场解析的 policy_joint_ids=[1,0,3,2,5,4,8,6,9,7]，对应 right_hip、left_hip、right_thigh、left_thigh、right_shank、left_shank、right_foot、left_foot、right_wheel、left_wheel。wheel_native_ids=[7,9]，对应资产原始顺序的 left_wheel、right_wheel。观测已重排为 policy 顺序后，函数继续使用原始资产索引 [7,9] 置零，实际命中 left_foot 和 left_wheel，漏掉 right_wheel。
- 左脚：joint_pos 块第 7 列 / 每帧第 16 列 / 当前帧在 410 维输入中的第 385 列。500 帧加噪前严格为 0（max_abs=0），但真实 q-default 范围为 [-0.273709625, 0.170225680] rad，均值 0.112556019 rad。模型实际输入范围 [-0.009983250, 0.009964751] rad，均值 0.000278955 rad。因此最终输入并非严格 0，而是错误置零后叠加的 ±0.01 rad 观测噪声。
- 右轮：joint_pos 块第 8 列 / 每帧第 17 列 / 当前帧在 410 维输入中的第 386 列。500 帧加噪前与原始资产右轮 q-default 逐帧相等，最大误差 0；q-default 范围 [0.048265688, 0.211964130] rad，均值 0.106006426 rad。模型实际输入范围 [0.045779414, 0.215136781] rad，均值 0.106112323 rad；输入减去 q-default 的范围 [-0.009990945, 0.009994373] rad，符合实际启用的 uniform additive [-0.01,0.01]。
- 同步核验：每条观测函数采样的 q/default_q 与同次模型调用入口采样值完全相同；全部 10 列实际输入与函数加噪前返回值之差均在 ±0.01 rad 内。左轮 joint_pos 第 9 列加噪前也严格为 0，作为对照。
- 具体例子：step=250（推理入口 t=5.00 s），左脚真实 q-default=0.123207688 rad，加噪前=0，模型输入=-0.006285425 rad；右轮真实 q-default=0.099405482 rad，加噪前=0.099405482 rad，模型输入=0.101328686 rad。
- 附带姿态读数：推理入口 t∈[5,10) s 的机身后仰角均值 2.274955°、标准差 0.700227°。角度由 body-to-world wxyz 四元数计算 asin(2*(qx*qz-qw*qy))，后仰为正。此窗口使用推理前状态，旧评估报告使用 step 后状态，两者有一帧采样时间差。
- 与上一份 MuJoCo 实测包对照，MuJoCo 保留左脚 q-default、将右轮位置置零；本次直接证明 Isaac 实际推理输入与该部署输入存在语义差异。该差异可作为优先排查对象；本批次没有改变输入语义的闭环 A/B，不能据此断言它单独造成或完全解释 MuJoCo 后仰。
- 首批 isaac-model-input-20260911-001 的 Native 仿真完成，但额外输入采集未发布；本批次使用新的 batch_id/attempt_id，并在模拟器关闭前完成采集发布。首批日志和结果保留，不能作为两列模型输入的直接证据。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [trace_launch.json](<raw/stand-trace-a02/trace_launch.json>)
- [model_input_trace_metadata.json](<raw/stand-trace-a02/model_input_trace_metadata.json>)
- [model_inputs.jsonl.gz](<raw/stand-trace-a02/model_inputs.jsonl.gz>)

## 建议与待授权事项

- 先在 MuJoCo 中对现有权重做受控兼容性 A/B：基线维持当前部署语义；对照按 Isaac 已证实的列语义构建全部 10 帧历史，匹配初态、指令、seed 和其余运行配置，比较后仰角、速度与稳定性。实现方案需另行批准。
- 随后再单独制定训练侧关节索引修复及旧策略兼容/重训方案；本批次未修改观测代码、训练参数或 policy_storage。

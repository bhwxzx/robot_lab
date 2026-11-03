# 评估批次 wheel-to-leg-fixed-start-20261103-001

任务：`RobotLab-Isaac-BeyondMimic-Flat-LW-wheel-v0`；训练：`2026-11-02_14-06-00`。

本报告汇总已校验的用例；completed 表示证据发布完成，策略表现须结合指标和视频判断。

| 用例 / 尝试 | 状态 | 场景、步数 / 环境 / seed | 遥测 | 结果与视频 |
| --- | --- | --- | --- | --- |
| fixed-start-nominal-a01 | completed | fixed-start-nominal，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-nominal-a01/result.json>) / [日志](<raw/fixed-start-nominal-a01/console.log>) |
| fixed-start-randomized-a01 | completed | fixed-start-randomized，1000 / 1 / 42 | complete | [结果](<raw/fixed-start-randomized-a01/result.json>) / [日志](<raw/fixed-start-randomized-a01/console.log>) |

## 策略与场景

- checkpoint SHA-256：`493ebd7fbecb527f763de79d45cf884b742bd001cea28d8ff5130b260783165a`；runner：`OnPolicyRunner`。
- [场景来源 scenario-bd7cec1cba2fd2b3be0d88477cbb53e894f6aa31b85eadf2d61f5ef667daf7d6.json](<../../provenance/scenario-bd7cec1cba2fd2b3be0d88477cbb53e894f6aa31b85eadf2d61f5ef667daf7d6.json>)
- [场景来源 scenario-df7f38a2edb6ae94a9eaedef0593de7a7a7f8ff57baccbf5f07fe155c3059154.json](<../../provenance/scenario-df7f38a2edb6ae94a9eaedef0593de7a7a7f8ff57baccbf5f07fe155c3059154.json>)
- [训练上下文 context-51a50409569313b0f83f235be701404c6ab754a9989edfca970d9134feecc7c2.json](<../../provenance/context-51a50409569313b0f83f235be701404c6ab754a9989edfca970d9134feecc7c2.json>)
- [训练有效配置 config-4728f2258c6b63e7673cdf93612581c7d5a9376cf4af83912adfc8497915ce9d.json](<../../provenance/config-4728f2258c6b63e7673cdf93612581c7d5a9376cf4af83912adfc8497915ce9d.json>)

- `fixed-start-nominal-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-nominal-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "nominal_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.undesired_contacts.weight": -1.0, "terminations.thigh_shank_ground_contact": null}`。
- `fixed-start-randomized-a01` 指标：`{"max_joint_velocity_utilization": "unavailable", "max_tilt": "unavailable", "termination_rate": "unavailable", "tracking_xy_rmse": "unavailable", "tracking_yaw_rmse": "unavailable"}`
- `fixed-start-randomized-a01` 命令调度：`[]`；训练配置覆盖：`{"evaluation.motion_ground_contacts": true, "evaluation.motion_mode": "randomized_start", "evaluation.motion_physics_window": [0, 169], "evaluation.motion_sha256": "8ffd6f093bdeacf2f8aec5fa0ccf0c660a51d0fdc6e073ecc7bc0ffe6fa5da03", "rewards.undesired_contacts.weight": -1.0, "terminations.thigh_shank_ground_contact": null}`。

## 观察与限制

- 训练来源已由用户于2026-11-03确认：bash scripts/start_beyondmimic.sh --type wheel，无额外CLI/Hydra覆盖。训练源码为e731215及日志保存的两处修改；与当前干净工作区区分记录。
- 本场次anchor_pos高度误差阈值0.25m、undesired_contacts=-1.0、大腿/小腿触地终止关闭、hip_pos_zero_l2=-2.0。场景覆盖只在评估进程内恢复保存配置，工作区源码不变。
- Native mean动作；两组均从参考第0帧起始，1环境、seed42、1000控制步；名义条件关闭随机化，随机化组保留训练随机化。每控制步20ms、每物理步5ms。未生成视频。
- 起跳窗口1.0<=t<1.5s；腾空分析窗口1.5<=t<2.0s；着地分析窗口2.0<=t<3.4s，均按参考相位定义，不能当作测得的离地/落地时刻。髋镜像误差为abs(q_left+q_right)。
- 对照为上次已评估的本机2026-10-30_12-41-07终版；当前改变了终止和奖励配置，完成率及reward不可直接比较，其他指标仅为描述性对照。单seed、少量周期不能证明泛化或收敛。
- 接触>1N仅沿用原触地终止的检测阈值；接触力排除了传感器测得的机器人自接触。八连杆支撑力不包含base/hip；驱动力矩不包含全部碰撞载荷。
- 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。
- nominal: 完整动作结束5次，失败0次，尾部未完整周期1；完整遥测1000控制样本、4000物理样本。
- nominal: 起跳髋镜像误差峰值4.335→3.098°，当前峰值细节{'phase_s': 1.34, 'episode_id': 0, 'left_hip_deg': 2.5824793706996, 'right_hip_deg': -5.680788287896903}；右髋相邻20ms最大变化0.471°。
- nominal: 全程俯仰跟踪RMSE 2.810→2.732°，最大绝对俯仰9.413°，发生相位1.520s；最大横向姿态roll 3.252°。
- nominal: 大腿/小腿地面接触峰值5220.266N，>1N累计0.330s、最长连续0.055s；八连杆地面竖直力总和峰值5220.266N。
- nominal: 起跳竖直速度峰值1.120→0.674m/s；水平anchor偏差峰值0.066m。
- nominal initialization: 俯仰跟踪RMSE 4.242°，高度跟踪RMSE 0.901cm。
- nominal takeoff: 俯仰跟踪RMSE 2.918°，高度跟踪RMSE 7.790cm。
- nominal flight: 俯仰跟踪RMSE 1.687°，高度跟踪RMSE 10.454cm。
- nominal landing: 俯仰跟踪RMSE 0.947°，高度跟踪RMSE 0.628cm。
- nominal left_hip_joint: 速度峰值5.566rad/s，驱动力矩峰值21.114Nm，力矩饱和0个5ms样本。
- nominal right_hip_joint: 速度峰值6.372rad/s，驱动力矩峰值28.952Nm，力矩饱和0个5ms样本。
- nominal left_thigh_joint: 速度峰值5.522rad/s，驱动力矩峰值24.472Nm，力矩饱和0个5ms样本。
- nominal right_thigh_joint: 速度峰值6.224rad/s，驱动力矩峰值31.355Nm，力矩饱和0个5ms样本。
- nominal left_shank_joint: 速度峰值6.608rad/s，驱动力矩峰值56.259Nm，力矩饱和0个5ms样本。
- nominal right_shank_joint: 速度峰值6.424rad/s，驱动力矩峰值46.816Nm，力矩饱和0个5ms样本。
- nominal left_foot_joint: 速度峰值6.913rad/s，驱动力矩峰值7.446Nm，力矩饱和0个5ms样本。
- nominal left_wheel_joint: 速度峰值6.941rad/s，驱动力矩峰值3.603Nm，力矩饱和0个5ms样本。
- nominal right_foot_joint: 速度峰值5.294rad/s，驱动力矩峰值10.159Nm，力矩饱和0个5ms样本。
- nominal right_wheel_joint: 速度峰值9.349rad/s，驱动力矩峰值3.574Nm，力矩饱和0个5ms样本。
- randomized: 完整动作结束5次，失败0次，尾部未完整周期1；完整遥测1000控制样本、4000物理样本。
- randomized: 起跳髋镜像误差峰值9.480→5.978°，当前峰值细节{'phase_s': 1.22, 'episode_id': 2, 'left_hip_deg': -0.11112664418757855, 'right_hip_deg': -5.866946801272894}；右髋相邻20ms最大变化0.957°。
- randomized: 全程俯仰跟踪RMSE 3.550→3.165°，最大绝对俯仰15.050°，发生相位0.240s；最大横向姿态roll 7.044°。
- randomized: 大腿/小腿地面接触峰值6759.396N，>1N累计0.245s、最长连续0.055s；八连杆地面竖直力总和峰值6759.396N。
- randomized: 起跳竖直速度峰值1.427→0.574m/s；水平anchor偏差峰值0.490m。
- randomized initialization: 俯仰跟踪RMSE 5.053°，高度跟踪RMSE 2.204cm。
- randomized takeoff: 俯仰跟踪RMSE 2.615°，高度跟踪RMSE 7.505cm。
- randomized flight: 俯仰跟踪RMSE 1.912°，高度跟踪RMSE 10.065cm。
- randomized landing: 俯仰跟踪RMSE 1.400°，高度跟踪RMSE 1.051cm。
- randomized left_hip_joint: 速度峰值9.371rad/s，驱动力矩峰值38.374Nm，力矩饱和0个5ms样本。
- randomized right_hip_joint: 速度峰值7.153rad/s，驱动力矩峰值48.645Nm，力矩饱和0个5ms样本。
- randomized left_thigh_joint: 速度峰值8.179rad/s，驱动力矩峰值49.127Nm，力矩饱和0个5ms样本。
- randomized right_thigh_joint: 速度峰值8.034rad/s，驱动力矩峰值34.276Nm，力矩饱和0个5ms样本。
- randomized left_shank_joint: 速度峰值10.626rad/s，驱动力矩峰值82.105Nm，力矩饱和0个5ms样本。
- randomized right_shank_joint: 速度峰值8.980rad/s，驱动力矩峰值59.784Nm，力矩饱和0个5ms样本。
- randomized left_foot_joint: 速度峰值6.927rad/s，驱动力矩峰值15.507Nm，力矩饱和0个5ms样本。
- randomized left_wheel_joint: 速度峰值14.850rad/s，驱动力矩峰值8.129Nm，力矩饱和0个5ms样本。
- randomized right_foot_joint: 速度峰值9.643rad/s，驱动力矩峰值20.605Nm，力矩饱和0个5ms样本。
- randomized right_wheel_joint: 速度峰值11.176rad/s，驱动力矩峰值5.850Nm，力矩饱和0个5ms样本。
- 未提供用户批准的收敛判据时，不作收敛判断；缺失遥测不能按零值解释。
- 视频需经机器人持续在画面内的人工检查后才可作为动作证据。

## 相关证据

- [config-diff-20261103-001.json](<../../evidence/training/config-diff-20261103-001.json>)
- [protection-before-20261103-001.json](<../../evidence/training/protection-before-20261103-001.json>)
- [training-origin-20261103-001.json](<../../evidence/training/training-origin-20261103-001.json>)
- [training-windows-20261103-001.json](<../../evidence/training/training-windows-20261103-001.json>)
- [result.json](<../../../2026-10-30_12-41-07/evaluations/wheel-to-leg-fixed-start-20261031-001/raw/fixed-start-nominal-a01/result.json>)
- [result.json](<../../../2026-10-30_12-41-07/evaluations/wheel-to-leg-fixed-start-20261031-001/raw/fixed-start-randomized-a01/result.json>)
- [analysis.json](<derived/analysis.json>)
- [hip-pitch-height.png](<derived/hip-pitch-height.png>)
- [joint-loads.csv](<derived/joint-loads.csv>)
- [protection-after-20261103-001.json](<../../evidence/training/protection-after-20261103-001.json>)

## 建议与待授权事项

- 结合逐物理步接触、髋镜像误差和俯仰判断动作；修改参数、追加训练、导出或提交须另行授权。

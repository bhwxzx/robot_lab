## [LRN-20260608-001] python_eval_namespace

**Logged**: 2026-06-08T20:32:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
When inheriting from RSL-RL runners (e.g. OnPolicyRunner), newly defined ActorCritic or Algorithm classes fail to initialize with NameError.

### Details
The `_construct_algorithm` method uses `eval(class_name)` to instantiate the classes. Because `eval` operates in the lexical scope of the file where it is defined (the parent runner's file), it cannot resolve child classes or custom algorithm classes unless they are explicitly imported into the parent runner's file (`on_policy_runner.py` or `on_policy_runner_amp.py`).

### Suggested Action
Whenever adding a custom network (e.g., `ActorCriticROA`) or algorithm (e.g., `AMPROAPPO`), explicitly add the import statement for the custom class in the parent runner file where the `eval()` call is located. Also ensure exact string matching for class names.

### Metadata
- Source: error
- Related Files: rsl_rl/runners/on_policy_runner.py, rsl_rl/runners/on_policy_runner_amp.py
- Tags: python, eval, namespace, rsl-rl

---
## [LRN-20260608-002] roa_normalization_logic

**Logged**: 2026-06-08T20:32:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
In ROA architectures, `actor_obs_normalizer` applies exclusively to the current frame observation, not the history latent vector or the history inputs.

### Details
During deployment wrapper construction, there was an incorrect assumption that `student_obs_normalizer` existed and that `actor_obs_normalizer` applied to the concatenated `[current_obs, hist_latent]` tensor. In reality, RSL-RL ROA implementation normalizes only the current frame (`current_obs`), and leaves the history sequence unnormalized before sending it to the 1D-CNN history encoder.

### Suggested Action
When exporting models (e.g., ONNX/JIT), mirror the exact normalization order from `act_inference`: slice the current frame, normalize it, encode the unnormalized history, then concatenate. Do not apply normalization to the final concatenated tensor.

### Metadata
- Source: user_feedback
- Related Files: scripts/reinforcement_learning/rsl_rl/play.py, rsl_rl/modules/actor_critic_roa.py
- Tags: roa, normalizer, deployment, tensor-ops

---
## [LRN-20260608-003] play_py_validation_hist_encoding

**Logged**: 2026-06-08T20:32:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
`play.py` model validation step fails for ROA because `act_inference` defaults to using the privileged encoder instead of the history encoder.

### Details
When generating the JIT/ONNX deployment wrapper, the model is strictly bound to use the history encoder (since privileged data isn't available on the real robot). However, `play.py` checks numerical consistency by calling `policy_nn.act_inference(obs)` which defaults to `hist_encoding=False` (privileged encoder). This causes a huge mathematical discrepancy.

### Suggested Action
In the validation block of `play.py`, inject `hist_encoding=True` when evaluating ROA policies: `policy_nn.act_inference(obs, hist_encoding=True)`. This ensures an apples-to-apples comparison between the deployment model and the original PyTorch model.

### Metadata
- Source: error
- Related Files: scripts/reinforcement_learning/rsl_rl/play.py
- Tags: validation, roa, onnx, jit

---
## [LRN-20260608-004] state_history_encoder_hardcoded_dims

**Logged**: 2026-06-08T20:32:00+08:00
**Priority**: medium
**Status**: resolved
**Area**: config

### Summary
The `StateHistoryEncoder` in IsaacLab / RSL-RL relies on hardcoded 1D-CNN kernels and strides, requiring `history_length` to strictly be 10, 20, or 50.

### Details
The user requested configuring the dimensions of the history encoder. Inspection revealed that the CNN architecture mathematically hardcodes the temporal convolution reduction. Any `history_length` other than 10, 20, or 50 will cause a `ValueError` or tensor mismatch. Furthermore, the output dimension is automatically slaved to the privileged encoder's latent dimension (`priv_out_dim`).

### Suggested Action
Do not attempt to expose history encoder hidden dimensions to the configuration files. Simply enforce that `ObservationsCfg.policy.history_length` is 10, 20, or 50. 

### Metadata
- Source: conversation
- Related Files: rsl_rl/modules/actor_critic_roa.py
- Tags: cnn, history_length, rsl-rl, config

---
## [LRN-20260608-005] dwaq_not_teacher_student

**Logged**: 2026-06-08T20:38:00+08:00
**Priority**: low
**Status**: resolved
**Area**: backend

### Summary
DWAQ is NOT a teacher-student distillation model.

### Details
During a conceptual summary, DWAQ was mistakenly grouped with ROA as a teacher-student distillation model. The user corrected this. DWAQ relies on an asymmetric actor-critic architecture, but it does not employ a two-stage teacher-student privilege distillation process like ROA or RMA.

### Suggested Action
When discussing DWAQ vs ROA, correctly identify their architectural differences. Do not refer to DWAQ as utilizing a teacher-student paradigm.

### Metadata
- Source: user_feedback
- Tags: dwaq, roa, reinforcement-learning, architecture

---

## [LRN-20260609-001] duck_typing_over_isinstance

**Logged**: 2026-06-09T10:55:00+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
When inheriting or creating custom algorithm classes (like `AMPROAPPO`), strict `isinstance(alg, AMPPPO)` checks in parent runners will fail, silently dropping checkpoint data.

### Details
In `on_policy_runner_amp.py`, the `save()` method originally used `isinstance(self.alg, AMPPPO)` to decide whether to save the discriminator and AMP normalizer. Since `AMPROAPPO` does not inherit from `AMPPPO`, this check silently evaluated to False. As a result, the `.pt` checkpoints lacked the `discriminator_state_dict`, causing a complete loss of discriminator weights upon resuming training.

### Suggested Action
Replace hardcoded `isinstance` class checks with duck typing using `hasattr`. E.g., `if hasattr(self.alg, "discriminator"):`. This gracefully supports any algorithm variant that implements the required attributes.

### Metadata
- Source: error
- Related Files: rsl_rl/runners/on_policy_runner_amp.py
- Tags: python, isinstance, duck-typing, checkpoint

---

## [LRN-20260609-002] ai_watchdog_workflow

**Logged**: 2026-06-09T11:20:00+08:00
**Priority**: high
**Status**: promoted
**Area**: infra

### Summary
Established an asynchronous "Watchdog" pattern for the AI agent to monitor long-running background RL training tasks and automatically recover from silent hangs.

### Details
When running long PPO/AMP-ROA trainings, the physics simulation or data loaders may silently hang without crashing (e.g. 0% GPU util, no new terminal logs). 
To solve this without blocking the user conversational flow:
1. Launch the training via `run_command` in the background, redirecting stdout to `train_watchdog.log`.
2. Use the `schedule` tool to set a cron timer (e.g. `*/10 * * * *`).
3. Upon wakeup, the agent executes a quick check on the log file mtime (and optionally `nvidia-smi`) to see if output stopped for >15 mins. (Do not rely on `.pt` files as they save too infrequently).
4. If a hang is detected, use `manage_task` to `kill` the background task and re-launch with `--resume`.
This allows the main AI agent to remain fully responsive to normal chat while seamlessly guarding the training (or delegating it to a subagent).

### Suggested Action
Apply this watchdog pattern (potentially via `invoke_subagent` and `schedule`) whenever the user requests monitoring of a long-running, crash-prone script.

### Metadata
- Source: user_feedback
- Tags: watchdog, asynchronous, background-task, schedule, subagent, recovery
- Promoted: AGENTS.md

---

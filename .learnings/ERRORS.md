## [ERR-20260611-001] ROADeploymentWrapper

**Logged**: 2026-06-11T11:13:36+08:00
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
Play script failed due to ROADeploymentWrapper concatenating a tuple instead of unpacked tensors.

### Error
```python
TypeError: expected Tensor as element 1 in argument 0, but got tuple
```

### Context
- Command/operation attempted: Playing the amp_roa trained policy (`play.py`).
- Input or parameters used: `obs_history_flat` passed to `history_encoder`.
- Environment details if relevant: `ActorCriticROA` history encoder returns `(hist_latent, code_vel)`, but `ROADeploymentWrapper` assigned them to a single variable `hist_latent`.

### Suggested Fix
Unpack the returned tuple `hist_latent, code_vel = self.history_encoder(obs_history_flat)` and concatenate in the correct order `[current_obs, code_vel, hist_latent]` as expected by the actor network.

### Metadata
- Reproducible: yes
- Related Files: scripts/reinforcement_learning/rsl_rl/play.py

### Resolution
- **Resolved**: 2026-06-11T11:13:36+08:00
- **Commit/PR**: N/A
- **Notes**: Unpacked `history_encoder` return values into `hist_latent` and `code_vel`, and updated `torch.cat` order to `(current_obs, code_vel, hist_latent)`.

---

## [ERR-20260717-001] OnPolicyRunnerAmpROA_no_log_mode

**Logged**: 2026-07-17T14:47:37+08:00
**Priority**: low
**Status**: resolved
**Area**: tests

### Summary
AMP_ROA runner crashes after its first iteration when `log_dir=None` unless logging is explicitly disabled.

### Error
```text
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

### Context
- Command/operation attempted: Reduced in-memory AMP_ROA training smoke with `log_dir=None`.
- Input or parameters used: `OnPolicyRunnerAmpROA(..., log_dir=None)` followed by `learn()`.
- Environment details if relevant: At the first iteration, `store_code_state(self.log_dir, ...)` is called when `disable_logs` is false, even though `self.log_dir` is `None`.

### Suggested Fix
Guard the initial code-state snapshot with `self.log_dir is not None`, matching the other logging/save branches. Until fixed, set `runner.disable_logs = True` for in-memory smoke tests.

### Metadata
- Reproducible: yes
- Related Files: rsl_rl/rsl_rl/runners/on_policy_runner_amp.py, rsl_rl/rsl_rl/runners/on_policy_runner_amp_roa.py, rsl_rl/rsl_rl/runners/on_policy_runner_amp_dwaq.py

### Resolution
- **Resolved**: 2026-07-22T15:51:11+08:00
- **Commit/PR**: 8d6090c
- **Notes**: Code-state snapshot calls are now guarded by `log_dir is not None` in AMP, AMP_DWAQ, and AMP_ROA. One-iteration no-log runner tests passed for the ordinary AMP and AMP_DWAQ paths, and the AMP_ROA path was previously verified.

---

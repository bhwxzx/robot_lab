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

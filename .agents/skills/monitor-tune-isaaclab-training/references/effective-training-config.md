# Effective training configuration evidence

Use the RSL-RL files written before `runner.learn()` as the authoritative
effective configuration for a run:

- `<absolute-run-log>/params/env.yaml` contains the resolved environment,
  reward, scene, command, observation, randomization, and termination config;
- `<absolute-run-log>/params/agent.yaml` contains the resolved runner,
  algorithm, seed, iteration, logger, and optimization config.

Do not reconstruct these values from console metrics, TensorBoard scalars, or
W&B metadata. Those sources may corroborate progress or activity but are not
the effective configuration authority.

## Capture contract

First prepare a new evidence snapshot and create its validated host-local run
identity. Once both YAML files exist, run:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/capture_effective_training_config.py \
  "$SOURCE_IDENTITY_PATH" --log-dir "$ABSOLUTE_RUN_LOG_DIRECTORY" \
  --output "$EFFECTIVE_CONFIG_PATH"
```

The helper requires an absolute run directory below the identity repository's
`logs/rsl_rl/` tree. It rejects symlinked inputs, missing files, duplicate YAML
keys, non-finite values, source changes during capture, seed or runner mismatch,
task-command mismatch, run-ID-directory mismatch, experiment-directory
mismatch, and existing output. It reads YAML nodes without constructing
Python-tagged objects.

The version-1 evidence binds task, run, host, and identity SHA-256; embeds the
exact UTF-8 content, path, size, and SHA-256 of both YAML files; records every
reward term including disabled `null` terms; and reports selected runtime
training values. It provides semantic fingerprints for the full environment,
agent, reward subtree, and combined effective configuration.

Treat the full raw YAML as the reproducibility record. Treat extracted reward
weights and training values as a deterministic index for review. A fingerprint
match proves identical captured configuration bytes or semantic YAML content,
not that the live process used an uncaptured file. Keep the run identity's
repository source hashes and command evidence alongside this artifact.

ETP-001 establishes capture. Until ETP-002 is separately approved and
implemented, tuning-experience events and historical queries do not enforce
the presence of this artifact; verify it manually before using parameter
history.

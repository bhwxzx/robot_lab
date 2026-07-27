# First-run configuration

Use this workflow before the first monitoring or tuning session on each
computer. The configuration is machine-local bootstrap state, not permission
to start training, tune parameters, publish Git jobs, archive policies, or run
physical tests.

## Contents

- Safety boundary
- Answer document
- Plan and approval
- Apply and verify
- Two-computer order
- Drift and recovery

## Safety boundary

- Put `configuration_dir`, state, evaluation, feedback, effective-config, and
  mailbox paths outside every `robot_lab` source worktree. Runtime files inside
  the source worktree make the source dirty and block distributed execution.
- Keep runtime/configuration paths outside both the mailbox worktree and
  `policy_storage`; overlapping repositories are rejected.
- Use an existing private HTTPS coordination repository with at least one
  initial commit. Create it in the chosen Git provider before this workflow.
- Never place usernames, passwords, tokens, query strings, or fragments in the
  remote URL or JSON.
- Keep authentication in the operating-system or Git credential helper.
- `apply` may create approved local directories and clone the approved remote.
  It never pushes, resets, stashes, deletes, installs packages, changes source
  branches, or overwrites a different configuration.
- A valid setup receipt does not replace the separate session approval.

## Answer document

Create one JSON answer document for each computer. The shared machine list and
distributed fields must match; change only `local_machine_id` and, where
necessary, the local absolute paths represented by that machine entry.

```json
{
  "version": 1,
  "setup_id": "lw-leg-lab",
  "setup_mode": "git_mailbox",
  "configuration_dir": "/absolute/outside/robot_lab/monitor-tune-config",
  "conda_env": "isaacsim-5.1",
  "default_seed": 42,
  "source_remote_url": "https://git.example.edu/user/robot_lab.git",
  "local_machine_id": "pc-a",
  "machines": [
    {
      "id": "pc-a",
      "source_repo": "/home/user-a/robot_lab",
      "mailbox_repo": "/home/user-a/robot-tuning-mailbox",
      "state_dir": "/home/user-a/robot-tuning-state",
      "effective_config_baseline_path": "/home/user-a/robot-tuning-state/effective-config.json",
      "evaluation_output_dir": "/home/user-a/robot-evaluation",
      "hardware_feedback_output_dir": "/home/user-a/robot-hardware-feedback",
      "policy_storage_root": "/home/user-a/policy_storage",
      "gpu_index": 0,
      "worker_branch": "tune/lw-leg/worker-pc-a"
    },
    {
      "id": "pc-b",
      "source_repo": "/home/user-b/robot_lab",
      "mailbox_repo": "/home/user-b/robot-tuning-mailbox",
      "state_dir": "/home/user-b/robot-tuning-state",
      "effective_config_baseline_path": "/home/user-b/robot-tuning-state/effective-config.json",
      "evaluation_output_dir": "/home/user-b/robot-evaluation",
      "hardware_feedback_output_dir": "/home/user-b/robot-hardware-feedback",
      "policy_storage_root": "/home/user-b/policy_storage",
      "gpu_index": 0,
      "worker_branch": "tune/lw-leg/worker-pc-b"
    }
  ],
  "distributed": {
    "transport": "git_mailbox",
    "remote_url": "https://git.example.edu/user/robot-tuning-mailbox.git",
    "coordinator_id": "pc-a",
    "coordinator_branch": "tune/lw-leg/coordinator",
    "poll_interval_seconds": 600,
    "remote_state_unknown_after_seconds": 1800,
    "artifact_policy": "metadata_only",
    "assignment_mode_default": "by_trial",
    "host_effect_calibration_default_enabled": false
  }
}
```

For one computer, use `setup_mode=single_host`, exactly one machine,
`distributed=null`, and set its `mailbox_repo` and `worker_branch` to `null`.
The default seed is a proposed session default only; every session still needs
explicit approval. Keep `host_effect_calibration_default_enabled=false` for
ordinary reward-weight and parameter searches so each exact trial is assigned
once across the machines. Set it to `true` only when planning an explicitly
approved host-effect diagnostic that repeats the unchanged baseline on every
host.

## Plan and approval

Locate the stable configuration path:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/configure_skill.py \
  locate
```

The default is
`$XDG_CONFIG_HOME/robot-lab/monitor-tune-isaaclab-training/configuration.json`,
falling back to
`~/.config/robot-lab/monitor-tune-isaaclab-training/configuration.json`.
Set `ROBOT_LAB_TUNER_CONFIG` to an absolute path ending in
`configuration.json` when using another location. The setup tool reports
whether the approved plan matches the currently discoverable path; it never
edits shell startup files.

Generate a non-executing plan:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/configure_skill.py \
  plan --answers FIRST_RUN_ANSWERS.json --output FIRST_RUN_PLAN.json
```

Inspect all operations, paths, machine IDs, branches, remote URL, and the
reported `plan_sha256`. Do not apply until the user approves that exact hash.
A changed plan requires new approval.

Apply the exact plan:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/configure_skill.py \
  apply --plan FIRST_RUN_PLAN.json \
  --approval-sha256 EXACT_APPROVED_SHA256
```

An identical second apply returns `already_configured`. A different existing
configuration or receipt is never overwritten.

## Apply and verify

Run full read-only verification after apply:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/configure_skill.py \
  verify --config /absolute/configuration_dir/configuration.json
```

Verification checks the exact source and mailbox Git roots, clean worktrees,
both origin URLs, non-interactive remote reachability, at least one remote branch,
the `isaacsim-5.1` environment, GPU index, required directories, and optional
policy-storage Git root. It never tests push access. Use `--offline` only when
network access is intentionally unavailable; the report then records that
remote connectivity remains unchecked.

Do not start a session unless `ready_for_training=true`. A successful first
mailbox publication is the later proof of push authority and remains a
separately approved external write.

## Two-computer order

1. Create one private coordination remote with an initial commit.
2. Prepare both answer documents and compare every shared field.
3. On PC A, generate the plan, approve its exact hash, apply, and verify.
4. On PC B, repeat with `local_machine_id=pc-b`.
5. Compare configuration fingerprints and machine/branch tables. The full
   fingerprints may differ because `local_machine_id` differs; the shared
   distributed object and machine list must be byte-identical.
6. Only then draft a version-7 training session. Session authorization remains
   responsible for the exact campaign, seed strategy, worker assignment,
   source commit, parameter domains, budgets, evaluation, and recovery rules.

## Drift and recovery

Run `verify` again after changing a repository location, GPU, Conda
environment, remote URL, worker identity, or policy-storage location. If the
configuration must change, generate a new plan in a new configuration
directory or explicitly move the old configuration aside after separate user
approval. Never edit the receipt, silently overwrite the configuration, or
repair a dirty source/mailbox worktree with automatic reset, stash, or delete.

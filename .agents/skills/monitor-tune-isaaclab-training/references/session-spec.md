# Session authorization specification

Use JSON version `6` for automated staged trial execution and robust multi-seed
ranking. Version `6` also supports physical deployment feedback. Version `5`
remains valid for feedback workflows without the executor, version `4` remains
valid for policy evaluation and archival without hardware feedback, and
version `3` remains accepted only for legacy sessions without `archive`.
Resolve the algorithm profile before approval, then validate with
`scripts/validate_session_spec.py`.

## Contents

- Monitor example
- Tune fields
- Automated execution
- Final policy evaluation
- Qualified policy archive
- Hardware-feedback retuning
- Runtime progress snapshot
- Results format

## Monitor example

```json
{
  "version": 5,
  "mode": "monitor",
  "algorithm": {
    "backend": "rsl_rl",
    "name": "PPO",
    "runner_class": "OnPolicyRunner",
    "profile_id": "rsl-rl-ppo",
    "profile_version": 2,
    "profile_fingerprint": "e6e31a40f6934323",
    "unknown_algorithm_policy": "reject"
  },
  "training": {
    "command": [
      "conda", "run", "-n", "isaacsim-5.1", "python",
      "scripts/reinforcement_learning/rsl_rl/train.py",
      "--task=EXACT_TASK",
      "--headless"
    ],
    "resume_command": [
      "conda", "run", "-n", "isaacsim-5.1", "python",
      "scripts/reinforcement_learning/rsl_rl/train.py",
      "--task=EXACT_TASK",
      "--resume",
      "--load_run=RUN_DIRECTORY",
      "--checkpoint=CHECKPOINT_FILE",
      "--headless"
    ],
    "cwd": "/absolute/path/to/robot_lab",
    "log_path": "/absolute/path/to/train_watchdog.log",
    "run_id": "user-chosen-run-id",
    "checkpoint_path": "/absolute/path/to/model_N.pt"
  },
  "monitoring": {
    "check_interval_seconds": 600,
    "stale_after_seconds": 1200,
    "pid": null,
    "gpu_index": 0,
    "tensorboard_path": "/absolute/path/to/run-or-event-file",
    "expected_process_pattern": "rsl_rl/train.py",
    "low_gpu_utilization_percent": 5
  },
  "recovery": {
    "enabled": true,
    "max_restarts": 2,
    "cooldown_seconds": 600
  },
  "tuning": null,
  "evaluation": null,
  "archive": null,
  "hardware_feedback": null,
  "execution": null,
  "cleanup": {
    "remove_created_temp_files": true
  }
}
```

Keep commands as argv arrays. Resolve task, run, checkpoint, backend, algorithm, and runner to exact values. The profile version and fingerprint bind authorization to the reviewed profile semantics.

Use `unknown_algorithm_policy` as follows:

- `reject`: require a known specific profile.
- `runtime_generic`: allow monitor-only operation through a generic profile.
- `propose_persistent`: allow monitor-only operation and generate a candidate profile upgrade.

Generic profiles are invalid in tune mode.

Set `pid` after starting or attaching. The validator checks profile-required resume tokens; it cannot prove that the selected checkpoint is healthy.

## Tune fields

For a version-6 automated run, change `version` to `6`, change `mode` to
`tune`, and replace `tuning`:

```json
{
  "allowed_parameters": [
    {
      "path": "agent.algorithm.learning_rate",
      "values": [0.0003, 0.0005],
      "baseline": 0.0005
    },
    {
      "path": "env.rewards.action_rate_l2.weight",
      "range": {
        "min": -0.2,
        "max": -0.1,
        "step": 0.05
      },
      "baseline": -0.15
    }
  ],
  "protected_parameters_unlocked": [],
  "max_trials": 6,
  "seeds": [42, 43, 44],
  "seed_strategy": {
    "screening_seeds": [42],
    "confirmation_seeds": [42, 43, 44],
    "confirmation_top_k": 2
  },
  "ranking": {
    "require_paired_baseline": true,
    "constraint_scope": "each_seed",
    "minimum_final_training_seeds": 3,
    "pareto_front_required": true
  },
  "trial_timeout_minutes": 120,
  "max_concurrent_trials": 1,
  "mutation_scope": "overrides_only",
  "objectives": [
    {
      "metric": "mean_reward",
      "goal": "maximize",
      "weight": 1,
      "minimum_improvement": 1.0
    },
    {
      "metric": "error_vel_xy",
      "goal": "minimize",
      "weight": 2
    }
  ],
  "constraints": [
    {
      "metric": "illegal_contact",
      "op": "<=",
      "value": 0.06,
      "scope": "each_seed"
    }
  ]
}
```

Treat all example parameter paths as illustrative. Inspect the current effective configuration first.

`max_trials` includes the unchanged baseline. Every configuration runs the same
screening seeds. The baseline and selected top-k candidates then run the
remaining confirmation seeds. `seeds` must exactly equal
`confirmation_seeds`, and screening must be a proper subset so confirmation
adds independent evidence. The plan builder selects a deterministic bounded
subset when the authorized grid is larger than the budget. The grid must expose
at least `confirmation_top_k` non-baseline trials.

Any allowed path matched by the selected profile's protected patterns also requires the exact same path in `protected_parameters_unlocked`.

Versions 3–5 retain the static plan: every trial runs every listed `seeds`
entry and they do not accept `seed_strategy`, `ranking`, or `execution`.

## Automated execution

Version-6 tune sessions require a root-level `execution` object:

```json
{
  "enabled": true,
  "state_dir": "/absolute/path/to/tuning_execution",
  "run_command": [
    "conda", "run", "-n", "isaacsim-5.1", "python",
    "/absolute/path/to/rsl_rl_trial_adapter.py",
    "--contract={adapter_contract_path}",
    "--executor-run-id={run_id}",
    "--overrides-json={overrides_json}",
    "--result={result_path}",
    "--summary={summary_path}",
    "--effective-config={effective_config_path}",
    "--terminal={terminal_path}",
    "--log-path={log_path}"
  ],
  "gpu_index": 0,
  "require_idle_gpu": true,
  "max_retries_per_run": 0,
  "effective_config": {
    "enabled": true,
    "baseline_path": "/absolute/path/to/baseline_effective_config.json",
    "require_exact_override_match": true,
    "allow_baseline_bootstrap": true
  },
  "adapter": {
    "id": "rsl-rl",
    "parameter_cli_map": {
      "agent.algorithm.learning_rate": "agent.algorithm.learning_rate"
    },
    "runtime_config_paths": {
      "agent.seed": "seed",
      "agent.run_name": "run_id",
      "env.seed": "seed"
    },
    "summary_last": 100,
    "require_checkpoint": true
  },
  "resource_limits": {
    "campaign_timeout_minutes": 1440,
    "min_free_disk_gb": 50,
    "max_gpu_temperature_c": 85,
    "stop_grace_seconds": 30
  },
  "reproducibility": {
    "enabled": true,
    "capture_git_diff": true,
    "capture_gpu": true,
    "package_names": ["torch", "rsl-rl-lib", "PyYAML"],
    "input_paths": [
      "/absolute/path/to/critical_training_config.py"
    ]
  },
  "quality_rules": [
    {
      "id": "throughput-collapse",
      "metric": "steps_per_second",
      "op": "<",
      "value": 10000,
      "consecutive_windows": 3,
      "minimum_progress": 10,
      "action": "mark_suspect"
    }
  ],
  "nonfinite_action": "stop_trial"
}
```

Legacy custom adapters require the run, trial, stage, seed, output, and
effective-config placeholders. The built-in `rsl-rl` adapter instead requires
the attempt contract, exact run-ID token, log, terminal, result, summary,
effective-config, and override placeholders shown above. Its parameter map must exactly cover the
approved tuning paths. Runtime config paths may bind only to `seed` or
`run_id`; they are identity checks, not additional tuning permissions.
`state_dir` and the effective baseline must be absolute regular paths.
`require_idle_gpu` and exact override matching must be true. Retries are
bounded from zero through three. The optional one-time baseline bootstrap is
valid only for the unchanged baseline. Adapter sessions must also approve the
campaign timeout, minimum free disk, maximum GPU temperature, and SIGTERM grace
period.

`reproducibility` is optional for backward compatibility. When enabled, package
names must be explicit and every input path must be an absolute file path.
Missing packages, linked/missing inputs, Git evidence failure, or GPU identity
query failure blocks the attempt before training.

The RSL-RL adapter derives exact child argv from `training.command`, adds seed
and unique run name, and maps only approved Hydra paths. It writes the complete
dumped effective JSON configuration, live rolling summaries, finite metrics,
newest checkpoint evidence, and a terminal receipt. `minimum_progress` may be
added to any quality rule to define its finite-metric warm-up boundary. Read
`execution-and-robust-ranking.md` for state transitions, output schemas,
anomaly semantics, and commands.

## Final policy evaluation

Set `evaluation` before calling any checkpoint a final policy. Read
`policy-evaluation.md` for the complete result and visual-review workflow.

```json
{
  "enabled": true,
  "require_for_final_selection": true,
  "artifacts": [
    {
      "kind": "native",
      "required": true,
      "command": [
        "conda", "run", "-n", "isaacsim-5.1", "python",
        "scripts/reinforcement_learning/rsl_rl/evaluate_policy.py",
        "--task=EXACT_TRAINING_TASK",
        "--headless",
        "--device=cuda:{gpu_index}",
        "{require_idle_gpu_flag}",
        "--checkpoint={checkpoint_path}",
        "--checkpoint_sha256={checkpoint_sha256}",
        "--artifact_kind={artifact_kind}",
        "--artifact_path={artifact_path}",
        "--artifact_sha256={artifact_sha256}",
        "--candidate_id={candidate_id}",
        "--scenario_id={scenario_id}",
        "--scenario_overrides_json={scenario_overrides_json}",
        "--command_schedule_json={command_schedule_json}",
        "--duration_steps={duration_steps}",
        "--seed={seed}",
        "--run_id={run_id}",
        "--result_path={result_path}",
        "--video_path={video_path}"
      ]
    },
    {
      "kind": "onnx",
      "required": true,
      "command": [
        "conda", "run", "-n", "isaacsim-5.1", "python",
        "scripts/reinforcement_learning/rsl_rl/evaluate_policy.py",
        "--task=EXACT_TRAINING_TASK",
        "--headless",
        "--device=cuda:{gpu_index}",
        "{require_idle_gpu_flag}",
        "--checkpoint={checkpoint_path}",
        "--checkpoint_sha256={checkpoint_sha256}",
        "--artifact_kind={artifact_kind}",
        "--artifact_path={artifact_path}",
        "--artifact_sha256={artifact_sha256}",
        "--candidate_id={candidate_id}",
        "--scenario_id={scenario_id}",
        "--scenario_overrides_json={scenario_overrides_json}",
        "--command_schedule_json={command_schedule_json}",
        "--duration_steps={duration_steps}",
        "--seed={seed}",
        "--run_id={run_id}",
        "--result_path={result_path}",
        "--video_path={video_path}"
      ]
    }
  ],
  "scenarios": [
    {
      "id": "nominal-command-sweep",
      "category": "nominal",
      "required": true,
      "seeds": [42, 43],
      "duration_steps": 2000,
      "overrides": {},
      "command_schedule": [
        {"start_step": 0, "end_step": 199, "command": [0.0, 0.0, 0.0]},
        {"start_step": 200, "end_step": 699, "command": [0.5, 0.0, 0.0]},
        {"start_step": 700, "end_step": 899, "command": [0.0, 0.0, 0.0]},
        {"start_step": 900, "end_step": 1399, "command": [-0.4, 0.0, 0.0]},
        {"start_step": 1400, "end_step": 1599, "command": [0.0, 0.0, 0.0]},
        {"start_step": 1600, "end_step": 1999, "command": [0.0, 0.0, 0.4]}
      ],
      "video": true
    },
    {
      "id": "mass-friction-stress",
      "category": "dynamics",
      "required": true,
      "seeds": [42, 43],
      "duration_steps": 2000,
      "overrides": {
        "events.randomize_rigid_body_mass_base.params.mass_distribution_params": [-1.0, 3.0]
      },
      "command_schedule": [
        {"start_step": 0, "end_step": 1999, "command": [0.4, 0.0, 0.0]}
      ],
      "video": true
    }
  ],
  "gates": [
    {
      "metric": "termination_rate",
      "op": "<=",
      "value": 0.001,
      "aggregation": "max",
      "artifacts": ["*"],
      "scenarios": ["*"]
    },
    {
      "metric": "tracking_xy_rmse",
      "op": "<=",
      "value": 0.6,
      "aggregation": "mean",
      "artifacts": ["*"],
      "scenarios": ["*"]
    }
  ],
  "parity": {
    "required": true,
    "reference_artifact": "native",
    "max_abs_action_error": 0.00001
  },
  "visual_review": {
    "required": true,
    "minimum_reviewed_videos": 2,
    "require_notes": true
  },
  "output_dir": "/absolute/path/to/policy_evaluation",
  "gpu_index": 0,
  "require_idle_gpu": true,
  "max_concurrent_runs": 1,
  "run_timeout_minutes": 30,
  "allow_reject_candidate": true,
  "allow_retune_on_failure": false
}
```

The example thresholds and overrides are illustrative. Inspect the exact task,
units, event configuration, command ranges, and hardware limits before approval.
Use the training task rather than a Play-only task that disables disturbances.

If evaluation is absent, disabled, incomplete, or failed, ranking may report a
training candidate but must leave `final_selection` null. Enabling
`allow_retune_on_failure` does not add parameters; it only permits another
bounded trial using the existing authorized tuning paths.

## Qualified policy archive

Policy archival is a separate external-write authorization. It is available
only in a version-4, version-5, or version-6 tune session after final policy
evaluation. The evaluation
must mark both `jit` and `onnx` artifact entries as required.

Record the training source state in `training` when the run starts:

```json
{
  "source_git_commit": "0123456789abcdef0123456789abcdef01234567",
  "source_git_dirty": false
}
```

Use the full commit SHA. If the training source was dirty, preserve that fact;
do not substitute the later archive-time worktree state.

```json
{
  "enabled": true,
  "copy_after_qualification": true,
  "storage_root": "/home/young/liufengrong/policy_storage",
  "collection": "LW/leg_loco",
  "directory_naming": "local_timestamp_seconds",
  "timezone": "Asia/Shanghai",
  "required_artifacts": ["jit", "onnx"],
  "require_clean_git_worktree": true,
  "write_manifest": true,
  "description_notes": "rough terrain policy; supervised hardware test candidate",
  "git_action": "none"
}
```

Set the root-level `archive` field to this object. The collection must be an
existing safe relative path under the exact Git worktree. The archiver refuses
dirty storage, symlinks, changed hashes, duplicate pairs, missing formats,
failed evaluation, absent visual review, and destination collisions.

The operation creates `policy.pt`, `policy.onnx`, `策略说明.txt`, and
`archive_manifest.json` in one timestamped directory. It never overwrites an
existing directory and never commits or pushes. The description must say that
simulation qualification permits only supervised hardware testing and does not
mean `hardware_ready`.

## Hardware-feedback retuning

Physical feedback processing is a separate version-5 or version-6
authorization. Read
`hardware-feedback-retuning.md` before accepting a report or suggesting another
training cycle.

```json
{
  "enabled": true,
  "output_mode": "proposal_only",
  "output_dir": "/absolute/path/to/hardware_feedback_results",
  "require_policy_manifest": true,
  "verify_artifact_hashes": true,
  "stop_on_safety_event": true,
  "require_new_session_approval": true
}
```

Set the root-level `hardware_feedback` field to this object. Use
`proposal_only` for diagnosis without a parameter-choice draft. Use
`prepare_authorized_draft` only in tune mode; it can expose existing authorized
parameter options but cannot select them or launch trials. Both modes bind the
report to an exact archived JIT/ONNX pair and require a new approved session
before feedback-driven tuning.

## Runtime progress snapshot

After every check, persist:

- observation timestamp;
- profile ID and fingerprint;
- latest log progress value;
- latest TensorBoard scalar step and wall time;
- process identity and GPU evidence;
- restart count.

Pass the prior progress values to the next health check. A live process, active GPU, fresh checkpoint, or growing `.wandb` file does not replace monotonic progress evidence.

## Results format

Provide one training row per trial and seed:

```json
{
  "runs": [
    {
      "trial_id": "baseline",
      "seed": 42,
      "status": "completed",
      "metrics": {
        "mean_reward": 148.2,
        "error_vel_xy": 0.52,
        "illegal_contact": 0.05
      }
    }
  ]
}
```

Use `completed` only for a finite, fully evaluated run. The ranker rejects
missing seeds, metrics, non-finite values, failed constraints, and an
incomplete baseline. Version 6 also requires the complete confirmation seed
set, same-seed baseline pairs, per-seed hard constraints, uncertainty
statistics, and Pareto evidence.

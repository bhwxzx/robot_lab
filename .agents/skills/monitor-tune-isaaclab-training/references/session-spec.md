# Session authorization specification

Use JSON version `7` for optional multi-host Git-mailbox execution. Use version
`6` for automated staged trial execution on one host. Versions `6` and `7`
support explicit fixed-single-seed selection or robust multi-seed ranking and
physical deployment feedback. Version `5`
remains valid for feedback workflows without the executor, version `4` remains
valid for policy evaluation and archival without hardware feedback, and
version `3` remains accepted only for legacy sessions without `archive`.
Resolve the algorithm profile before approval, then validate with
`scripts/validate_session_spec.py`.

## Contents

- Monitor example
- Tune fields
- Bounded historical prior and adaptive rounds
- Automated execution
- Distributed Git mailbox
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
    "mode": "robust_multi_seed",
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

For the user-approved fixed-seed workflow, use this shape instead:

```json
{
  "seeds": [42],
  "seed_strategy": {
    "mode": "fixed_single_seed",
    "screening_seeds": [42],
    "confirmation_seeds": [42],
    "confirmation_top_k": 2,
    "final_authority": "supervised_hardware"
  },
  "ranking": {
    "require_paired_baseline": true,
    "constraint_scope": "each_seed",
    "minimum_final_training_seeds": 1,
    "pareto_front_required": true
  }
}
```

This mode creates no extra confirmation-seed runs. Its training result is
`single_seed_selected`, not robustly ranked. It requires enabled Play and
deployment-artifact evaluation plus the supervised hardware qualification
contract below; it makes no cross-seed or generalization claim.

Any allowed path matched by the selected profile's protected patterns also requires the exact same path in `protected_parameters_unlocked`.

Versions 3–5 retain the static plan: every trial runs every listed `seeds`
entry and they do not accept `seed_strategy`, `ranking`, or `execution`.

## Bounded historical prior and adaptive rounds

Versions 6 and 7 may add both `history_prior` and `adaptive_search` only with
`fixed_single_seed`. Read `history-informed-adaptive-search.md` for the exact
schema, commands, and evidence boundaries. The validator enforces a global
maximum of 6 selected historical runs, at most 100 retained points per
required metric per run, and at most 0.5 historical influence on first-round
candidates. `config_path_map` and `metric_key_map` must exactly cover the
already approved parameter and ranking contracts. This feature does not grant
new tuning paths or values.

The history contract also requires an explicit source policy and exact task,
profile, observation-contract, and reward-config context; a W&B progress key;
minimum final progress and retained points; and one stability metric with
approved tail deviation and slope gates. The adaptive contract requires one
objective metric, raw minimum improvement, patience, and minimum feasible
trial count for deterministic stopping.

The first plan requires a merged hash-bound prior. Each later plan appends one
deterministic round only after all existing fixed-seed trials have completed.
Existing trials and runs are immutable, historical combinations remain
excluded only when compatible for guidance, and the complete campaign remains
bounded by `tuning.max_trials`. A terminal adaptive decision appends no runs.

## Synchronous multi-fidelity training

Versions 6 and 7 may instead add `multi_fidelity` with
`fixed_single_seed`. It is mutually exclusive with `adaptive_search`:

```json
{
  "enabled": true,
  "metric": "mean_reward",
  "minimum_margin": 5.0,
  "minimum_rungs_before_performance_pruning": 2,
  "required_consecutive_underperformance": 2,
  "resume_same_worker": true,
  "rungs": [
    {"budget": 1000, "target_promoted_candidates": 5},
    {"budget": 3000, "target_promoted_candidates": 2},
    {"budget": 10000, "target_promoted_candidates": 0}
  ]
}
```

The metric must be an approved objective. Budgets strictly increase, the final
target is zero, and every pre-pruning rung protects all configured candidates.
The penultimate target must cover `confirmation_top_k`. Every allowed
parameter requires an explicit baseline. Version 7 requires `by_trial`
assignment, and every promoted trial resumes on its parent worker. Read
`multi-fidelity-training.md` for commands, decision rules, and limitations.

## Campaign controller

Versions 6 and 7 may add:

```json
{
  "campaign_controller": {
    "enabled": true,
    "mode": "shadow",
    "role": "single_host",
    "auto_launch_trials": true,
    "auto_advance_plans": true,
    "stop_before_evaluation": true,
    "worker_mailbox_repos": {}
  }
}
```

Version 6 requires `single_host` and an empty mapping. Version 7 requires
`role=distributed` and a mapping that exactly covers every approved worker ID
with its absolute local mailbox-clone path. Keep the same complete mapping on
all hosts so the session hash is identical; select the local identity at
runtime with `--worker-id`. `advance` requires approved `mode=execute`.
The controller may automate training transitions only and must stop before
evaluation. It also emits a session- and ranking-bound checkpoint inventory.
Read `campaign-controller.md`.

## Evaluation handoff controller

Versions 6 and 7 may separately authorize:

```json
{
  "evaluation_handoff": {
    "enabled": true,
    "mode": "shadow",
    "top_k": 1,
    "require_pareto": true,
    "checkpoint_seed": 42,
    "evaluation_worker_id": null,
    "artifact_path_templates": {
      "jit": "{rsl_rl_run_dir}/exported/policy.pt",
      "onnx": "{rsl_rl_run_dir}/exported/policy.onnx"
    },
    "auto_build_plan": true,
    "auto_execute_evaluation": true,
    "stop_before_visual_review": true
  }
}
```

This requires an executable Campaign Controller and executable policy
evaluation in the same tune session. Version 7 requires one exact worker ID;
version 6 requires null. Template keys must exactly cover every selected
non-Native evaluation artifact and resolve to already exported absolute files.
The controller selects only the approved Pareto Top-K at the exact checkpoint
seed, builds the candidate manifest and evaluation matrix, advances one cell at
a time, and stops at `awaiting_visual_review`. Read
`evaluation-handoff-controller.md`.

## Distributed Git mailbox

For two or more HTTPS-connected hosts, change the automated tune session to
version `7`, record a clean full source commit in `training`, and add a
root-level `distributed` contract. Read
`distributed-git-mailbox.md` for the complete schema and commands.

Version 7 retains the version-6 execution, seed, ranking, evaluation, archive,
and hardware-feedback contracts. With `assignment_mode=by_seed`, worker seed
assignments exactly partition `tuning.seeds`. With `fixed_single_seed`, require
`assignment_mode=by_trial` and assign the same single seed to every worker;
candidate trials are then divided deterministically and every exact
seed-and-overrides combination is published once. Set
`distributed.calibration.enabled=false` with `worker_ids=[]` for the default
parameter-search campaign. This deliberately leaves host effects uncontrolled,
so reports must not claim host invariance. Enable calibration only for a
separately approved host-effect diagnostic; then `worker_ids` must contain
every worker exactly once and the unchanged calibration baseline runs on each
host without entering candidate ranking. Git transports only immutable JSON
metadata, not model artifacts, videos, logs, or credentials.

For one shared policy-storage remote, set `archive.storage_root=null` and add
this object inside `archive`:

```json
{
  "distributed_lease": {
    "enabled": true,
    "storage_remote_url": "https://gitee.example/user/policy_storage.git",
    "storage_branch": "master",
    "authorized_worker_ids": ["pc-a", "pc-b"],
    "worker_storage_roots": {
      "pc-a": "/home/user-a/policy_storage",
      "pc-b": "/home/user-b/policy_storage"
    },
    "takeover_policy": "explicit_revoke_only"
  }
}
```

The worker roots must correspond exactly to the authorized distributed workers
and be absolute local clone paths. The coordinator may grant only one request
at a time. Do not automatically expire, steal, or reassign a lease. PT/ONNX
files stay out of the coordination repository; only request, grant, completion,
release, and explicit-revoke metadata is exchanged.

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
    "require_checkpoint": true,
    "multi_fidelity": {
      "budget_cli_path": "agent.max_iterations",
      "resume_cli_paths": {
        "enabled": "agent.resume",
        "load_run": "agent.load_run",
        "load_checkpoint": "agent.load_checkpoint"
      },
      "load_run_reference": "basename"
    }
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

Omit `execution.adapter.multi_fidelity` when the root contract is absent. When
present, its four managed Hydra paths must be unique, must not overlap an
allowed tuning or runtime path, and must not already appear in
`training.command`. `load_run_reference` is `basename` or `absolute`.

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
        "--executor_run_id", "{executor_run_id}",
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
        "--executor_run_id", "{executor_run_id}",
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
    "max_abs_action_error": 0.00001,
    "closed_loop_metrics": [
      {
        "metric": "tracking_xy_rmse",
        "max_abs_delta": 0.1,
        "aggregation": "max"
      },
      {
        "metric": "termination_rate",
        "max_abs_delta": 0.001,
        "aggregation": "max"
      }
    ]
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
  "allow_retune_on_failure": false,
  "execution": {
    "state_dir": "/absolute/path/to/policy_evaluation/.executor",
    "max_retries_per_run": 1,
    "stop_grace_seconds": 30,
    "min_free_disk_gb": 10,
    "max_gpu_temperature_c": 85,
    "minimum_video_bytes": 1024
  }
}
```

The example thresholds and overrides are illustrative. Inspect the exact task,
units, event configuration, command ranges, and hardware limits before approval.
Use the training task rather than a Play-only task that disables disturbances.
`evaluation.execution` is optional for legacy manual evaluation, but required
by `execute_evaluation_plan.py`. Its state directory must be inside
`evaluation.output_dir`. Automated commands must pass `{executor_run_id}` as a
standalone argv token so recovery can identify an orphan without relying on a
partial command match. Automated deployment-artifact evaluation also requires
at least one approved `closed_loop_metrics` delta against the Native run with
the same scenario and seed.

If evaluation is absent, disabled, incomplete, or failed, ranking may report a
training candidate but must leave `final_selection` null. Enabling
`allow_retune_on_failure` does not add parameters; it only permits another
bounded trial using the existing authorized tuning paths.

## Qualified policy archive

Policy archival is a separate external-write authorization. It is available
only in a version-4, version-5, version-6, or version-7 tune session after final policy
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

Physical feedback processing is a separate version-5, version-6, or version-7
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
  "require_new_session_approval": true,
  "qualification": {
    "enabled": true,
    "final_authority": "supervised_hardware",
    "minimum_total_tests": 4,
    "required_scenarios": ["standing", "start_stop", "low_speed", "turn"],
    "minimum_tests_per_scenario": 1,
    "require_high_evidence_confidence": true,
    "required_telemetry_channels": [
      "action",
      "control_timestamp",
      "imu_roll"
    ],
    "require_all_assessments_pass": true,
    "require_zero_safety_events": true,
    "status_label": "hardware_validated_for_test_envelope"
  }
}
```

Set the root-level `hardware_feedback` field to this object. Use
`proposal_only` for diagnosis without a parameter-choice draft. Use
`prepare_authorized_draft` only in tune mode; it can expose existing authorized
parameter options but cannot select them or launch trials. Both modes bind the
report to an exact archived JIT/ONNX pair and require a new approved session
before feedback-driven tuning.

`qualification` is optional for feedback diagnosis but mandatory when
`fixed_single_seed` names supervised hardware as final authority. The required
scenario list must contain at least three unique scenarios, and the minimum
test count must cover each scenario at the approved repetition count. The
qualification report can state only
`hardware_validated_for_test_envelope`; it always keeps
`hardware_ready=false` and `generalization_claim=false`.

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

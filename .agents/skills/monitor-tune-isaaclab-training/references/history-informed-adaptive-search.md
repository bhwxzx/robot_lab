# Bounded history-informed adaptive search

## Contents

- Hard bounds and evidence
- Compatibility and quality
- Session fields
- Candidate provenance and stopping
- Single-host workflow
- Two-host Git-mailbox workflow

Use this optional workflow only in an approved version-6 or version-7 tune
session with `fixed_single_seed`. It uses completed local W&B records to choose
part of the first round, then chooses later rounds from completed trial
metrics. It does not use the W&B API, sync data, authorize new parameter paths,
expand values, edit source, or decide physical readiness.

## Hard bounds and evidence

- `max_selected_runs` is a global post-merge limit from 1 through 6, not a
  per-computer final allowance.
- `lookback_days` is 1 through 365; use 30 unless the approved session says
  otherwise.
- Read no more than twice the run limit as recent candidate directories on
  each host and retain at most `max_points_per_run`, capped at 100, for each
  required metric.
- Require the exact W&B project, every approved tuning parameter, every
  objective/constraint metric, finite values, an authorized grid combination,
  and a terminal exit record. Failed runs are excluded unless
  `include_failed_runs=true`.
- Apply the approved `exact`, `compatible`, or `advisory` source policy. Never
  silently treat a source mismatch as equivalent.
- Historical combinations are excluded from new trials. History may influence
  at most `max_first_round_fraction`, which cannot exceed 0.5. The remaining
  first-round candidates are deterministic diverse exploration.
- Later rounds use only one completed result for every existing fixed-seed
  trial. Constraints are applied before objectives. Every expansion is
  append-only, hash-bound, bounded by `max_trials`, and reproducible from its
  embedded prior and result snapshots.

## Compatibility and quality

Bind every run to the approved task ID, algorithm profile fingerprint,
observation-contract hash, and reward-configuration hash:

- `exact`: require the approved source commit.
- `compatible`: accept the exact source commit or all four exact context
  values.
- `advisory`: retain mismatched metadata for reporting, but do not use it as a
  sampling anchor or exclude its parameter combination from a fresh run.

Require a successful terminal record unless failed runs are explicitly
retained. Also require the approved final progress, minimum points per metric,
and stability-metric tail standard deviation and absolute slope. Record the
tail count, mean, population standard deviation, slope, and final progress.
Reject missing, non-finite, unstable, or under-trained evidence.

## Session fields

Add both root-level objects:

```json
{
  "history_prior": {
    "enabled": true,
    "source": "local_wandb",
    "wandb_project": "approved-project",
    "lookback_days": 30,
    "max_selected_runs": 6,
    "max_points_per_run": 100,
    "include_failed_runs": false,
    "max_first_round_fraction": 0.5,
    "explicit_run_ids": [],
    "config_path_map": {
      "env.rewards.tracking.weight": "env.rewards.tracking.weight"
    },
    "metric_key_map": {
      "mean_reward": "mean_reward",
      "illegal_contact": "illegal_contact"
    },
    "worker_roots": {
      "local": "/absolute/path/to/wandb"
    },
    "compatibility": {
      "source_policy": "compatible",
      "expected_context": {
        "task_id": "approved-task-id",
        "profile_fingerprint": "approved-profile-fingerprint",
        "observation_contract_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "reward_config_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
      },
      "context_path_map": {
        "task_id": "context.task_id",
        "profile_fingerprint": "context.profile_fingerprint",
        "observation_contract_sha256": "context.observation_contract_sha256",
        "reward_config_sha256": "context.reward_config_sha256"
      }
    },
    "quality_gates": {
      "progress_key": "_step",
      "minimum_final_progress": 1000,
      "minimum_points_per_metric": 20,
      "stability": {
        "metric": "mean_reward",
        "max_standard_deviation": 5.0,
        "max_abs_slope": 0.1
      }
    }
  },
  "adaptive_search": {
    "enabled": true,
    "max_rounds": 3,
    "trials_per_round": 2,
    "exploration_fraction": 0.5,
    "stop_policy": {
      "enabled": true,
      "metric": "mean_reward",
      "minimum_improvement": 1.0,
      "patience_rounds": 2,
      "minimum_feasible_trials": 1
    }
  }
}
```

For version 7, `worker_roots` must contain every configured worker ID instead
of `local`. The config map must cover exactly `tuning.allowed_parameters`; the
metric map must cover exactly all objective and constraint metrics. Every
allowed parameter needs an explicit `baseline`. Round capacity must cover the
non-baseline trial budget.

The example thresholds are illustrative. Approve task-specific progress,
metric scale, variability, slope, and improvement thresholds before use.

## Candidate provenance and stopping

Every selected trial records whether it is historical exploitation, current
trial exploitation, or diverse exploration; the anchor ID, normalized grid
distance, override hash, and explicit non-duplicate assertions are immutable
plan evidence.

After each completed round, compare the approved stop objective for the newest
round against the best earlier constraint-satisfying trial. Stop without new
runs when there are no feasible trials, too few feasible trials, no sufficient
improvement for the approved patience, no remaining authorized grid point, no
budget, or no remaining round. The decision records the raw metric values,
improvement, feasible count, remaining candidates, action, and reason. It never
claims final policy quality.

## Single-host workflow

Create a new absolute index file, merge it to a new absolute prior, and build
the first plan:

```bash
python scripts/index_local_wandb_history.py SESSION.json \
  --worker-id local --output /absolute/path/HISTORY_INDEX.json

python scripts/merge_historical_priors.py SESSION.json \
  /absolute/path/HISTORY_INDEX.json \
  --output /absolute/path/HISTORY_PRIOR.json

python scripts/build_trial_plan.py SESSION.json \
  --history-prior /absolute/path/HISTORY_PRIOR.json \
  --output /absolute/path/PLAN_ROUND_1.json
```

After every current run completes, build a new plan file and adopt it:

```bash
python scripts/build_adaptive_round.py SESSION.json PLAN_ROUND_1.json \
  COMPLETED_RESULTS.json --output /absolute/path/PLAN_ROUND_2.json

python scripts/execute_trial_plan.py SESSION.json PLAN_ROUND_2.json \
  --action adopt-plan
```

`adopt-plan` refuses an active run, incomplete old runs, changed old entries,
or more than one new decision. A stop decision adds no run and leaves executor
state at `adaptive_stopped`.

## Two-host Git-mailbox workflow

The coordinator first publishes only the history collection manifest:

```bash
python scripts/git_mailbox.py history-initialize \
  --repo COORDINATOR_CLONE --session SESSION_V7.json
```

Each worker builds its local index and publishes the metadata from its own
branch:

```bash
python scripts/git_mailbox.py history-publish \
  --repo WORKER_CLONE --session SESSION_V7.json \
  --worker pc-b --index-json /absolute/path/PC_B_HISTORY_INDEX.json
```

After every configured worker publishes, the coordinator merges the global
bounded prior:

```bash
python scripts/git_mailbox.py history-collect \
  --repo COORDINATOR_CLONE --session SESSION_V7.json \
  --output /absolute/path/HISTORY_PRIOR.json
```

Build and publish round 1 normally. After collecting all current results,
build the expanded plan and publish only its newly appended jobs:

```bash
python scripts/git_mailbox.py publish-adaptive-round \
  --repo COORDINATOR_CLONE --session SESSION_V7.json \
  --previous-plan PLAN_ROUND_1.json \
  --expanded-plan PLAN_ROUND_2.json
```

The mailbox rejects missing worker indexes, invalid hashes, incomplete or
invalid results, changed previous plan entries, duplicate run IDs, or a plan
whose embedded result snapshot differs from collected immutable results.
When the decision is `stop`, it publishes one immutable
`adaptive_search_stopped` manifest and zero jobs.

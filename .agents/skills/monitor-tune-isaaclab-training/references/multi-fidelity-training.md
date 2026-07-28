# Synchronous multi-fidelity training

Use this workflow only with an approved version-6 or version-7
`fixed_single_seed` tune session. It reduces wasted training by advancing
promising configurations through increasing budgets while preserving a
conservative evidence trail.

## Contents

- Safety model
- Session contract
- RSL-RL adapter contract
- Single-host workflow
- Git-mailbox workflow
- Final ranking and limits

## Safety model

This is a synchronous rung protocol, not asynchronous successive halving.
Every active baseline and candidate must finish the current rung before the
next decision is created.

- Apply declared hard constraints immediately on every rung.
- Never performance-prune before the approved minimum rung.
- Require the approved number of consecutive underperforming rungs before
  ordinary performance elimination.
- Keep the unchanged baseline in every nonterminal rung.
- If protected candidates exceed the target, exceed the target rather than
  weakening the safety rule.
- Resume only from the exact parent checkpoint, SHA-256, step, and RSL-RL run
  directory recorded in the immutable rung result.
- Keep one seed and one trial identity across all rungs. A resumed rung is not
  a new independent seed.

The final rung promotes no further candidates. It creates an immutable
`complete` or `stop` decision and zero new jobs.

## Session contract

Add this root-level object:

```json
{
  "multi_fidelity": {
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
}
```

`metric` must be an approved objective. Budgets strictly increase. Candidate
targets are positive and non-increasing before the final zero target. Before
performance pruning is allowed, each target must protect all configured
candidates. The penultimate target must be at least `confirmation_top_k`.

This mode requires a baseline for every allowed parameter and cannot share a
session with `adaptive_search`. Version 7 also requires
`distributed.assignment_mode=by_trial`.

## RSL-RL adapter contract

The included executable adapter requires this additional
`execution.adapter.multi_fidelity` object:

```json
{
  "budget_cli_path": "agent.max_iterations",
  "resume_cli_paths": {
    "enabled": "agent.resume",
    "load_run": "agent.load_run",
    "load_checkpoint": "agent.load_checkpoint"
  },
  "load_run_reference": "basename"
}
```

Use only paths verified for the current backend and runner. The managed budget
and resume paths cannot overlap an allowed parameter or runtime identity path
and cannot already appear in the base command. `require_checkpoint` must be
true.

The RSL-RL adapter verifies that the parent checkpoint is a regular,
non-symlink file inside its recorded run directory, that its current SHA-256
matches the parent result, and that the child checkpoint step advances.
Other algorithms may use the generic plan contract only after their own
budget/resume mappings, checkpoint completeness, and RNG restoration behavior
are reviewed and implemented. Never infer resume flags from an algorithm name.

## Single-host workflow

Build and execute rung 1 with the normal commands:

```bash
python scripts/build_trial_plan.py SESSION.json --output PLAN_R1.json
python scripts/execute_trial_plan.py \
  SESSION.json PLAN_R1.json --action initialize
```

Use `launch-next` and `reconcile` until every current-rung run has one valid
completed result. Then create exactly one deterministic expansion:

```bash
python scripts/build_multifidelity_rung.py \
  SESSION.json PLAN_R1.json RESULTS_R1.json --output PLAN_R2.json
python scripts/execute_trial_plan.py \
  SESSION.json PLAN_R2.json --action adopt-plan
```

Repeat only after the new current rung is complete. `adopt-plan` accepts an
append-only rung and its one hash-bound decision; it rejects changed trials,
earlier runs, or prior decisions. Do not manually edit a plan or promotion.

## Git-mailbox workflow

Publish rung 1 through the normal `git_mailbox.py publish` command. Each worker
must include its current-rung checkpoint path and SHA-256 in both the result
and artifact manifest. The coordination repository transports metadata only;
it never transports checkpoints.

After collecting a complete rung, build the expanded plan on the coordinator,
then publish it:

```bash
python scripts/git_mailbox.py publish-multifidelity-rung \
  --repo /absolute/path/to/coordinator-mailbox \
  --session SESSION_V7.json \
  --previous-plan PLAN_R1.json \
  --expanded-plan PLAN_R2.json
```

The command verifies the immutable result snapshot, checkpoint artifact
entries, decision hash, and append-only jobs. A promoted trial remains on the
worker that produced its parent checkpoint. If that worker cannot access or
verify the parent file, the rung remains blocked; do not reassign it or claim
that Git transferred the model. A terminal decision publishes zero jobs.

## Final ranking and limits

After a completed final rung, convert its immutable input snapshot to the
ordinary ranking schema:

```bash
python scripts/finalize_multifidelity_results.py \
  SESSION.json FINAL_PLAN.json --output FINAL_RESULTS.json
python scripts/rank_trials.py SESSION.json FINAL_RESULTS.json
```

The final result remains `single_seed_selected` with
`generalization_claim=false`. Multi-fidelity evidence saves compute; it does
not establish multi-seed robustness, simulation-to-real transfer, or hardware
readiness. Continue through Play, Native/JIT/ONNX artifact checks, visual
motion review, and the approved supervised physical qualification matrix
before promotion or policy-storage archival.

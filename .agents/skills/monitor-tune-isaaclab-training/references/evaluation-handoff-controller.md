# Evaluation handoff controller

Use this controller only after the Campaign Controller has produced immutable
training ranking and checkpoint-inventory files. It automates the bounded
transition from training evidence to the existing closed-loop policy-evaluation
executor. It does not visually approve motion, archive a policy, or start a
physical test.

## Contents

- [Session permission](#session-permission)
- [Inputs and commands](#inputs-and-commands)
- [Distributed boundary](#distributed-boundary)
- [Stopping boundary](#stopping-boundary)

## Session permission

Add `evaluation_handoff` to an approved version-6-or-7 tune session:

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

`top_k` cannot exceed the approved confirmation Top-K. `checkpoint_seed` must
be an approved confirmation seed. Version 6 requires
`evaluation_worker_id=null`. Version 7 requires one exact distributed worker
ID; pass that same identity at runtime.

The artifact-template keys must exactly cover selected non-Native evaluation
artifacts. Supported fields are `candidate_id`, `trial_id`, `seed`,
`checkpoint_path`, `checkpoint_dir`, and `rsl_rl_run_dir`. Templates locate
already exported artifacts. The controller never runs an exporter or guesses a
path. A missing, linked, moved, or changed artifact blocks plan creation.

## Inputs and commands

The Campaign Controller writes:

- `training_ranking.json`;
- `checkpoint_inventory.json`, bound to the session and ranking hashes.

Inspect without writes:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/evaluation_handoff_controller.py \
  SESSION.json TRAINING_RANKING.json CHECKPOINT_INVENTORY.json \
  --action status
```

For version 7, add `--worker-id EXACT_APPROVED_WORKER`.

After separately approving `evaluation_handoff.mode=execute`, advance one
transition:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/evaluation_handoff_controller.py \
  SESSION.json TRAINING_RANKING.json CHECKPOINT_INVENTORY.json \
  --action advance
```

The transition sequence is:

1. initialize the hash-bound handoff state;
2. select the approved Pareto Top-K and exact checkpoint seed;
3. verify local checkpoint and deployment-artifact hashes;
4. write immutable candidate manifest and evaluation plan;
5. initialize the existing evaluation executor;
6. reconcile or launch one matrix cell per invocation;
7. stop at `awaiting_visual_review`.

The handoff state and hash-chained journal live under
`evaluation.output_dir/.handoff`. Evaluation execution continues to use
`evaluation.execution.state_dir` and the shared training/evaluation GPU lock.
Repeated terminal invocations are idempotent.

## Distributed boundary

Git mailbox transport remains metadata-only. It does not copy checkpoints,
JIT, ONNX, or video files. Every selected Top-K checkpoint and deployment
artifact must already be locally readable on the single approved evaluation
worker with matching hashes. If ranking selects a policy owned by another
worker, stop and arrange a separately authorized artifact transfer or approve
a new evaluation-worker contract. Never silently evaluate a different
candidate.

## Stopping boundary

`awaiting_visual_review` means only that all automatic matrix cells completed.
Inspect required videos and review windows, record notes, consolidate results,
and run `validate_policy_evaluation.py`. The controller never:

- infer visual acceptance from metrics;
- mark a policy simulation-qualified;
- create a policy-storage archive;
- commit or push Git state;
- initiate or qualify a physical deployment.

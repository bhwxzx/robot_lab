# Distributed Git mailbox

Use this workflow only with an approved version-7 tune session when direct
machine-to-machine control is unavailable but every host can fetch and push the
same private Git remote over HTTPS.

## Trust and transport boundaries

- Use a dedicated coordination repository. Do not use the `robot_lab` source
  repository as the mailbox.
- Keep credentials in the operating-system or Git credential helper. Reject
  credentials, query strings, and fragments in `distributed.remote_url`.
- Store only finite JSON metadata. Never commit checkpoints, ONNX/JIT files,
  videos, full logs, credentials, or environment secrets to the mailbox.
- Publish each complete trial-plan JSON under its content hash so a worker can
  validate the exact plan named by its job without copying it out of band.
- Treat the coordinator branch as an immutable job/control inbox. Give each
  worker its own result branch; never let two workers push the same branch.
- Bind every job to the approved session hash, trial-plan hash, source commit,
  worker, seed, stage, run ID, and branch.
- Require a clean worker source repository at the exact approved commit before
  claim and before result publication. Refuse automatic pull, checkout, stash,
  reset, or cleanup of a dirty source worktree.

## Authorization shape

Add `distributed` to a version-7 tune session:

```json
{
  "enabled": true,
  "transport": "git_mailbox",
  "campaign_id": "lw-leg-ppo-round-01",
  "remote_url": "https://git.example.edu/user/robot-tuning-mailbox.git",
  "coordinator_id": "pc-a",
  "coordinator_branch": "tune/lw-leg-round-01/coordinator",
  "poll_interval_seconds": 600,
  "remote_state_unknown_after_seconds": 1800,
  "artifact_policy": "metadata_only",
  "assignment_mode": "by_trial",
  "workers": [
    {
      "id": "pc-a",
      "branch": "tune/lw-leg-round-01/worker-pc-a",
      "assigned_seeds": [42],
      "source_repo": "/absolute/path/to/robot_lab",
      "state_dir": "/absolute/path/to/tuning/pc-a",
      "effective_config_baseline_path": "/absolute/path/to/tuning/pc-a/effective-config.json",
      "gpu_index": 0,
      "max_active_jobs": 1
    },
    {
      "id": "pc-b",
      "branch": "tune/lw-leg-round-01/worker-pc-b",
      "assigned_seeds": [42],
      "source_repo": "/absolute/path/to/robot_lab",
      "state_dir": "/absolute/path/to/tuning/pc-b",
      "effective_config_baseline_path": "/absolute/path/to/tuning/pc-b/effective-config.json",
      "gpu_index": 0,
      "max_active_jobs": 1
    }
  ],
  "calibration": {
    "enabled": false,
    "seed": 42,
    "worker_ids": []
  }
}
```

Partition all confirmation seeds exactly once across workers. Keep every
candidate and its unchanged baseline for a seed on the same worker. With
calibration disabled, publish every exact seed-and-overrides combination once
across the campaign. Record `host_effects_uncontrolled=true` in the campaign
notes or report and do not claim host-invariant results.

For `fixed_single_seed`, instead set `assignment_mode` to `by_trial` and give
every worker the same one-element `assigned_seeds` array. Candidate
`trial-NNN` jobs are assigned round-robin in configured worker order; the
unchanged plan baseline goes to the coordinator exactly once. This splits
different reward weights or parameter combinations, not random seeds, and
does not duplicate an exact trial merely to compare machines.

To test host effects later, approve a separate diagnostic and set
`calibration.enabled=true` with every configured worker ID exactly once. The
mailbox then adds the same unchanged calibration baseline on every host. Do not
include calibration runs in candidate ranking or consume the ordinary
reward-weight search budget with them. Treat materially different calibration
results as a host-comparability blocker before comparing trials from different
machines.

Version 7 also requires `training.source_git_commit` to be a full lowercase
commit and `training.source_git_dirty=false`. The source paths may differ by
worker, but the source commit, profile, parameter authorization, plan, and
ranking contract may not.

Before `status` or `claim`, update each source clone explicitly with ordinary
read-only `git fetch`, inspect local changes, and move a clean execution
worktree to the exact approved commit. Never let the mailbox script pull,
checkout, reset, stash, or overwrite the user's source tree. If the worktree is
dirty or its HEAD differs, the worker remains blocked until the user resolves
it.

## Bounded history and adaptive rounds

When the approved fixed-seed session enables bounded local-W&B priors, follow
`history-informed-adaptive-search.md`. Before publishing the first trial plan,
the coordinator runs `history-initialize`; every worker scans only its
authorized local root and runs `history-publish`; the coordinator requires all
worker indexes and runs `history-collect`. The merged run cap remains global
across both computers. The mailbox carries only finite, hash-bound summary
metadata and never copies W&B files or contacts W&B cloud services.
Each index exposes its compatibility and quality evidence. Only guidance-
eligible runs can steer candidate selection across hosts.

After every current trial has a valid terminal result, build one deterministic
append-only plan expansion. `publish-adaptive-round` verifies the previous
plan, immutable collected results, embedded result hash, budget, unique run
IDs, and exact newly appended jobs before publication. Workers never invent a
new round or change earlier jobs. A deterministic stop decision publishes one
immutable stop manifest with zero jobs.

## Synchronous multi-fidelity rungs

When the approved session contains `multi_fidelity`, follow
`multi-fidelity-training.md`. Publish rung 1 normally. Every result must bind
its current checkpoint path and SHA-256, and its artifact manifest must contain
the same checkpoint entry. Git transports this evidence, not the model file.

After all current-rung results are valid, build one deterministic plan
expansion and publish it:

```bash
python scripts/git_mailbox.py publish-multifidelity-rung \
  --repo /absolute/path/to/coordinator-mailbox \
  --session SESSION_V7.json \
  --previous-plan PREVIOUS_PLAN.json \
  --expanded-plan EXPANDED_PLAN.json
```

The publisher verifies the rung barrier, immutable result snapshot, decision,
checkpoint manifest, and append-only jobs. Every promoted trial remains on the
worker that produced its parent checkpoint. A terminal `complete` or `stop`
decision creates zero jobs. Never reassign a missing parent checkpoint or
commit it to the mailbox.

## Coordinator workflow

Use a clean clone of the dedicated coordination repository:

```bash
python scripts/git_mailbox.py publish \
  --repo /absolute/path/to/coordination-clone \
  --session SESSION_V7.json \
  --plan TRIAL_PLAN.json
```

Publication creates immutable campaign and job JSON files on the approved
coordinator branch. Repeating an identical publication is idempotent. A
different document at an existing path is a hard collision. The same
publication also stores the complete JSON plan snapshot at a content-addressed
metadata path; `prepare-job` returns that plan after validating its hash.

When the approved session enables `campaign_controller`, read
`campaign-controller.md`. All hosts keep one identical controller contract
containing the complete worker-to-mailbox-path mapping, then choose their local
identity with `--worker-id`. Git still carries no model artifact.

After workers publish outputs:

```bash
python scripts/git_mailbox.py collect \
  --repo /absolute/path/to/coordination-clone \
  --session SESSION_V7.json \
  --output /absolute/path/to/COLLECTED_RESULTS.json
```

Accept only envelopes whose worker identity, job hash, result hash, artifact
manifest hash, session hash, and source commit match. Reject invalid envelopes
without silently dropping their error from the collection report.

After every screening job has a valid terminal result, publish the deterministic
top-k selection or confirmation jobs:

```bash
python scripts/git_mailbox.py publish-confirmation \
  --repo /absolute/path/to/coordination-clone \
  --session SESSION_V7.json \
  --plan TRIAL_PLAN.json
```

This command collects the worker branches again, blocks on invalid or incomplete
screening evidence, applies the approved screening constraints/objectives and
top-k rule, and records an immutable hash-bound selection. Robust mode publishes
the baseline plus selected candidates on remaining confirmation seeds. Fixed
mode returns `single_seed_selection_published` with zero extra jobs. Do not
manufacture confirmation runs or manually edit selection JSON.

Publish a cancellation request with:

```bash
python scripts/git_mailbox.py cancel \
  --repo /absolute/path/to/coordination-clone \
  --session SESSION_V7.json \
  --worker pc-b \
  --job-id EXACT_JOB_ID \
  --reason "approved bounded cancellation reason"
```

A cancel document is a request, not evidence that the remote process stopped.
The worker must reconcile the local executor, verify its exact PID/process
group, apply the existing stop rules, and then publish a terminal result.

## Worker workflow

Fetch and inspect only assigned jobs:

```bash
python scripts/git_mailbox.py status \
  --repo /absolute/path/to/worker-mailbox-clone \
  --session SESSION_V7.json \
  --worker pc-b
```

Before starting any local executor, publish the claim:

```bash
python scripts/git_mailbox.py claim \
  --repo /absolute/path/to/worker-mailbox-clone \
  --session SESSION_V7.json \
  --worker pc-b \
  --job-id EXACT_JOB_ID \
  --attempt 1
```

Start nothing unless the command returns `claim_published`. Then execute exactly
one job through the existing local `execute_trial_plan.py` safety path. First
materialize the exact remotely visible claim:

```bash
python scripts/git_mailbox.py prepare-job \
  --repo /absolute/path/to/worker-mailbox-clone \
  --session SESSION_V7.json \
  --worker pc-b \
  --job-id EXACT_JOB_ID \
  --attempt 1 \
  --output /absolute/path/to/PREPARED_JOB.json
```

Then initialize, launch, and reconcile only that job:

```bash
python scripts/execute_trial_plan.py \
  SESSION_V7.json TRIAL_PLAN.json \
  --distributed-job /absolute/path/to/PREPARED_JOB.json \
  --worker-id pc-b \
  --action initialize

python scripts/execute_trial_plan.py \
  SESSION_V7.json TRIAL_PLAN.json \
  --distributed-job /absolute/path/to/PREPARED_JOB.json \
  --worker-id pc-b \
  --action launch-next

python scripts/execute_trial_plan.py \
  SESSION_V7.json TRIAL_PLAN.json \
  --distributed-job /absolute/path/to/PREPARED_JOB.json \
  --worker-id pc-b \
  --action reconcile
```

The distributed mode revalidates the full approved plan, job/session/plan
hashes, exact worker, source commit and cleanliness, and exact run membership.
It isolates state under `workers[].state_dir/jobs/<job_id>`, uses the worker
GPU and effective-config baseline, and retains the existing GPU lock, process
identity, reproducibility, timeout, retry, quality, and effective-config gates.
Never derive shell text from mailbox JSON; pass structured values into the
already-approved argv contract.

Publish progress only at approved milestones, normally no more often than the
session poll interval:

```bash
python scripts/git_mailbox.py progress \
  --repo /absolute/path/to/worker-mailbox-clone \
  --session SESSION_V7.json \
  --worker pc-b \
  --job-id EXACT_JOB_ID \
  --attempt 1 \
  --sequence 1 \
  --progress-json PROGRESS.json
```

Publish a terminal result and metadata-only artifact manifest:

```bash
python scripts/git_mailbox.py result \
  --repo /absolute/path/to/worker-mailbox-clone \
  --session SESSION_V7.json \
  --worker pc-b \
  --job-id EXACT_JOB_ID \
  --attempt 1 \
  --result-json RESULT.json \
  --artifact-manifest ARTIFACT_MANIFEST.json
```

The artifact manifest contains only `kind`, absolute local `path`, lowercase
SHA-256, and `size_bytes`. Transfer a qualified top-k artifact separately under
the existing evaluation and archive authorization.

## Shared policy-storage lease

When both computers use separate clones of one `policy_storage` remote, enable
`archive.distributed_lease` in the version-7 session. Configure the exact HTTPS
remote, branch, authorized workers, and each worker's absolute local storage
root. The mailbox still carries JSON metadata only; JIT/ONNX artifacts never
pass through the coordination repository.

The worker builds and publishes a hash-bound archive request only after its
storage clone is clean and exactly at the remote head. The coordinator reviews
all requests and grants exactly one active lease:

```bash
python scripts/git_mailbox.py archive-status \
  --repo /absolute/path/to/coordinator-mailbox \
  --session SESSION_V7.json

python scripts/git_mailbox.py archive-grant \
  --repo /absolute/path/to/coordinator-mailbox \
  --session SESSION_V7.json \
  --worker pc-b \
  --request-id EXACT_REQUEST_ID
```

The selected worker materializes that immutable grant, runs the qualified
policy archiver, and leaves the new policy directory uncommitted. Staging,
committing, and pushing `policy_storage` require separate explicit approval.
After that approved push, the worker publishes completion and the coordinator
rechecks the shared remote branch before releasing the lease.

See [policy-archive.md](policy-archive.md#shared-repository-lease) for the full
request, grant, archive, completion, release, and explicit-revoke commands.
Never grant a second lease because a worker is merely silent or old. Only an
explicitly approved `archive-revoke` can recover an incomplete lease; a lease
with completion evidence must be released.

## Failure semantics

- If claim push fails, do not start training. Fix HTTPS access and repeat the
  same claim; immutable events make this retry idempotent.
- If connectivity fails after a published claim, keep the local watchdog
  active and mark remote state unknown. Do not reassign the job solely because
  Git progress is stale.
- Never use Git timestamps as training-health evidence. Use monotonic log or
  TensorBoard progress locally.
- Reject duplicate run IDs, duplicate worker branches, seed overlap in
  `by_seed`, unequal worker seed declarations in `by_trial`, mismatched source
  commits, dirty source trees, changed job
  bytes, non-finite JSON, and payloads above the metadata limit.
- Include campaign, job, worker, attempt, stage, and seed in external run IDs
  such as W&B identities. Never attach two processes to the same run ID.

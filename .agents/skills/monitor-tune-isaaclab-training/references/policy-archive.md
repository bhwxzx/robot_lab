# Qualified policy archival

Archive a policy only after bounded tuning and closed-loop evaluation produce a
simulation-qualified final selection. Archival prepares a candidate for
supervised hardware testing; it does not approve physical deployment.

## Contents

- Storage convention
- Authorization and prerequisites
- Inspection
- Archive evidence
- Shared-repository lease
- Atomic write behavior
- Failure rules
- Git and hardware boundaries

## Storage convention

The current policy storage uses:

```text
policy_storage/
└── LW/
    └── leg_loco/
        └── YYYY-MM-DD-HH-MM-SS/
            ├── policy.pt
            ├── policy.onnx
            ├── 策略说明.txt
            └── archive_manifest.json
```

Treat `policy.pt` as the evaluated JIT deployment artifact, not the native
training checkpoint. Keep the checkpoint in the training run. Use an approved
existing collection such as `LW/leg_loco`; never infer a collection by writing
to the nearest-looking directory.

## Authorization and prerequisites

Require all of the following:

1. a validated version-4, version-5, version-6, or version-7 tune session with
   `archive.enabled=true`;
2. `copy_after_qualification=true`;
3. an exact absolute storage root and safe relative collection;
4. JIT and ONNX listed as required evaluation artifacts;
5. final ranking status `simulation_qualified_hardware_candidate`;
6. complete metric, parity, video, and visual-review evidence;
7. unchanged source artifact SHA-256 values;
8. an exact clean Git worktree at the storage root;
9. the full training-source Git commit and its recorded dirty flag;
10. `cleanup.remove_created_temp_files=true`;
11. `git_action=none`.

Reject a version-3 session, monitor mode, a generic algorithm profile, a
training-only ranking, an optional or unevaluated artifact, or any
`hardware_ready=true` claim.

## Inspection

Inspect before archiving:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/inspect_policy_storage.py \
  /home/young/liufengrong/policy_storage --hash-artifacts
```

Review:

- exact Git root, branch, base commit, and clean state;
- available collections;
- prior JIT/ONNX pairs and descriptions;
- missing descriptions or formats;
- duplicate artifact hashes;
- symlinked files.

The inspector is read-only. Do not treat a directory name or recent commit as
proof that a stored policy passed current evaluation gates.

## Archive evidence

Run the archiver with raw training and evaluation evidence:

```bash
conda run -n isaacsim-5.1 python \
  .agents/skills/monitor-tune-isaaclab-training/scripts/archive_policy_candidate.py \
  SESSION.json TRAINING_RESULTS.json EVALUATION_PLAN.json \
  EVALUATION_RESULTS.json --output /absolute/ARCHIVE_RECEIPT.json
```

The archiver recomputes policy evaluation and final ranking rather than
trusting a hand-edited final-selection document. It also rebuilds the
authorized trial plan and rejects a selected trial that was never authorized.

`策略说明.txt` records:

- task, algorithm, runner, profile, run, and candidate;
- recorded training-source commit and dirty state;
- observation-history contract;
- selected tuning overrides and aggregate metrics;
- required scenarios and visual-review notes;
- JIT and ONNX SHA-256 values;
- user-approved description notes;
- simulation qualification and the explicit real-robot limitation.

`archive_manifest.json` preserves the same information as structured JSON,
plus gate definitions, parity expectations, the recorded training source,
archive-time source observation, and the storage base commit. When the training
source was dirty, state that the commit alone cannot fully reproduce the run.

## Shared-repository lease

Use this workflow only for a version-7 session whose `archive` contract enables
`distributed_lease`. Each worker keeps a separate local clone of the same
policy-storage remote. The coordination repository carries metadata only.

On the worker that holds the final evaluated artifacts, build the request:

```bash
conda run -n isaacsim-5.1 python \
  scripts/archive_policy_candidate.py \
  SESSION.json TRAINING_RESULTS.json EVALUATION_PLAN.json \
  EVALUATION_RESULTS.json --prepare-lease-request \
  --worker-id pc-b --output /absolute/ARCHIVE_REQUEST.json

conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-request \
  --repo /absolute/worker-mailbox --session SESSION.json \
  --worker pc-b --request-json /absolute/ARCHIVE_REQUEST.json
```

The request is accepted only when the worker's policy-storage clone is clean,
on the approved branch, and exactly equal to its remote head. It binds the
session, worker, candidate, JIT/ONNX hashes, remote, branch, and base commit.

On the coordinator, inspect all requests and grant one:

```bash
conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-status \
  --repo /absolute/coordinator-mailbox --session SESSION.json

conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-grant \
  --repo /absolute/coordinator-mailbox --session SESSION.json \
  --worker pc-b --request-id EXACT_REQUEST_ID
```

No second grant is allowed while that lease is active. On the selected worker,
materialize the remote grant and archive with it:

```bash
conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-prepare \
  --repo /absolute/worker-mailbox --session SESSION.json \
  --worker pc-b --lease-id EXACT_LEASE_ID \
  --output /absolute/ARCHIVE_GRANT.json

conda run -n isaacsim-5.1 python scripts/archive_policy_candidate.py \
  SESSION.json TRAINING_RESULTS.json EVALUATION_PLAN.json \
  EVALUATION_RESULTS.json --worker-id pc-b \
  --lease-grant /absolute/ARCHIVE_GRANT.json \
  --output /absolute/ARCHIVE_RECEIPT.json
```

The archiver still performs no Git action. After separate approval, commit and
push the new policy directory in `policy_storage`. Only then publish completion
and let the coordinator release the lease:

```bash
conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-complete \
  --repo /absolute/worker-mailbox --session SESSION.json \
  --worker pc-b --lease-id EXACT_LEASE_ID \
  --archive-receipt /absolute/ARCHIVE_RECEIPT.json

conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-release \
  --repo /absolute/coordinator-mailbox --session SESSION.json \
  --lease-id EXACT_LEASE_ID
```

Completion requires a clean local storage worktree whose HEAD is the exact
remote branch head and tracks the receipt's four files and hashes. Release
rechecks that remote head. If the holder fails before completion, keep the
lease active until the user explicitly approves:

```bash
conda run -n isaacsim-5.1 python scripts/git_mailbox.py archive-revoke \
  --repo /absolute/coordinator-mailbox --session SESSION.json \
  --lease-id EXACT_LEASE_ID --reason "APPROVED_REASON"
```

Never infer revocation from elapsed time, stale progress, or a missing local
process. A lease with completion evidence must be released, not revoked.

## Atomic write behavior

The archiver:

1. verifies the pair and checks for a prior identical archive;
2. locks the storage Git worktree against another skill archive;
3. repeats the clean-worktree check under the lock;
4. creates one private temporary directory inside the collection;
5. copies both artifacts and verifies their hashes again;
6. writes the description and manifest;
7. atomically renames the complete directory to its timestamped destination;
8. removes only its own temporary directory if any step fails.

Never overwrite, merge into, or repair an existing destination. A collision is
an error that requires a fresh authorized run.

## Failure rules

Do not archive when:

- no candidate passes every final-selection gate;
- either JIT or ONNX is missing, optional, failed, or hash-changed;
- the artifact is a symlink or has the wrong extension;
- the storage root is not the exact Git top level;
- the collection is absent, symlinked, or escapes the root;
- the storage worktree is dirty;
- the pair already exists;
- another archive operation holds the lock.

Report the failed prerequisite. Do not weaken a gate, choose another candidate,
create a collection, clean the repository, or commit on the user's behalf.

## Git and hardware boundaries

Successful archival intentionally leaves new untracked files in
`policy_storage`. Report this state and the archive receipt. Do not stage,
commit, push, pull, or resolve repository conflicts without separate explicit
authorization. In shared-repository mode, the lease remains active until a
separately approved commit/push is proven and the coordinator releases it.

Use this exact hardware statement:

> 仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。

Before a real-robot test, separately approve the deployment configuration,
operator, test area, emergency stop, telemetry, limits, rollback policy, and
progressive test envelope.

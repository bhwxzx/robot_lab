# Idempotent campaign controller

Use the controller only with an approved version-6 or version-7 tune session.
It connects existing validated transitions; it does not create new tuning,
evaluation, archive, Git, or hardware authority.

## Contents

- Authorization contract
- Shadow inspection
- Single-host execution
- Distributed execution
- State and recovery
- Deliberate stopping boundaries

## Authorization contract

For one host, add:

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

Change `mode` to `execute` only in a newly approved session that may launch
the exact planned trials and adopt approved adaptive or multi-fidelity
expansions. All three booleans must remain true; partial automation is rejected
because it would make state ownership ambiguous.

For version 7, every computer uses the same session JSON:

```json
{
  "campaign_controller": {
    "enabled": true,
    "mode": "execute",
    "role": "distributed",
    "auto_launch_trials": true,
    "auto_advance_plans": true,
    "stop_before_evaluation": true,
    "worker_mailbox_repos": {
      "pc-a": "/absolute/path/on-pc-a/to/mailbox",
      "pc-b": "/absolute/path/on-pc-b/to/mailbox"
    }
  }
}
```

The mapping must exactly cover all approved workers. Paths may exist only on
their named machines, but every host keeps the complete identical mapping so
the session hash stays common. Select the local machine with `--worker-id`.
The distributed coordinator automatically performs both coordinator and local
worker duties.

## Shadow inspection

`status` never initializes controller state, launches training, writes a
result, commits, or pushes:

```bash
python scripts/campaign_controller.py \
  SESSION.json INITIAL_PLAN.json --action status
```

For version 7 add `--worker-id pc-a`. Distributed status may fetch remote
tracking refs to observe the mailbox, but it makes no remote change. When no
controller state exists, it reports `initialize_controller` and the exact
state path it would create.

`advance` is rejected unless the approved contract uses `mode=execute`.

## Single-host execution

Invoke one bounded transition at each approved polling interval:

```bash
python scripts/campaign_controller.py \
  SESSION.json INITIAL_PLAN.json --action advance
```

Each call does at most one of:

- initialize the controller or executor;
- reconcile the exact recorded child;
- launch one pending child through `execute_trial_plan.py` safety gates;
- create and adopt one deterministic adaptive round or multi-fidelity rung;
- write final training results and ranking.

The controller uses the executor's GPU lock, exact PID/start-time/argv checks,
resource preflight, effective-config gate, checkpoint hashes, retry limits, and
hash-chained journal. It never loops or sleeps internally. Use an external
scheduler for recurring calls.

## Distributed execution

On each host invoke the same command with its worker ID. The controller:

1. lets the coordinator publish the initial immutable campaign;
2. claims only a job assigned to the local worker;
3. materializes the claim plus the exact hash-bound plan snapshot;
4. advances that job through the isolated local executor;
5. publishes its finite result and checkpoint metadata;
6. lets the coordinator wait for the complete barrier and publish one plan
   expansion.

The mailbox publishes full JSON plan snapshots because a job hash alone is not
enough for worker-side deterministic validation. It still transports no
checkpoint, log, video, JIT, ONNX, or credential. A promoted multi-fidelity
trial remains on the worker holding its parent checkpoint.

An active local job may continue when its remote progress is stale if its
exact local executor identity remains valid. Without an active local binding,
`remote_state_unknown` blocks the controller and is never reassigned.

Version 7 automation supports fixed-single-seed static, adaptive, and
multi-fidelity campaigns. Robust multi-seed confirmation publication remains
an explicit existing mailbox step; the controller reports `manual_required`
instead of ranking screening-only evidence.

## State and recovery

Controller state lives under the approved execution or worker state directory.
It binds:

- the exact session file SHA-256;
- the initial and active plan SHA-256;
- local worker identity;
- active distributed job and prepared-plan paths;
- final result and ranking paths.

After final ranking, it also writes `checkpoint_inventory.json`. The inventory
binds the session and ranking hashes and records trial, seed, run, worker,
checkpoint path/hash/step, and final-rung identity. It is metadata only on a
distributed coordinator; the designated evaluation worker must revalidate the
actual local file.

Every state mutation appends a hash-chained journal event before atomically
replacing the state file. A changed session, initial plan, active plan, worker
identity, state snapshot, or journal chain blocks execution. Repeated
`status`, publication, claim, plan snapshot, and terminal calls are
idempotent only when their exact content is unchanged.

## Deliberate stopping boundaries

The controller stops after training ranking with `evaluation_required`.
It never:

- select new parameter paths or values;
- change a seed, rung, margin, objective, constraint, or budget;
- bypass invalid, incomplete, or remote-unknown evidence;
- create an evaluation plan or visually approve motion;
- archive or commit a policy;
- initiate, continue, or qualify a physical test.

Continue manually, or use the separately approved
`evaluation-handoff-controller.md` workflow. Play, deployment-artifact, video
review, supervised hardware, and policy-storage authority remain separate.

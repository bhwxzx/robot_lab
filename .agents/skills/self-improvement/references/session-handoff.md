# Repeated-context-compaction session handoff

Use this workflow when the current session has undergone at least five verified
context compactions. Trigger earlier when the exact count is unavailable or
inconsistent but continuing would require guessing about user approvals,
protected files, completed operations, external state, or the current
objective.

Do not claim an exact compaction count unless it is directly observable. A user
statement that the session has been compressed too many times is sufficient to
trigger the workflow.

## Count current Codex compactions

Before applying the threshold, read the `inspect-context-compactions` skill and
run:

```bash
python3 \
  .agents/skills/inspect-context-compactions/scripts/inspect_context_compactions.py
```

The inspector resolves the current rollout, waits through bounded transient
tail changes, and returns content-free JSON. Treat the result as follows:

- `available`: use `compaction_count` and `threshold_reached` directly;
- `unavailable`: no unique current-session source could be resolved;
- `inconsistent`: the JSONL, session identity, window sequence, or mirrored
  event count did not validate.

Never add `context_compacted` notification events to the top-level `compacted`
record count. Never print or copy `replacement_history`, messages, tool output,
or other rollout content into the handoff. Do not reimplement its counting
rules in this workflow. When the inspector is not available on another agent
runtime, record the exact count as unavailable and use the safety fallback
above.

## Workflow

1. Finish only the current safe, bounded atomic step. Do not start another
   substantial implementation, training run, deployment, remote write, or
   destructive operation.
2. Re-read the repository instructions and relevant skills.
3. Verify inexpensive live state rather than copying it from an old summary:
   repository root, branch, HEAD, worktree changes, relevant processes, and
   task-specific configuration or output paths.
4. Preserve user-owned dirty files and distinguish:
   - live facts verified during handoff;
   - conversation-derived decisions that may now be stale;
   - pending or explicitly unverified work.
5. Create
   `.learnings/session_handoffs/YYYYMMDD-HHMMSS-<task-slug>.md`. Use ASCII
   lowercase letters, digits, and hyphens in `<task-slug>`.
6. Scan the document for passwords, PATs, API keys, private tokens, cookies,
   credential-helper payloads, and unnecessary personal data. Record remote
   names or repository URLs only when required for continuity.
7. Tell the user that a new session is recommended, link the handoff document,
   and provide the same copyable prompt in the response.
8. Stop substantive work in the old session unless the user explicitly directs
   it to continue.

## Required document sections

Use all of these headings:

```markdown
# Session handoff: <task>

## Handoff metadata
## Current objective
## User decisions and protected constraints
## Completed work
## Live-verified state
## Pending and unverified work
## Risks and do-not-repeat actions
## Recommended next steps
## New-session prompt
```

The metadata must include the creation timestamp, repository root, branch and
HEAD when applicable, why the handoff was triggered, compaction query status,
threshold, evidence time, and whether the count is exact or unavailable. When
available, include the thread ID and verified count without including rollout
content.

The completed-work section must cite concrete commits, paths, validations, or
receipts where available. Never represent a plan, approval, attempted command,
or conversation statement as completed execution.

The live-state section must include the verification time. If a live check is
unsafe, expensive, requires network access, or would mutate state, mark it
unverified instead of performing it silently.

The pending section must identify the next bounded action and any approval it
requires. Include existing dirty files so the next session does not accidentally
stage, overwrite, restart, or delete them.

## New-session prompt requirements

Write a self-contained prompt in the user's working language. It must:

- give the absolute repository root and handoff-document path;
- require reading `AGENTS.md`, the handoff, and only the relevant skills;
- begin with read-only verification of HEAD, worktree, relevant processes, and
  drift-prone external state;
- name the current objective and the first bounded next action;
- preserve user-owned files and previously approved safety boundaries;
- prohibit repeating completed work or treating stale conversation evidence as
  current;
- require a modification plan and approval when repository rules demand it.

Keep the prompt concise enough to paste into a new conversation. The handoff
document carries the detail; the prompt carries the navigation, objective, and
safety boundaries.

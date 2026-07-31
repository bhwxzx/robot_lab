# Agent Workflows and Rules

## Long-Running Task Monitoring (Watchdog Pattern)
When asked to monitor a long-running training process (like IsaacLab RL) for silent hangs:
1. **Background Execution**: Start the training task in the background and redirect output to a log file (e.g., `train_watchdog.log`).
2. **Scheduled Checks**: Use the `schedule` tool to wake up periodically (e.g., every 10-15 minutes).
3. **Health Metrics**: Check the `mtime` of the `train_watchdog.log` file. Do NOT rely on checkpoint (`.pt`) file timestamps, as save intervals are often too large.
4. **Recovery**: If the log is stale and GPU utilization drops, use `manage_task` to kill the stalled task and restart it with the `--resume` flag.
5. **Non-blocking**: Inform the user that this background monitoring does NOT block the main conversation thread. Subagents can also be spawned to handle the watchdog loop independently.


## Code Modification Workflow
**CRITICAL RULE**: Always propose a modification plan BEFORE directly modifying any code.
- Provide a clear explanation of what files will be changed and how.
- Do not execute file modification tools (e.g., replace_file_content, sed scripts) until the user explicitly approves the plan.

## Workspace Cleanliness
**CRITICAL RULE**: Clean up intermediate files after use.
- Any temporary scripts, patch files, or intermediate data generated to accomplish a task MUST be deleted immediately after the task is completed or the files are no longer needed. Do not leave the workspace cluttered.

## Session Handoff After Repeated Context Compaction
When the same logical task has undergone at least three explicitly observed
context compactions, use the `self-improvement` skill and follow
`.agents/skills/self-improvement/references/session-handoff.md`.
- Trigger earlier if the exact count is unavailable but continuing would require
  guessing about approvals, protected files, completed work, or live state.
- Never invent a compaction count. Finish only the current safe atomic step,
  verify live state, and create the handoff under
  `.learnings/session_handoffs/`.
- Tell the user to continue in a new session and provide a copyable prompt.
  Stop expanding the old session unless the user explicitly asks to continue.

## Environment Requirements
- The required conda environment for running this project (IsaacLab/RSL-RL) is `isaacsim-5.1`. Do NOT use `isaaclab` or base environments as they may contain incorrect dependency versions.

## Installation and Deletion Workflow
**CRITICAL RULE**: Explicit user consent is strictly required before installing or deleting any packages, libraries, or files.
- You must always propose an installation or deletion plan and wait for the user's explicit approval.
- Do not autonomously run package managers (e.g., `pip install`, `apt-get`, `conda install`) or file deletion commands/tools without permission.

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

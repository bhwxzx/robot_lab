#!/usr/bin/env python3
"""Inspect or advance one authorized tuning campaign by one safe transition."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from build_trial_plan import (
    advance_multifidelity_plan,
    extend_adaptive_plan,
    validate_trial_plan,
)
from execute_trial_plan import (
    ACTIVE_STATUSES,
    _collect_completed_results,
    _load_object,
    _persist_state,
    _prepare_distributed_job,
    _state_path,
    adopt_expanded_plan,
    execution_state_lock,
    initialize_state,
    launch_next,
    reconcile,
    state_summary,
)
from finalize_multifidelity_results import final_rung_results
from git_mailbox import (
    MailboxError,
    _collect_report,
    _fetch,
    _repo,
    _verify_remote,
    claim as mailbox_claim,
    prepare_job as mailbox_prepare_job,
    publish as mailbox_publish,
    publish_adaptive_round,
    publish_multifidelity_rung,
    result as mailbox_result,
    status as mailbox_status,
)
from rank_trials import rank
from validate_session_spec import SpecError, load_and_validate


STATE_NAME = "campaign_controller_state.json"
JOURNAL_NAME = "campaign_controller_events.jsonl"
LOCK_NAME = ".campaign-controller.lock"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SpecError(f"{label} must be an existing absolute regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _write_immutable(path: Path, value: Any) -> None:
    encoded = _canonical_bytes(value) + b"\n"
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            raise SpecError(f"immutable controller artifact collides: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, value)


def _controller_root(
    spec: dict[str, Any],
    worker_id: str | None,
) -> Path:
    contract = spec["campaign_controller"]
    if contract["role"] == "single_host":
        if worker_id is not None:
            raise SpecError("single-host controller does not accept --worker-id")
        root = Path(spec["execution"]["state_dir"]) / "controller"
    else:
        if worker_id not in contract["worker_mailbox_repos"]:
            raise SpecError(
                "distributed controller requires an approved --worker-id"
            )
        worker = next(
            item
            for item in spec["distributed"]["workers"]
            if item["id"] == worker_id
        )
        root = Path(worker["state_dir"]) / "controller"
    if root.exists() and (not root.is_dir() or root.is_symlink()):
        raise SpecError("campaign controller root must be a regular directory")
    return root


@contextmanager
def _controller_lock(root: Path, *, create: bool) -> Iterator[None]:
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir() or root.is_symlink():
        raise SpecError("campaign controller root is unavailable")
    stream = (root / LOCK_NAME).open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise SpecError("campaign controller journal must be a regular file")
    events: list[dict[str, Any]] = []
    previous = None
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError("campaign controller journal is corrupt") from exc
        expected = event.get("event_sha256") if isinstance(event, dict) else None
        unsigned = dict(event) if isinstance(event, dict) else {}
        unsigned.pop("event_sha256", None)
        if (
            not isinstance(event, dict)
            or event.get("version") != 1
            or event.get("sequence") != index + 1
            or event.get("previous_event_sha256") != previous
            or expected != _object_sha256(unsigned)
        ):
            raise SpecError("campaign controller journal hash chain is invalid")
        events.append(event)
        previous = expected
    return events


def _persist_controller(root: Path, state: dict[str, Any], action: str) -> None:
    journal = root / JOURNAL_NAME
    events = _read_events(journal)
    event = {
        "version": 1,
        "sequence": len(events) + 1,
        "recorded_at": time.time(),
        "action": action,
        "previous_event_sha256": (
            events[-1]["event_sha256"] if events else None
        ),
        "state_sha256": _object_sha256(state),
        "state": state,
    }
    event["event_sha256"] = _object_sha256(
        {key: value for key, value in event.items() if key != "event_sha256"}
    )
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_bytes(event).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_write(root / STATE_NAME, state)


def _load_controller(
    root: Path,
    session_path: Path,
    initial_plan_path: Path,
    worker_id: str | None,
) -> dict[str, Any] | None:
    state_path = root / STATE_NAME
    if not state_path.exists():
        return None
    state = _load_json(state_path, "campaign controller state")
    events = _read_events(root / JOURNAL_NAME)
    if (
        not events
        or events[-1].get("state") != state
        or events[-1].get("state_sha256") != _object_sha256(state)
        or state.get("version") != 1
        or state.get("session_sha256") != _file_sha256(session_path)
        or state.get("initial_plan_sha256") != _file_sha256(initial_plan_path)
        or state.get("worker_id") != worker_id
    ):
        raise SpecError("campaign controller state binding is invalid")
    return state


def _new_controller_state(
    spec: dict[str, Any],
    session_path: Path,
    initial_plan_path: Path,
    worker_id: str | None,
) -> dict[str, Any]:
    role = (
        "single_host"
        if worker_id is None
        else (
            "coordinator_worker"
            if worker_id == spec["distributed"]["coordinator_id"]
            else "worker"
        )
    )
    return {
        "version": 1,
        "session_sha256": _file_sha256(session_path),
        "initial_plan_sha256": _file_sha256(initial_plan_path),
        "role": role,
        "worker_id": worker_id,
        "active_plan_path": str(initial_plan_path),
        "active_plan_sha256": _file_sha256(initial_plan_path),
        "initial_plan_published": False,
        "active_job": None,
        "training_results_path": None,
        "training_ranking_path": None,
        "checkpoint_inventory_path": None,
        "last_action": "initialized",
        "updated_at": time.time(),
    }


def _active_plan(state: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(state["active_plan_path"])
    plan = _load_json(path, "active trial plan")
    if _file_sha256(path) != state["active_plan_sha256"]:
        raise SpecError("active trial plan changed after controller binding")
    return path, plan


def _executor_state(spec: dict[str, Any]) -> dict[str, Any] | None:
    path = Path(spec["execution"]["state_dir"]) / "execution_state.json"
    return _load_json(path, "execution state") if path.exists() else None


def _single_decision(
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    plan_path, plan = _active_plan(state)
    executor = _executor_state(spec)
    if executor is None:
        action, reason = "initialize_executor", "executor_state_absent"
        summary = None
    else:
        if (
            executor.get("session_sha256") != state["session_sha256"]
            or executor.get("plan_sha256") != _file_sha256(plan_path)
        ):
            raise SpecError("executor state differs from controller bindings")
        summary = state_summary(executor)
        runs = list(executor["runs"].values())
        if any(run["status"] in ACTIVE_STATUSES for run in runs):
            action, reason = "reconcile", "authorized_child_active"
        elif any(run["status"] == "failed" for run in runs):
            action, reason = "blocked", "executor_contains_failed_run"
        elif any(run["status"] == "pending" for run in runs):
            action, reason = "launch_next", "pending_run_available"
        elif runs and all(run["status"] == "completed" for run in runs):
            if (
                isinstance(plan.get("multi_fidelity"), dict)
                and plan["multi_fidelity"]["status"] == "running"
            ):
                action, reason = "advance_plan", "rung_barrier_complete"
            elif (
                isinstance(plan.get("adaptive"), dict)
                and plan["adaptive"]["status"] == "running"
            ):
                action, reason = "advance_plan", "adaptive_round_complete"
            elif (
                state.get("training_ranking_path") is None
                or state.get("checkpoint_inventory_path") is None
            ):
                action, reason = "finalize_training", "training_runs_complete"
            else:
                action, reason = (
                    "evaluation_required",
                    "controller_stops_before_evaluation",
                )
        else:
            action, reason = "blocked", "executor_state_has_no_progressable_run"
    return {
        "next_action": action,
        "reason": reason,
        "active_plan_path": str(plan_path),
        "active_plan_sha256": state["active_plan_sha256"],
        "executor": summary,
    }


def _current_rung_results(
    plan: dict[str, Any],
    executor: dict[str, Any],
) -> list[dict[str, Any]]:
    current_rung = len(plan["multi_fidelity"]["rungs"])
    results = []
    for run in executor["runs"].values():
        if run.get("rung") != current_rung:
            continue
        if run["status"] != "completed":
            raise SpecError("multi-fidelity rung barrier is incomplete")
        results.append(_load_json(Path(run["result_path"]), "run result"))
    return results


def _snapshot_plan(
    root: Path,
    plan: dict[str, Any],
) -> Path:
    path = root / "plans" / f"{_object_sha256(plan)}.json"
    _write_immutable(path, plan)
    return path


def _executor_persist_callback(
    spec: dict[str, Any],
    state_path: Path,
):
    def persist(value: dict[str, Any], action: str) -> None:
        _persist_state(spec, state_path, value, f"controller-{action}")

    return persist


def _checkpoint_step(path: Path, explicit: Any = None) -> int:
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    if match is None:
        raise SpecError(
            "checkpoint step is absent and cannot be derived from model_N.pt"
        )
    return int(match.group(1))


def _single_checkpoint_entries(
    executor: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for run in executor["runs"].values():
        if run.get("status") != "completed":
            continue
        terminal = run.get("terminal_receipt")
        checkpoint = (
            terminal.get("checkpoint") if isinstance(terminal, dict) else None
        )
        if not isinstance(checkpoint, dict):
            continue
        path = Path(checkpoint.get("path", ""))
        expected_hash = checkpoint.get("sha256")
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or not isinstance(expected_hash, str)
            or _file_sha256(path) != expected_hash
        ):
            raise SpecError("completed training checkpoint evidence changed")
        entries.append(
            {
                "trial_id": run["trial_id"],
                "seed": run["seed"],
                "run_id": run["run_id"],
                "worker_id": None,
                "checkpoint_path": str(path),
                "checkpoint_sha256": expected_hash,
                "checkpoint_step": _checkpoint_step(
                    path,
                    checkpoint.get("step"),
                ),
                "rsl_rl_run_dir": terminal.get("rsl_rl_run_dir"),
                "rung": run.get("rung"),
                "target_budget": run.get("target_budget"),
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            item["trial_id"],
            item["seed"],
            item["checkpoint_step"],
            item["run_id"],
        ),
    )


def _distributed_checkpoint_entries(
    envelopes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for envelope in envelopes:
        result = envelope.get("result")
        manifest = envelope.get("artifact_manifest")
        artifacts = (
            manifest.get("artifacts") if isinstance(manifest, dict) else None
        )
        checkpoints = [
            item
            for item in artifacts or []
            if isinstance(item, dict) and item.get("kind") == "checkpoint"
        ]
        if not isinstance(result, dict) or len(checkpoints) != 1:
            raise SpecError(
                "distributed training result lacks one checkpoint manifest"
            )
        checkpoint = checkpoints[0]
        path = Path(checkpoint.get("path", ""))
        expected_hash = checkpoint.get("sha256")
        result_checkpoint = result.get("checkpoint")
        explicit_step = (
            result_checkpoint.get("step")
            if isinstance(result_checkpoint, dict)
            else None
        )
        rsl_rl_run_dir = (
            result_checkpoint.get("rsl_rl_run_dir")
            if isinstance(result_checkpoint, dict)
            else str(path.parent)
        )
        if (
            not path.is_absolute()
            or not isinstance(expected_hash, str)
            or not isinstance(envelope.get("worker_id"), str)
        ):
            raise SpecError("distributed checkpoint metadata is invalid")
        entries.append(
            {
                "trial_id": result["trial_id"],
                "seed": result["seed"],
                "run_id": envelope["job_id"],
                "worker_id": envelope["worker_id"],
                "checkpoint_path": str(path),
                "checkpoint_sha256": expected_hash,
                "checkpoint_step": _checkpoint_step(path, explicit_step),
                "rsl_rl_run_dir": rsl_rl_run_dir,
                "rung": result.get("rung"),
                "target_budget": result.get("target_budget"),
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            item["trial_id"],
            item["seed"],
            item["checkpoint_step"],
            item["run_id"],
        ),
    )


def _checkpoint_inventory(
    spec: dict[str, Any],
    session_path: Path,
    ranking_path: Path,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not entries:
        raise SpecError("training produced no checkpoint inventory entries")
    return {
        "version": 1,
        "session_sha256": _file_sha256(session_path),
        "training_ranking_sha256": _file_sha256(ranking_path),
        "training_run_id": spec["training"]["run_id"],
        "algorithm": spec["algorithm"],
        "entries": entries,
    }


def _advance_single(
    spec: dict[str, Any],
    session_path: Path,
    root: Path,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    plan_path, plan = _active_plan(state)
    action = decision["next_action"]
    if action in {"blocked", "evaluation_required"}:
        return {"action_taken": "none", **decision}
    if action == "initialize_executor":
        with execution_state_lock(spec):
            state_path, executor = initialize_state(
                spec,
                session_path,
                plan,
                plan_path,
            )
            _persist_state(spec, state_path, executor, "controller-initialize")
    elif action == "reconcile":
        with execution_state_lock(spec):
            state_path, executor = initialize_state(
                spec,
                session_path,
                plan,
                plan_path,
            )
            executor = reconcile(spec, plan, executor)
            _persist_state(spec, state_path, executor, "controller-reconcile")
    elif action == "launch_next":
        with execution_state_lock(spec):
            state_path, executor = initialize_state(
                spec,
                session_path,
                plan,
                plan_path,
            )
            executor = reconcile(spec, plan, executor)
            _persist_state(
                spec,
                state_path,
                executor,
                "controller-reconcile-before-launch",
            )
            launch_next(
                spec,
                executor,
                _executor_persist_callback(spec, state_path),
            )
    elif action == "advance_plan":
        executor = _executor_state(spec)
        if executor is None:
            raise SpecError("executor state disappeared before plan advancement")
        if isinstance(plan.get("multi_fidelity"), dict):
            expanded = advance_multifidelity_plan(
                spec,
                plan,
                _current_rung_results(plan, executor),
            )
        else:
            expanded = extend_adaptive_plan(
                spec,
                plan,
                _collect_completed_results(executor),
            )
        expanded_path = _snapshot_plan(root, expanded)
        with execution_state_lock(spec):
            state_path, adopted = adopt_expanded_plan(
                spec,
                session_path,
                expanded,
                expanded_path,
            )
            _persist_state(spec, state_path, adopted, "controller-adopt-plan")
        state["active_plan_path"] = str(expanded_path)
        state["active_plan_sha256"] = _file_sha256(expanded_path)
    elif action == "finalize_training":
        executor = _executor_state(spec)
        if executor is None:
            raise SpecError("executor state disappeared before final ranking")
        if isinstance(plan.get("multi_fidelity"), dict):
            results = final_rung_results(spec, plan)
        else:
            results = {"runs": _collect_completed_results(executor)}
        ranking = rank(spec, results["runs"])
        results_path = root / "training_results.json"
        ranking_path = root / "training_ranking.json"
        inventory_path = root / "checkpoint_inventory.json"
        _write_immutable(results_path, results)
        _write_immutable(ranking_path, ranking)
        inventory = _checkpoint_inventory(
            spec,
            session_path,
            ranking_path,
            _single_checkpoint_entries(executor),
        )
        _write_immutable(inventory_path, inventory)
        state["training_results_path"] = str(results_path)
        state["training_ranking_path"] = str(ranking_path)
        state["checkpoint_inventory_path"] = str(inventory_path)
    else:
        raise SpecError(f"unsupported single-host controller action: {action}")
    state["last_action"] = action
    state["updated_at"] = time.time()
    _persist_controller(root, state, action)
    return {
        "action_taken": action,
        **_single_decision(spec, state),
        "training_results_path": state["training_results_path"],
        "training_ranking_path": state["training_ranking_path"],
        "checkpoint_inventory_path": state.get("checkpoint_inventory_path"),
    }


def _prepared_job_paths(root: Path, job_id: str) -> dict[str, Path]:
    base = root / "jobs" / job_id
    return {
        "base": base,
        "prepared": base / "prepared.json",
        "plan": base / "plan.json",
        "artifacts": base / "artifacts.json",
    }


def _runtime_job(
    spec: dict[str, Any],
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    active = state.get("active_job")
    if not isinstance(active, dict):
        raise SpecError("campaign controller has no active distributed job")
    prepared = _load_json(Path(active["prepared_path"]), "prepared job")
    plan = _load_json(Path(active["plan_path"]), "distributed trial plan")
    runtime_spec, runtime_plan = _prepare_distributed_job(
        spec,
        plan,
        prepared,
        state["worker_id"],
    )
    return prepared, runtime_spec, runtime_plan


def _job_artifacts(run: dict[str, Any], output_path: Path) -> None:
    terminal = run.get("terminal_receipt")
    checkpoint = (
        terminal.get("checkpoint") if isinstance(terminal, dict) else None
    )
    artifacts: list[dict[str, Any]] = []
    if isinstance(checkpoint, dict):
        path = Path(checkpoint["path"])
        if (
            not path.is_absolute()
            or not path.is_file()
            or path.is_symlink()
            or _file_sha256(path) != checkpoint["sha256"]
        ):
            raise SpecError("distributed result checkpoint evidence changed")
        artifacts.append(
            {
                "kind": "checkpoint",
                "path": str(path),
                "sha256": checkpoint["sha256"],
                "size_bytes": path.stat().st_size,
            }
        )
    _write_immutable(output_path, {"artifacts": artifacts})


def _worker_decision(
    spec: dict[str, Any],
    state: dict[str, Any],
    *,
    inspect_source: bool,
) -> dict[str, Any]:
    contract = spec["campaign_controller"]
    worker_id = state["worker_id"]
    report = mailbox_status(
        contract["worker_mailbox_repos"][worker_id],
        state["session_path"],
        worker_id,
        inspect_source=inspect_source,
    )
    active = state.get("active_job")
    if isinstance(active, dict):
        matching = next(
            (
                job
                for job in report["jobs"]
                if job["job_id"] == active["job_id"]
            ),
            None,
        )
        if matching is None:
            action, reason = "blocked", "active_job_missing_from_mailbox"
        elif matching["state"] == "completed":
            action, reason = "clear_completed_job", "remote_result_visible"
        else:
            _, runtime_spec, runtime_plan = _runtime_job(spec, state)
            executor = _executor_state(runtime_spec)
            if executor is None:
                action, reason = "initialize_worker_executor", "job_not_initialized"
            else:
                runs = list(executor["runs"].values())
                if any(run["status"] in ACTIVE_STATUSES for run in runs):
                    action, reason = "reconcile_worker", "worker_child_active"
                elif any(run["status"] == "failed" for run in runs):
                    action, reason = "blocked", "worker_executor_failed"
                elif any(run["status"] == "pending" for run in runs):
                    action, reason = "launch_worker", "worker_job_pending"
                elif runs and all(run["status"] == "completed" for run in runs):
                    action, reason = "publish_worker_result", "worker_job_complete"
                else:
                    action, reason = "blocked", "worker_executor_not_progressable"
    else:
        candidates = [
            job
            for job in report["jobs"]
            if job["state"] in {"pending", "claimed"}
        ]
        unknown = [
            job for job in report["jobs"] if job["state"] == "remote_state_unknown"
        ]
        if candidates:
            action, reason = "claim_job", "assigned_job_available"
        elif unknown:
            action, reason = "blocked", "remote_state_unknown"
        else:
            action, reason = "wait_remote", "no_local_job_available"
    return {
        "next_action": action,
        "reason": reason,
        "worker_id": worker_id,
        "mailbox": report,
    }


def _advance_worker(
    spec: dict[str, Any],
    session_path: Path,
    root: Path,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    action = decision["next_action"]
    contract = spec["campaign_controller"]
    worker_id = state["worker_id"]
    repo = contract["worker_mailbox_repos"][worker_id]
    if action in {"wait_remote", "blocked"}:
        return {"action_taken": "none", **decision}
    if action == "claim_job":
        candidate = next(
            job
            for job in decision["mailbox"]["jobs"]
            if job["state"] in {"pending", "claimed"}
        )
        job_id = candidate["job_id"]
        attempt = 1
        mailbox_claim(repo, session_path, worker_id, job_id, attempt)
        paths = _prepared_job_paths(root, job_id)
        mailbox_prepare_job(
            repo,
            session_path,
            worker_id,
            job_id,
            attempt,
            paths["prepared"],
        )
        prepared = _load_json(paths["prepared"], "prepared job")
        plan = prepared.get("plan")
        if not isinstance(plan, dict):
            raise SpecError("prepared distributed job lacks a plan snapshot")
        _write_immutable(paths["plan"], plan)
        state["active_job"] = {
            "job_id": job_id,
            "attempt": attempt,
            "prepared_path": str(paths["prepared"]),
            "plan_path": str(paths["plan"]),
        }
    elif action == "clear_completed_job":
        state["active_job"] = None
    else:
        prepared, runtime_spec, runtime_plan = _runtime_job(spec, state)
        active = state["active_job"]
        job_id = active["job_id"]
        plan_path = Path(active["plan_path"])
        if action == "initialize_worker_executor":
            with execution_state_lock(runtime_spec):
                state_path, executor = initialize_state(
                    runtime_spec,
                    session_path,
                    runtime_plan,
                    plan_path,
                    plan_already_validated=True,
                )
                _persist_state(
                    runtime_spec,
                    state_path,
                    executor,
                    "controller-initialize-worker",
                )
        elif action == "reconcile_worker":
            with execution_state_lock(runtime_spec):
                state_path, executor = initialize_state(
                    runtime_spec,
                    session_path,
                    runtime_plan,
                    plan_path,
                    plan_already_validated=True,
                )
                executor = reconcile(runtime_spec, runtime_plan, executor)
                _persist_state(
                    runtime_spec,
                    state_path,
                    executor,
                    "controller-reconcile-worker",
                )
        elif action == "launch_worker":
            with execution_state_lock(runtime_spec):
                state_path, executor = initialize_state(
                    runtime_spec,
                    session_path,
                    runtime_plan,
                    plan_path,
                    plan_already_validated=True,
                )
                executor = reconcile(runtime_spec, runtime_plan, executor)
                _persist_state(
                    runtime_spec,
                    state_path,
                    executor,
                    "controller-reconcile-before-worker-launch",
                )
                launch_next(
                    runtime_spec,
                    executor,
                    _executor_persist_callback(runtime_spec, state_path),
                )
        elif action == "publish_worker_result":
            executor = _executor_state(runtime_spec)
            if executor is None or len(executor["runs"]) != 1:
                raise SpecError("distributed worker executor state is invalid")
            run = next(iter(executor["runs"].values()))
            paths = _prepared_job_paths(root, job_id)
            _job_artifacts(run, paths["artifacts"])
            mailbox_result(
                repo,
                session_path,
                worker_id,
                job_id,
                active["attempt"],
                run["result_path"],
                paths["artifacts"],
            )
            state["active_job"] = None
        else:
            raise SpecError(f"unsupported distributed worker action: {action}")
    state["last_action"] = action
    state["updated_at"] = time.time()
    _persist_controller(root, state, action)
    return {"action_taken": action, "active_job": state["active_job"]}


def _collect_mailbox_results(
    spec: dict[str, Any],
    session_path: Path,
    root: Path,
    worker_id: str,
) -> dict[str, Any]:
    del root, session_path
    repo = _repo(
        spec["campaign_controller"]["worker_mailbox_repos"][worker_id]
    )
    _verify_remote(repo, spec)
    _fetch(repo)
    return _collect_report(repo, spec)


def _coordinator_decision(
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    if not state["initial_plan_published"]:
        return {
            "next_action": "publish_initial_plan",
            "reason": "campaign_not_published",
        }
    _, plan = _active_plan(state)
    report = _collect_mailbox_results(
        spec,
        Path(state["session_path"]),
        _controller_root(spec, state["worker_id"]),
        state["worker_id"],
    )
    if report["invalid_results"]:
        return {
            "next_action": "blocked",
            "reason": "mailbox_contains_invalid_results",
        }
    accepted_envelopes = {
        envelope["job_id"]: envelope
        for envelope in report["accepted_results"]
        if isinstance(envelope, dict)
        and isinstance(envelope.get("result"), dict)
    }
    accepted = {
        envelope["job_id"]: envelope["result"]
        for envelope in accepted_envelopes.values()
    }
    current_runs = (
        [
            run
            for run in plan["runs"]
            if run.get("rung") == len(plan["multi_fidelity"]["rungs"])
        ]
        if isinstance(plan.get("multi_fidelity"), dict)
        else plan["runs"]
    )
    expected_ids = {run["run_id"] for run in current_runs}
    if not expected_ids <= set(accepted):
        return {
            "next_action": "wait_remote",
            "reason": "distributed_barrier_incomplete",
            "completed": len(expected_ids & set(accepted)),
            "expected": len(expected_ids),
        }
    if (
        isinstance(plan.get("multi_fidelity"), dict)
        and plan["multi_fidelity"]["status"] == "running"
    ) or (
        isinstance(plan.get("adaptive"), dict)
        and plan["adaptive"]["status"] == "running"
    ):
        return {
            "next_action": "publish_plan_expansion",
            "reason": "distributed_barrier_complete",
            "results": [accepted[run["run_id"]] for run in current_runs],
        }
    if (
        not isinstance(plan.get("multi_fidelity"), dict)
        and not isinstance(plan.get("adaptive"), dict)
        and spec["tuning"]["seed_strategy"].get(
            "mode",
            "robust_multi_seed",
        )
        == "robust_multi_seed"
    ):
        return {
            "next_action": "manual_required",
            "reason": "distributed_confirmation_publication_not_automated",
        }
    if (
        state["training_ranking_path"] is None
        or state.get("checkpoint_inventory_path") is None
    ):
        return {
            "next_action": "finalize_training",
            "reason": "distributed_training_complete",
            "results": [accepted[run["run_id"]] for run in current_runs],
            "checkpoint_envelopes": [
                accepted_envelopes[run["run_id"]] for run in current_runs
            ],
        }
    return {
        "next_action": "evaluation_required",
        "reason": "controller_stops_before_evaluation",
    }


def _advance_coordinator(
    spec: dict[str, Any],
    session_path: Path,
    root: Path,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    action = decision["next_action"]
    plan_path, plan = _active_plan(state)
    repo = spec["campaign_controller"]["worker_mailbox_repos"][
        state["worker_id"]
    ]
    if action in {
        "wait_remote",
        "blocked",
        "manual_required",
        "evaluation_required",
    }:
        return {"action_taken": "none", **decision}
    if action == "publish_initial_plan":
        mailbox_publish(repo, session_path, plan_path)
        state["initial_plan_published"] = True
    elif action == "publish_plan_expansion":
        if isinstance(plan.get("multi_fidelity"), dict):
            expanded = advance_multifidelity_plan(
                spec,
                plan,
                decision["results"],
            )
        elif isinstance(plan.get("adaptive"), dict):
            expanded = extend_adaptive_plan(
                spec,
                plan,
                decision["results"],
            )
        else:
            raise SpecError(
                "distributed controller plan expansion supports adaptive or "
                "multi-fidelity plans"
            )
        expanded_path = _snapshot_plan(root, expanded)
        if isinstance(plan.get("multi_fidelity"), dict):
            publish_multifidelity_rung(
                repo,
                session_path,
                plan_path,
                expanded_path,
            )
        else:
            publish_adaptive_round(
                repo,
                session_path,
                plan_path,
                expanded_path,
            )
        state["active_plan_path"] = str(expanded_path)
        state["active_plan_sha256"] = _file_sha256(expanded_path)
    elif action == "finalize_training":
        if isinstance(plan.get("multi_fidelity"), dict):
            results = final_rung_results(spec, plan)
        else:
            results = {"runs": decision["results"]}
        ranking = rank(spec, results["runs"])
        results_path = root / "training_results.json"
        ranking_path = root / "training_ranking.json"
        inventory_path = root / "checkpoint_inventory.json"
        _write_immutable(results_path, results)
        _write_immutable(ranking_path, ranking)
        inventory = _checkpoint_inventory(
            spec,
            session_path,
            ranking_path,
            _distributed_checkpoint_entries(
                decision["checkpoint_envelopes"],
            ),
        )
        _write_immutable(inventory_path, inventory)
        state["training_results_path"] = str(results_path)
        state["training_ranking_path"] = str(ranking_path)
        state["checkpoint_inventory_path"] = str(inventory_path)
    else:
        raise SpecError(f"unsupported coordinator action: {action}")
    state["last_action"] = action
    state["updated_at"] = time.time()
    _persist_controller(root, state, action)
    return {
        "action_taken": action,
        "active_plan_path": state["active_plan_path"],
        "training_ranking_path": state["training_ranking_path"],
        "checkpoint_inventory_path": state.get("checkpoint_inventory_path"),
    }


def inspect_or_advance(
    session_path: Path,
    initial_plan_path: Path,
    *,
    execute: bool,
    worker_id: str | None = None,
) -> dict[str, Any]:
    spec = load_and_validate(session_path)
    contract = spec.get("campaign_controller")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        raise SpecError("session does not enable campaign_controller")
    if execute and contract["mode"] != "execute":
        raise SpecError(
            "campaign_controller advance requires session mode=execute"
        )
    plan = _load_json(initial_plan_path, "initial trial plan")
    validate_trial_plan(spec, plan)
    root = _controller_root(spec, worker_id)
    state = (
        _load_controller(root, session_path, initial_plan_path, worker_id)
        if root.exists()
        else None
    )
    if state is None:
        state = _new_controller_state(
            spec,
            session_path,
            initial_plan_path,
            worker_id,
        )
        state["session_path"] = str(session_path)
        if not execute:
            return {
                "mode": "shadow",
                "role": state["role"],
                "next_action": "initialize_controller",
                "reason": "controller_state_absent",
                "would_write": str(root / STATE_NAME),
            }
        root.mkdir(parents=True, exist_ok=True)
        with _controller_lock(root, create=False):
            _persist_controller(root, state, "initialize-controller")
        return {
            "mode": "execute",
            "role": state["role"],
            "action_taken": "initialize_controller",
            "state_path": str(root / STATE_NAME),
        }
    role = state["role"]
    with _controller_lock(root, create=False):
        if role == "single_host":
            decision = _single_decision(spec, state)
            result = (
                _advance_single(
                    spec,
                    session_path,
                    root,
                    state,
                    decision,
                )
                if execute
                else {"action_taken": "none", **decision}
            )
        else:
            if role == "coordinator_worker" and not state[
                "initial_plan_published"
            ]:
                coordinator_decision = _coordinator_decision(spec, state)
                result = (
                    _advance_coordinator(
                        spec,
                        session_path,
                        root,
                        state,
                        coordinator_decision,
                    )
                    if execute
                    else {"action_taken": "none", **coordinator_decision}
                )
            else:
                worker_decision = _worker_decision(
                    spec,
                    state,
                    inspect_source=execute,
                )
                if worker_decision["next_action"] != "wait_remote":
                    result = (
                        _advance_worker(
                            spec,
                            session_path,
                            root,
                            state,
                            worker_decision,
                        )
                        if execute
                        else {"action_taken": "none", **worker_decision}
                    )
                elif role == "coordinator_worker":
                    coordinator_decision = _coordinator_decision(spec, state)
                    result = (
                        _advance_coordinator(
                            spec,
                            session_path,
                            root,
                            state,
                            coordinator_decision,
                        )
                        if execute
                        else {"action_taken": "none", **coordinator_decision}
                    )
                else:
                    result = {"action_taken": "none", **worker_decision}
    return {
        "mode": "execute" if execute else "shadow",
        "role": role,
        "state_path": str(root / STATE_NAME),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("initial_plan")
    parser.add_argument(
        "--worker-id",
        help="Required local worker identity for a distributed controller",
    )
    parser.add_argument(
        "--action",
        choices={"status", "advance"},
        required=True,
    )
    args = parser.parse_args()
    try:
        result = inspect_or_advance(
            Path(args.session).resolve(),
            Path(args.initial_plan).resolve(),
            execute=args.action == "advance",
            worker_id=args.worker_id,
        )
    except (MailboxError, OSError, SpecError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

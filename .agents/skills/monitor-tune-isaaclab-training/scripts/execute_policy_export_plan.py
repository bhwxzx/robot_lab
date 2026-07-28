#!/usr/bin/env python3
"""Execute approved policy exports with transactional attempts and parity gates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from build_policy_export_plan import validate_plan
from execute_evaluation_plan import (
    _find_process_with_exact_token,
    _gpu_idle,
    _process_is_exact,
    _process_start_ticks,
    _stop_exact_process,
)
from execute_trial_plan import gpu_lock_path
from validate_session_spec import SpecError, load_and_validate


ACTIVE_STATUSES = {
    "launching",
    "running",
    "stopping_timeout",
    "stopping_forced",
}
JOURNAL_NAME = "policy_export_events.jsonl"
STATE_NAME = "policy_export_state.json"
LOCK_NAME = ".policy-export-state.lock"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PersistCallback = Callable[[dict[str, Any], str], None]
_ACTIVE_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"{label} is invalid JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical_bytes(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_immutable(path: Path, value: Any) -> None:
    encoded = _canonical_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise SpecError(f"immutable policy export artifact changed: {path}")
        return
    _atomic_write(path, value)


def _state_dir(spec: dict[str, Any]) -> Path:
    return Path(spec["policy_export"]["output_dir"]) / ".executor"


def _state_path(spec: dict[str, Any]) -> Path:
    return _state_dir(spec) / STATE_NAME


def _journal_path(spec: dict[str, Any]) -> Path:
    return _state_dir(spec) / JOURNAL_NAME


@contextmanager
def export_state_lock(spec: dict[str, Any]) -> Iterator[None]:
    root = _state_dir(spec)
    root.mkdir(parents=True, exist_ok=True)
    stream = (root / LOCK_NAME).open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError("policy export state is already being modified") from exc
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _read_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous: str | None = None
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines):
                break
            raise SpecError(
                f"invalid policy export journal JSON at line {index}"
            ) from exc
        if not isinstance(event, dict):
            raise SpecError("policy export journal event must be an object")
        unsigned = {
            key: value for key, value in event.items()
            if key != "event_sha256"
        }
        if (
            event.get("sequence") != index
            or event.get("previous_event_sha256") != previous
            or event.get("event_sha256") != _object_sha256(unsigned)
            or event.get("state_sha256")
            != _object_sha256(event.get("state"))
        ):
            raise SpecError("policy export journal chain is invalid")
        previous = event["event_sha256"]
        events.append(event)
    return events


def _persist_state(
    spec: dict[str, Any],
    state: dict[str, Any],
    action: str,
) -> None:
    journal_path = _journal_path(spec)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    events = _read_journal(journal_path)
    if journal_path.exists():
        canonical_journal = b"".join(
            _canonical_bytes(event) + b"\n" for event in events
        )
        if journal_path.read_bytes() != canonical_journal:
            temporary = journal_path.with_name(
                f".{journal_path.name}.{os.getpid()}.tmp"
            )
            with temporary.open("wb") as stream:
                stream.write(canonical_journal)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, journal_path)
    event: dict[str, Any] = {
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
    event["event_sha256"] = _object_sha256(event)
    with journal_path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_bytes(event).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_write(_state_path(spec), state)


def recover_state_from_journal(
    spec: dict[str, Any],
    session_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    events = _read_journal(_journal_path(spec))
    if not events:
        raise SpecError("policy export journal has no recoverable state")
    state = events[-1]["state"]
    if (
        state.get("session_sha256") != _sha256(session_path)
        or state.get("plan_sha256") != _sha256(plan_path)
    ):
        raise SpecError("policy export journal binding is invalid")
    _atomic_write(_state_path(spec), state)
    return state


def _run_record(spec: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    run_dir = _state_dir(spec) / "runs" / run["run_id"]
    return {
        "run_id": run["run_id"],
        "candidate_id": run["candidate_id"],
        "trial_id": run["trial_id"],
        "seed": run["seed"],
        "status": "pending",
        "attempts": 0,
        "run_dir": str(run_dir),
        "attempt_dir": None,
        "result_path": None,
        "log_path": None,
        "launch_receipt_path": None,
        "launch_receipt_sha256": None,
        "argv": None,
        "pid": None,
        "process_group": None,
        "process_start_ticks": None,
        "reserved_at": None,
        "launched_at": None,
        "stop_requested_at": None,
        "stop_reason": None,
        "finished_at": None,
        "failure_reason": None,
        "resource_report": None,
        "result_sha256": None,
        "artifacts": None,
        "parity": None,
    }


def initialize_state(
    spec: dict[str, Any],
    session_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
) -> dict[str, Any]:
    validate_plan(spec, plan)
    state_path = _state_path(spec)
    session_hash = _sha256(session_path)
    if plan.get("session_sha256") != session_hash:
        raise SpecError("policy export plan is not bound to the exact session")
    plan_hash = _sha256(plan_path)
    if state_path.exists():
        state = _load_object(state_path, "policy export state")
        events = _read_journal(_journal_path(spec))
        if (
            state.get("session_sha256") != session_hash
            or state.get("plan_sha256") != plan_hash
            or not events
            or events[-1].get("state") != state
        ):
            raise SpecError("existing policy export state binding is invalid")
        return state
    manifest_path = Path(spec["policy_export"]["output_dir"]) / "export_manifest.json"
    if manifest_path.exists():
        raise SpecError("policy export manifest already exists before initialization")
    now = time.time()
    return {
        "version": 1,
        "session_sha256": session_hash,
        "plan_sha256": plan_hash,
        "created_at": now,
        "updated_at": now,
        "stage": "executing",
        "manifest_path": None,
        "manifest_sha256": None,
        "runs": {
            run["run_id"]: _run_record(spec, run)
            for run in plan["runs"]
        },
    }


def _resource_preflight(spec: dict[str, Any]) -> dict[str, Any]:
    contract = spec["policy_export"]["execution"]
    free_gb = shutil.disk_usage(_state_dir(spec)).free / 1_000_000_000
    if free_gb < float(contract["min_free_disk_gb"]):
        raise SpecError("policy export free disk is below the approved minimum")
    gpu_index = spec["policy_export"]["gpu_index"]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        temperature = int(result.stdout.strip().splitlines()[0])
    except (
        FileNotFoundError,
        IndexError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        raise SpecError(f"cannot verify policy export GPU health: {exc}") from exc
    if temperature > contract["max_gpu_temperature_c"]:
        raise SpecError("policy export GPU temperature exceeds approved maximum")
    return {
        "checked_at": time.time(),
        "free_disk_gb": free_gb,
        "gpu_temperature_c": temperature,
    }


def _render_command(
    spec: dict[str, Any],
    expected: dict[str, Any],
    attempt_dir: Path,
) -> tuple[list[str], dict[str, Path]]:
    filenames = expected["artifact_filenames"]
    paths = {
        kind: attempt_dir / filename
        for kind, filename in filenames.items()
    }
    values: dict[str, Any] = {
        "candidate_id": expected["candidate_id"],
        "checkpoint_path": expected["checkpoint_path"],
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "export_run_id": expected["run_id"],
        "gpu_index": spec["policy_export"]["gpu_index"],
        "history_contract": spec["policy_export"]["parity"][
            "history_contract"
        ],
        "max_abs_action_error": spec["policy_export"]["parity"][
            "max_abs_action_error"
        ],
        "minimum_parity_samples": spec["policy_export"]["parity"][
            "minimum_samples"
        ],
        "normalization_contract": spec["policy_export"]["parity"][
            "normalization_contract"
        ],
        "require_idle_gpu_flag": "--require_idle_gpu",
        "result_path": str(attempt_dir / "result.json"),
        "seed": expected["seed"],
        "trial_id": expected["trial_id"],
    }
    for kind, path in paths.items():
        values[f"{kind}_path"] = str(path)
    try:
        command = [
            token.format_map(values) for token in expected["command_template"]
        ]
    except (KeyError, ValueError) as exc:
        raise SpecError(f"cannot render policy export command: {exc}") from exc
    if command.count(expected["run_id"]) < 1:
        raise SpecError("policy export command lacks the exact export run ID token")
    return command, paths


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _validate_shape(value: Any, path: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise SpecError(f"{path} must be a non-empty positive integer array")
    return value


def _load_valid_result(
    spec: dict[str, Any],
    expected: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    result_path = Path(run["result_path"])
    result = _load_object(result_path, "policy export result")
    required = {
        "version",
        "export_run_id",
        "candidate_id",
        "checkpoint_path",
        "checkpoint_sha256",
        "status",
        "artifacts",
        "parity",
    }
    if (
        set(result) != required
        or result.get("version") != 1
        or result.get("export_run_id") != expected["run_id"]
        or result.get("candidate_id") != expected["candidate_id"]
        or result.get("checkpoint_path") != expected["checkpoint_path"]
        or result.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or result.get("status") != "completed"
    ):
        raise SpecError("policy export result identity is invalid")
    checkpoint = Path(expected["checkpoint_path"])
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or _sha256(checkpoint) != expected["checkpoint_sha256"]
    ):
        raise SpecError("policy export checkpoint changed after launch")
    artifacts = result["artifacts"]
    if (
        not isinstance(artifacts, dict)
        or set(artifacts) != set(expected["artifact_filenames"])
    ):
        raise SpecError("policy export result artifact coverage is invalid")
    minimum_bytes = spec["policy_export"]["execution"][
        "minimum_artifact_bytes"
    ]
    limit = float(spec["policy_export"]["parity"]["max_abs_action_error"])
    attempt_dir = Path(run["attempt_dir"])
    shapes: dict[str, tuple[list[int], list[int]]] = {}
    for kind, filename in expected["artifact_filenames"].items():
        record = artifacts[kind]
        expected_path = attempt_dir / filename
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
            "input_shape",
            "output_shape",
            "finite",
            "max_abs_action_error",
        }:
            raise SpecError(f"policy export {kind} artifact receipt is invalid")
        path = Path(record.get("path", ""))
        error = record.get("max_abs_action_error")
        if (
            path != expected_path
            or not path.is_file()
            or path.is_symlink()
            or not _valid_digest(record.get("sha256"))
            or _sha256(path) != record["sha256"]
            or record.get("size_bytes") != path.stat().st_size
            or path.stat().st_size < minimum_bytes
            or record.get("finite") is not True
            or isinstance(error, bool)
            or not isinstance(error, (int, float))
            or not math.isfinite(float(error))
            or float(error) > limit
        ):
            raise SpecError(f"policy export {kind} artifact or parity gate failed")
        input_shape = _validate_shape(
            record.get("input_shape"),
            f"artifacts.{kind}.input_shape",
        )
        output_shape = _validate_shape(
            record.get("output_shape"),
            f"artifacts.{kind}.output_shape",
        )
        shapes[kind] = (input_shape, output_shape)
    parity = result["parity"]
    contract = spec["policy_export"]["parity"]
    if not isinstance(parity, dict) or set(parity) != {
        "sample_count",
        "observation_batch_sha256",
        "native_output_sha256",
        "history_contract",
        "normalization_contract",
    }:
        raise SpecError("policy export parity receipt is invalid")
    if (
        isinstance(parity.get("sample_count"), bool)
        or not isinstance(parity.get("sample_count"), int)
        or parity["sample_count"] < contract["minimum_samples"]
        or not _valid_digest(parity.get("observation_batch_sha256"))
        or not _valid_digest(parity.get("native_output_sha256"))
        or parity.get("history_contract") != contract["history_contract"]
        or parity.get("normalization_contract")
        != contract["normalization_contract"]
    ):
        raise SpecError("policy export parity evidence does not match approval")
    sample_count = parity["sample_count"]
    if (
        len({
            (tuple(input_shape), tuple(output_shape))
            for input_shape, output_shape in shapes.values()
        })
        != 1
        or any(
            input_shape[0] != sample_count
            or output_shape[0] != sample_count
            for input_shape, output_shape in shapes.values()
        )
    ):
        raise SpecError("policy export artifact parity shapes are inconsistent")
    return result


def _record_failure(
    spec: dict[str, Any],
    state: dict[str, Any],
    run: dict[str, Any],
    reason: str,
    persist: PersistCallback | None = None,
) -> None:
    run["failure_reason"] = reason
    run["finished_at"] = time.time()
    if run["attempts"] <= spec["policy_export"]["execution"][
        "max_retries_per_run"
    ]:
        run["status"] = "pending"
        run["pid"] = None
        run["process_group"] = None
        run["process_start_ticks"] = None
        run["launched_at"] = None
    else:
        run["status"] = "failed"
        state["stage"] = "blocked"
    state["updated_at"] = time.time()
    if persist is not None:
        persist(state, "policy-export-attempt-failed")


def _load_launch_receipt(run: dict[str, Any]) -> dict[str, Any]:
    path = Path(run["launch_receipt_path"])
    receipt = _load_object(path, "policy export launch receipt")
    if (
        not path.is_file()
        or path.is_symlink()
        or _sha256(path) != run.get("launch_receipt_sha256")
        or receipt.get("version") != 1
        or receipt.get("run_id") != run["run_id"]
        or receipt.get("attempt") != run["attempts"]
        or receipt.get("argv") != run["argv"]
        or receipt.get("process_group") != receipt.get("pid")
        or not isinstance(receipt.get("pid"), int)
        or not isinstance(receipt.get("process_start_ticks"), int)
    ):
        raise SpecError("policy export launch receipt identity is invalid")
    return receipt


def _restore_launch_receipt(run: dict[str, Any], receipt: dict[str, Any]) -> None:
    run["pid"] = receipt["pid"]
    run["process_group"] = receipt["process_group"]
    run["process_start_ticks"] = receipt["process_start_ticks"]
    run["launched_at"] = receipt["launched_at"]
    run["status"] = "running"


def _finalize_manifest(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> None:
    if not state["runs"] or not all(
        run["status"] == "completed" for run in state["runs"].values()
    ):
        return
    candidates: list[dict[str, Any]] = []
    receipts: dict[str, Any] = {}
    expected_by_id = {run["run_id"]: run for run in plan["runs"]}
    for run_id in sorted(state["runs"]):
        run = state["runs"][run_id]
        expected = expected_by_id[run_id]
        artifacts = run["artifacts"]
        candidates.append(
            {
                "candidate_id": expected["candidate_id"],
                "checkpoint_path": expected["checkpoint_path"],
                "checkpoint_sha256": expected["checkpoint_sha256"],
                "artifacts": {
                    kind: record["path"]
                    for kind, record in artifacts.items()
                },
                "artifact_sha256": {
                    kind: record["sha256"]
                    for kind, record in artifacts.items()
                },
            }
        )
        receipts[expected["candidate_id"]] = {
            "export_run_id": run_id,
            "result_path": run["result_path"],
            "result_sha256": run["result_sha256"],
            "parity": run["parity"],
        }
    manifest = {
        "version": 1,
        "session_sha256": state["session_sha256"],
        "plan_sha256": state["plan_sha256"],
        "training_ranking_sha256": plan["training_ranking_sha256"],
        "checkpoint_inventory_sha256": plan["checkpoint_inventory_sha256"],
        "training_run_id": plan["training_run_id"],
        "algorithm": plan["algorithm"],
        "worker_id": plan["worker_id"],
        "candidates": candidates,
        "receipts": receipts,
    }
    path = Path(spec["policy_export"]["output_dir"]) / "export_manifest.json"
    _write_immutable(path, manifest)
    state["manifest_path"] = str(path)
    state["manifest_sha256"] = _sha256(path)
    state["stage"] = "completed"


def reconcile(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    expected_by_id = {run["run_id"]: run for run in plan["runs"]}
    now = time.time()
    for run in state["runs"].values():
        if run["status"] not in ACTIVE_STATUSES:
            continue
        previous = run["status"]
        if previous == "launching":
            try:
                receipt = _load_launch_receipt(run)
            except SpecError as exc:
                if _find_process_with_exact_token(run["run_id"]) is not None:
                    run["status"] = "failed"
                    state["stage"] = "blocked"
                    run["failure_reason"] = (
                        "orphaned policy export launch has no valid receipt: "
                        f"{exc}"
                    )
                else:
                    _record_failure(spec, state, run, str(exc))
                continue
            _restore_launch_receipt(run, receipt)
            previous = "running"
        if _process_is_exact(run):
            elapsed = now - float(run["launched_at"])
            timeout = spec["policy_export"]["run_timeout_minutes"] * 60
            if run["status"] == "running" and elapsed >= timeout:
                _stop_exact_process(run)
                run["status"] = "stopping_timeout"
                run["stop_requested_at"] = now
                run["stop_reason"] = "approved_policy_export_timeout"
            elif run["status"] in {"stopping_timeout", "stopping_forced"}:
                grace = spec["policy_export"]["execution"]["stop_grace_seconds"]
                if (
                    run["status"] != "stopping_forced"
                    and now - float(run["stop_requested_at"]) >= grace
                ):
                    _stop_exact_process(run, signal.SIGKILL)
                    run["status"] = "stopping_forced"
            continue
        pid = run.get("pid")
        if isinstance(pid, int):
            process = _ACTIVE_CHILDREN.pop(pid, None)
            if process is not None:
                process.wait(timeout=1)
        if previous != "running":
            _record_failure(
                spec,
                state,
                run,
                run.get("stop_reason") or "policy export process stopped",
            )
            continue
        try:
            result = _load_valid_result(
                spec,
                expected_by_id[run["run_id"]],
                run,
            )
        except (OSError, SpecError, ValueError) as exc:
            _record_failure(spec, state, run, str(exc))
            continue
        run["status"] = "completed"
        run["finished_at"] = now
        run["failure_reason"] = None
        run["result_sha256"] = _sha256(Path(run["result_path"]))
        run["artifacts"] = result["artifacts"]
        run["parity"] = result["parity"]
    if any(run["status"] == "failed" for run in state["runs"].values()):
        state["stage"] = "blocked"
    _finalize_manifest(spec, plan, state)
    state["updated_at"] = time.time()
    return state


def launch_next(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    persist: PersistCallback | None = None,
) -> dict[str, Any]:
    if any(run["status"] in ACTIVE_STATUSES for run in state["runs"].values()):
        raise SpecError("a policy export run is already active")
    pending = next(
        (run for run in state["runs"].values() if run["status"] == "pending"),
        None,
    )
    if pending is None:
        raise SpecError("no pending policy export run is available")
    expected = next(
        run for run in plan["runs"] if run["run_id"] == pending["run_id"]
    )
    if _find_process_with_exact_token(pending["run_id"]) is not None:
        raise SpecError("policy export run ID is already active")
    checkpoint = Path(expected["checkpoint_path"])
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or _sha256(checkpoint) != expected["checkpoint_sha256"]
    ):
        raise SpecError("approved policy export checkpoint changed")
    pending["resource_report"] = _resource_preflight(spec)
    gpu_index = spec["policy_export"]["gpu_index"]
    if spec["policy_export"]["require_idle_gpu"] and not _gpu_idle(gpu_index):
        raise SpecError("approved policy export GPU is not idle")
    lock_stream = gpu_lock_path(gpu_index).open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise SpecError("another skill executor holds the approved GPU") from exc
    if spec["policy_export"]["require_idle_gpu"] and not _gpu_idle(gpu_index):
        lock_stream.close()
        raise SpecError("approved policy export GPU became busy before launch")
    os.set_inheritable(lock_stream.fileno(), True)
    attempt = pending["attempts"] + 1
    run_dir = Path(pending["run_dir"])
    attempt_dir = run_dir / f"attempt-{attempt}"
    if attempt_dir.exists():
        lock_stream.close()
        raise SpecError("policy export attempt directory already exists")
    command, _ = _render_command(spec, expected, attempt_dir)
    pending.update(
        {
            "status": "launching",
            "attempts": attempt,
            "attempt_dir": str(attempt_dir),
            "result_path": str(attempt_dir / "result.json"),
            "log_path": str(attempt_dir / "policy_export.log"),
            "launch_receipt_path": str(attempt_dir / "launch_receipt.json"),
            "launch_receipt_sha256": None,
            "argv": command,
            "pid": None,
            "process_group": None,
            "process_start_ticks": None,
            "reserved_at": time.time(),
            "launched_at": None,
            "stop_requested_at": None,
            "stop_reason": None,
            "finished_at": None,
            "failure_reason": None,
        }
    )
    state["updated_at"] = time.time()
    if persist is not None:
        persist(state, "reserve-policy-export-attempt")
    log_stream = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir.mkdir()
        log_stream = Path(pending["log_path"]).open("ab")
        process = subprocess.Popen(
            command,
            cwd=spec["training"]["cwd"],
            stdin=subprocess.DEVNULL,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(lock_stream.fileno(),),
        )
    except (OSError, ValueError) as exc:
        if log_stream is not None:
            log_stream.close()
        lock_stream.close()
        _record_failure(
            spec,
            state,
            pending,
            f"failed to launch policy export: {exc}",
            persist,
        )
        return state
    log_stream.close()
    lock_stream.close()
    start_ticks = _process_start_ticks(process.pid)
    if start_ticks is None:
        process.terminate()
        process.wait(timeout=5)
        _record_failure(
            spec,
            state,
            pending,
            "policy export process identity could not be recorded",
            persist,
        )
        return state
    launched_at = time.time()
    pending.update(
        {
            "pid": process.pid,
            "process_group": process.pid,
            "process_start_ticks": start_ticks,
            "launched_at": launched_at,
        }
    )
    receipt = {
        "version": 1,
        "run_id": pending["run_id"],
        "attempt": attempt,
        "pid": process.pid,
        "process_group": process.pid,
        "process_start_ticks": start_ticks,
        "launched_at": launched_at,
        "argv": command,
    }
    try:
        receipt_path = Path(pending["launch_receipt_path"])
        _atomic_write(receipt_path, receipt)
        pending["launch_receipt_sha256"] = _sha256(receipt_path)
    except (OSError, ValueError) as exc:
        process.terminate()
        process.wait(timeout=5)
        _record_failure(
            spec,
            state,
            pending,
            f"policy export launch receipt failed: {exc}",
            persist,
        )
        return state
    pending["status"] = "running"
    _ACTIVE_CHILDREN[process.pid] = process
    state["updated_at"] = time.time()
    if persist is not None:
        persist(state, "policy-export-launch-started")
    return state


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(run["status"] for run in state["runs"].values())
    return {
        "version": state["version"],
        "stage": state["stage"],
        "counts": dict(sorted(counts.items())),
        "manifest_path": state["manifest_path"],
        "active": [
            {
                "run_id": run["run_id"],
                "status": run["status"],
                "pid": run["pid"],
                "attempts": run["attempts"],
            }
            for run in state["runs"].values()
            if run["status"] in ACTIVE_STATUSES
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("plan")
    parser.add_argument(
        "--action",
        required=True,
        choices={
            "initialize",
            "launch-next",
            "reconcile",
            "recover-state",
            "status",
        },
    )
    args = parser.parse_args()
    session_path = Path(args.session).resolve()
    plan_path = Path(args.plan).resolve()
    try:
        spec = load_and_validate(session_path)
        if not isinstance(spec.get("policy_export"), dict):
            raise SpecError("session does not authorize policy export")
        if (
            args.action != "status"
            and spec["policy_export"]["mode"] != "execute"
        ):
            raise SpecError(
                "policy export mutation requires policy_export.mode=execute"
            )
        plan = _load_object(plan_path, "policy export plan")
        validate_plan(spec, plan)
        if args.action == "status":
            state = initialize_state(
                spec,
                session_path,
                plan,
                plan_path,
            )
        else:
            with export_state_lock(spec):
                if args.action == "recover-state":
                    state = recover_state_from_journal(
                        spec,
                        session_path,
                        plan_path,
                    )
                else:
                    state = initialize_state(
                        spec,
                        session_path,
                        plan,
                        plan_path,
                    )
                    if args.action == "initialize":
                        _persist_state(spec, state, "initialize-policy-export")
                    elif args.action == "reconcile":
                        state = reconcile(spec, plan, state)
                        _persist_state(spec, state, "reconcile-policy-export")
                    elif args.action == "launch-next":
                        state = reconcile(spec, plan, state)
                        _persist_state(
                            spec,
                            state,
                            "reconcile-before-policy-export-launch",
                        )

                        def persist(value: dict[str, Any], action: str) -> None:
                            _persist_state(spec, value, action)

                        state = launch_next(spec, plan, state, persist)
        summary = state_summary(state)
        summary["state_path"] = str(_state_path(spec))
    except SpecError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

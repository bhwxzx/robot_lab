#!/usr/bin/env python3
"""Idempotently launch and reconcile one authorized training run at a time."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from build_trial_plan import build_confirmation_runs, build_plan
from collect_training_health import tensorboard_progress
from detect_training_anomalies import detect_anomalies
from execution_safety import gpu_lock_path
from rank_trials import select_confirmation_candidates
from validate_effective_config import validate_effective_config
from validate_session_spec import SpecError, load_and_validate


STATE_FILENAME = "execution_state.json"
STATE_LOCK_FILENAME = ".execution-state.lock"
JOURNAL_FILENAME = "execution_events.jsonl"
ACTIVE_STATUSES = {
    "launching",
    "running",
    "stopping_quality_rule",
    "stopping_trial_timeout",
    "stopping_campaign_timeout",
    "stopping_forced",
}
_ACTIVE_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}


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
        raise SpecError(f"{label} file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    temporary.write_text(encoded + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _state_path(spec: dict[str, Any]) -> Path:
    state_dir = Path(spec["execution"]["state_dir"])
    if state_dir.exists() and (not state_dir.is_dir() or state_dir.is_symlink()):
        raise SpecError("execution.state_dir must be a regular directory")
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / STATE_FILENAME


@contextmanager
def execution_state_lock(spec: dict[str, Any]) -> Iterator[None]:
    """Serialize state transitions across independent scheduler invocations."""
    state_path = _state_path(spec)
    lock_path = state_path.parent / STATE_LOCK_FILENAME
    stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError(
                "another executor invocation holds the execution-state lock"
            ) from exc
        yield
    finally:
        stream.close()


def _journal_path(spec: dict[str, Any]) -> Path:
    return _state_path(spec).parent / JOURNAL_FILENAME


def _read_journal(
    path: Path,
    allow_truncated_tail: bool = False,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise SpecError("execution event journal must be a regular file")
    raw_text = path.read_text(encoding="utf-8")
    lines = raw_text.splitlines()
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if (
                allow_truncated_tail
                and line_number == len(lines)
                and not raw_text.endswith("\n")
            ):
                break
            raise SpecError(
                f"invalid execution journal JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(event, dict):
            raise SpecError(f"execution journal line {line_number} is not an object")
        claimed_hash = event.get("event_sha256")
        unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
        invalid = (
            event.get("version") != 1
            or event.get("sequence") != line_number
            or event.get("previous_event_sha256") != previous_hash
            or not isinstance(claimed_hash, str)
            or _object_sha256(unsigned) != claimed_hash
            or not isinstance(event.get("state"), dict)
            or event.get("state_sha256") != _object_sha256(event["state"])
        )
        if invalid:
            if (
                allow_truncated_tail
                and line_number == len(lines)
                and not raw_text.endswith("\n")
            ):
                break
            raise SpecError(
                f"execution journal integrity check failed at line {line_number}"
            )
        previous_hash = claimed_hash
        events.append(event)
    return events


def _rewrite_journal(path: Path, events: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = "".join(
        _canonical_bytes(event).decode("utf-8") + "\n"
        for event in events
    )
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _persist_state(
    spec: dict[str, Any],
    state_path: Path,
    state: dict[str, Any],
    action: str,
) -> None:
    """Append a recoverable, hash-chained snapshot before replacing state."""
    journal_path = _journal_path(spec)
    events = _read_journal(journal_path)
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
    _atomic_write(state_path, state)


def recover_state_from_journal(
    spec: dict[str, Any],
    session_path: Path,
    plan_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Restore the latest validated state snapshot from the event journal."""
    journal_path = _journal_path(spec)
    events = _read_journal(journal_path, allow_truncated_tail=True)
    if not events:
        raise SpecError("execution event journal is empty")
    canonical_journal = "".join(
        _canonical_bytes(event).decode("utf-8") + "\n"
        for event in events
    )
    if journal_path.read_text(encoding="utf-8") != canonical_journal:
        _rewrite_journal(journal_path, events)
    state = events[-1]["state"]
    if (
        state.get("session_sha256") != _sha256(session_path)
        or state.get("plan_sha256") != _sha256(plan_path)
    ):
        raise SpecError("journal state is bound to different session or plan")
    state_path = _state_path(spec)
    _atomic_write(state_path, state)
    return state_path, state


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _load_object(path, "trial plan")
    if (
        plan.get("version") != 4
        or not isinstance(plan.get("runs"), list)
        or not isinstance(plan.get("trials"), list)
        or not isinstance(plan.get("stages"), dict)
    ):
        raise SpecError("version-6 execution requires a version-4 staged trial plan")
    run_ids = [run.get("run_id") for run in plan["runs"]]
    if (
        any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
        or len(run_ids) != len(set(run_ids))
    ):
        raise SpecError("trial plan contains invalid or duplicate run IDs")
    return plan


def _validate_plan_against_spec(
    spec: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    expected = build_plan(spec)
    if plan != expected:
        raise SpecError(
            "trial plan does not exactly match the validated session; "
            "rebuild it with build_trial_plan.py"
        )


def _run_record(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    run_dir = Path(spec["execution"]["state_dir"]) / "runs" / run["run_id"]
    return {
        **run,
        "status": "pending",
        "attempts": 0,
        "pid": None,
        "process_group": None,
        "process_start_ticks": None,
        "reserved_at": None,
        "launched_at": None,
        "finished_at": None,
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "training.log"),
        "result_path": str(run_dir / "result.json"),
        "summary_path": str(run_dir / "training_summary.json"),
        "effective_config_path": str(run_dir / "effective_config.json"),
        "terminal_path": str(run_dir / "terminal.json"),
        "adapter_contract_path": str(run_dir / "adapter_contract.json"),
        "launch_receipt_path": str(run_dir / "launch_receipt.json"),
        "launch_receipt_sha256": None,
        "reproducibility_path": str(run_dir / "reproducibility.json"),
        "reproducibility_sha256": None,
        "config_report": None,
        "quality_report": None,
        "resource_report": None,
        "terminal_receipt": None,
        "failure_reason": None,
        "stop_reason": None,
        "stop_requested_at": None,
        "argv": None,
    }


def initialize_state(
    spec: dict[str, Any],
    session_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create or load an execution state bound to exact session and plan hashes."""
    _validate_plan_against_spec(spec, plan)
    path = _state_path(spec)
    session_hash = _sha256(session_path)
    plan_hash = _sha256(plan_path)
    if path.exists():
        state = _load_object(path, "execution state")
        if (
            state.get("session_sha256") != session_hash
            or state.get("plan_sha256") != plan_hash
        ):
            raise SpecError(
                "existing execution state is bound to different session or plan"
            )
        return path, state
    state = {
        "version": 1,
        "session_path": str(session_path.resolve()),
        "session_sha256": session_hash,
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": plan_hash,
        "algorithm": spec["algorithm"],
        "training_run_id": spec["training"]["run_id"],
        "created_at": time.time(),
        "updated_at": time.time(),
        "stage": "screening",
        "screening_selection": [],
        "selection_failure_reason": None,
        "runs": {
            run["run_id"]: _run_record(spec, run) for run in plan["runs"]
        },
    }
    _atomic_write(path, state)
    return path, state


def _process_start_ticks(pid: int) -> int | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text(
            encoding="utf-8"
        ).split()
        return int(fields[21])
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def _process_argv(pid: int) -> list[str] | None:
    try:
        encoded = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, OSError):
        return None
    if not encoded:
        return None
    return [
        os.fsdecode(token)
        for token in encoded.rstrip(b"\0").split(b"\0")
    ]


def _process_is_exact(run: dict[str, Any]) -> bool:
    pid = run.get("pid")
    start = run.get("process_start_ticks")
    argv = run.get("argv")
    if (
        not isinstance(pid, int)
        or not isinstance(start, int)
        or not isinstance(argv, list)
        or not all(isinstance(token, str) for token in argv)
        or run.get("process_group") != pid
    ):
        return False
    return _process_start_ticks(pid) == start and _process_argv(pid) == argv


def _load_launch_receipt(run: dict[str, Any]) -> dict[str, Any]:
    path = Path(run["launch_receipt_path"])
    if not path.is_file() or path.is_symlink():
        raise SpecError("launch receipt must be a regular file")
    receipt = _load_object(path, "launch receipt")
    recorded_hash = run.get("launch_receipt_sha256")
    if isinstance(recorded_hash, str) and _sha256(path) != recorded_hash:
        raise SpecError("launch receipt hash changed")
    if (
        receipt.get("version") != 1
        or receipt.get("run_id") != run["run_id"]
        or receipt.get("trial_id") != run["trial_id"]
        or receipt.get("attempt") != run["attempts"]
        or receipt.get("argv") != run["argv"]
        or receipt.get("process_group") != receipt.get("pid")
        or not isinstance(receipt.get("pid"), int)
        or not isinstance(receipt.get("process_start_ticks"), int)
        or not isinstance(receipt.get("launched_at"), (int, float))
    ):
        raise SpecError("launch receipt identity is invalid")
    return receipt


def _restore_launch_receipt(
    run: dict[str, Any],
    receipt: dict[str, Any],
) -> None:
    run["pid"] = receipt["pid"]
    run["process_group"] = receipt["process_group"]
    run["process_start_ticks"] = receipt["process_start_ticks"]
    run["launched_at"] = receipt["launched_at"]
    run["launch_receipt_sha256"] = _sha256(
        Path(run["launch_receipt_path"])
    )
    run["status"] = "running"


PersistCallback = Callable[[dict[str, Any], str], None]


def _record_launch_failure(
    spec: dict[str, Any],
    state: dict[str, Any],
    run: dict[str, Any],
    reason: str,
    persist: PersistCallback | None,
) -> dict[str, Any]:
    run["failure_reason"] = reason
    run["finished_at"] = time.time()
    if run["attempts"] <= spec["execution"]["max_retries_per_run"]:
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
        persist(state, "launch-failed")
    return state


def _find_process_with_exact_token(token: str) -> int | None:
    for proc_entry in Path("/proc").iterdir():
        if not proc_entry.name.isdigit():
            continue
        pid = int(proc_entry.name)
        if pid == os.getpid():
            continue
        argv = _process_argv(pid)
        if argv is not None and token in argv:
            return pid
    return None


def _gpu_idle(gpu_index: int) -> bool:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise SpecError(f"cannot verify approved GPU is idle: {exc}") from exc
    return not any(line.strip() for line in result.stdout.splitlines())


def _resource_preflight(spec: dict[str, Any]) -> dict[str, Any] | None:
    limits = spec["execution"].get("resource_limits")
    if limits is None:
        return None
    state_dir = Path(spec["execution"]["state_dir"])
    free_gb = shutil.disk_usage(state_dir).free / 1_000_000_000
    if free_gb < float(limits["min_free_disk_gb"]):
        raise SpecError(
            f"free disk {free_gb:.2f} GB is below the approved minimum "
            f"{limits['min_free_disk_gb']} GB"
        )
    command = [
        "nvidia-smi",
        f"--id={spec['execution']['gpu_index']}",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
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
        raise SpecError(f"cannot verify approved GPU health: {exc}") from exc
    if temperature > limits["max_gpu_temperature_c"]:
        raise SpecError(
            f"GPU temperature {temperature} C exceeds approved maximum "
            f"{limits['max_gpu_temperature_c']} C"
        )
    return {
        "checked_at": time.time(),
        "free_disk_gb": free_gb,
        "gpu_temperature_c": temperature,
    }


def _run_evidence_command(
    command: list[str],
    cwd: Path | None = None,
) -> bytes:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise SpecError(
            f"cannot capture reproducibility evidence for {command[0]}: {exc}"
        ) from exc
    return result.stdout


def _git_reproducibility(
    cwd: Path,
    capture_diff: bool,
) -> dict[str, Any]:
    root = Path(
        os.fsdecode(
            _run_evidence_command(
                ["git", "rev-parse", "--show-toplevel"],
                cwd,
            )
        ).strip()
    ).resolve()
    if not root.is_dir() or root.is_symlink():
        raise SpecError("training Git root is not a regular directory")
    head = os.fsdecode(
        _run_evidence_command(["git", "rev-parse", "HEAD"], root)
    ).strip()
    status = _run_evidence_command(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ],
        root,
    )
    diff = (
        _run_evidence_command(["git", "diff", "--binary", "HEAD", "--"], root)
        if capture_diff
        else None
    )
    return {
        "root": str(root),
        "head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if diff is not None else None
        ),
    }


def _gpu_reproducibility(gpu_index: int) -> dict[str, Any]:
    output = os.fsdecode(
        _run_evidence_command(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    ).strip()
    parts = [part.strip() for part in output.split(",", maxsplit=2)]
    if len(parts) != 3 or not all(parts):
        raise SpecError("GPU reproducibility query returned an invalid row")
    return {
        "index": gpu_index,
        "name": parts[0],
        "uuid": parts[1],
        "driver_version": parts[2],
    }


def _build_reproducibility_manifest(
    spec: dict[str, Any],
    state: dict[str, Any],
    run: dict[str, Any],
    command: list[str],
) -> dict[str, Any] | None:
    contract = spec["execution"].get("reproducibility")
    if contract is None or not contract["enabled"]:
        return None
    training_cwd = Path(spec["training"]["cwd"])
    inputs = []
    for path_value in contract["input_paths"]:
        path = Path(path_value)
        if not path.is_file() or path.is_symlink():
            raise SpecError(
                f"reproducibility input must be a regular file: {path}"
            )
        inputs.append(
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    packages: dict[str, str] = {}
    for package_name in contract["package_names"]:
        try:
            packages[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SpecError(
                f"reproducibility package is not installed: {package_name}"
            ) from exc
    adapter_contract = (
        _adapter_contract(spec, run)
        if spec["execution"].get("adapter") is not None
        else None
    )
    return {
        "version": 1,
        "captured_at": time.time(),
        "run_id": run["run_id"],
        "trial_id": run["trial_id"],
        "stage": run["stage"],
        "seed": run["seed"],
        "session_sha256": state["session_sha256"],
        "plan_sha256": state["plan_sha256"],
        "algorithm": spec["algorithm"],
        "training_cwd": str(training_cwd),
        "training_argv": spec["training"]["command"],
        "executor_argv": command,
        "adapter_contract_sha256": (
            _object_sha256(adapter_contract)
            if adapter_contract is not None
            else None
        ),
        "git": _git_reproducibility(
            training_cwd,
            contract["capture_git_diff"],
        ),
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        },
        "gpu": (
            _gpu_reproducibility(spec["execution"]["gpu_index"])
            if contract["capture_gpu"]
            else None
        ),
        "inputs": inputs,
    }


def _campaign_timed_out(spec: dict[str, Any], state: dict[str, Any]) -> bool:
    limits = spec["execution"].get("resource_limits")
    if limits is None:
        return False
    return (
        time.time() - float(state["created_at"])
        >= limits["campaign_timeout_minutes"] * 60
    )


def _adapter_contract(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    adapter = spec["execution"]["adapter"]
    required_metrics = {
        item["metric"] for item in spec["tuning"]["objectives"]
    }
    required_metrics.update(
        item["metric"] for item in spec["tuning"]["constraints"]
    )
    return {
        "version": 1,
        "adapter_id": adapter["id"],
        "profile_id": spec["algorithm"]["profile_id"],
        "training_argv": spec["training"]["command"],
        "training_cwd": spec["training"]["cwd"],
        "parameter_cli_map": adapter["parameter_cli_map"],
        "summary_last": adapter["summary_last"],
        "required_metrics": sorted(required_metrics),
        "require_checkpoint": adapter["require_checkpoint"],
        "run_id": run["run_id"],
        "trial_id": run["trial_id"],
        "stage": run["stage"],
        "seed": run["seed"],
    }


def _runtime_config_values(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    adapter = spec["execution"].get("adapter")
    if adapter is None:
        return {}
    identities = {
        "seed": run["seed"],
        "run_id": run["run_id"],
    }
    return {
        path: identities[identity]
        for path, identity in adapter["runtime_config_paths"].items()
    }


def _render_command(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> list[str]:
    replacements = {
        "adapter_contract_path": run["adapter_contract_path"],
        "effective_config_path": run["effective_config_path"],
        "gpu_index": str(spec["execution"]["gpu_index"]),
        "log_path": run["log_path"],
        "overrides_json": json.dumps(
            run["overrides"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        "result_path": run["result_path"],
        "run_dir": run["run_dir"],
        "run_id": run["run_id"],
        "seed": str(run["seed"]),
        "stage": run["stage"],
        "summary_path": run["summary_path"],
        "terminal_path": run["terminal_path"],
        "trial_id": run["trial_id"],
    }
    return [
        token.format_map(replacements)
        for token in spec["execution"]["run_command"]
    ]


def _collect_completed_results(state: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for run in state["runs"].values():
        if run["status"] != "completed":
            continue
        result = _load_object(Path(run["result_path"]), "run result")
        results.append(result)
    return results


def _append_confirmation_runs(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> None:
    screening_runs = [
        run for run in state["runs"].values() if run["stage"] == "screening"
    ]
    if not screening_runs or any(
        run["status"] != "completed" for run in screening_runs
    ):
        return
    if state["stage"] != "screening":
        return
    try:
        selected = select_confirmation_candidates(
            spec,
            _collect_completed_results(state),
        )
        confirmation = build_confirmation_runs(spec, plan, selected)
    except SpecError as exc:
        state["stage"] = "blocked"
        state["selection_failure_reason"] = str(exc)
        return
    for run in confirmation:
        if run["run_id"] in state["runs"]:
            raise SpecError("confirmation run ID collides with existing state")
        state["runs"][run["run_id"]] = _run_record(spec, run)
    state["screening_selection"] = selected
    state["stage"] = "confirmation"


def _load_valid_result(run: dict[str, Any]) -> dict[str, Any]:
    result = _load_object(Path(run["result_path"]), "run result")
    if (
        result.get("trial_id") != run["trial_id"]
        or result.get("seed") != run["seed"]
        or result.get("status") != "completed"
        or not isinstance(result.get("metrics"), dict)
    ):
        raise SpecError("run result identity or completion status is invalid")
    for metric, value in result["metrics"].items():
        if (
            not isinstance(metric, str)
            or not metric
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise SpecError("run result metrics must be finite numbers")
    return result


def _stop_exact_process(
    run: dict[str, Any],
    requested_signal: signal.Signals = signal.SIGTERM,
) -> None:
    if not _process_is_exact(run):
        raise SpecError("refusing to stop a process whose identity changed")
    pid = run["pid"]
    pgid = run["process_group"]
    try:
        current_group = os.getpgid(pid)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise SpecError("cannot inspect the recorded process group") from exc
    if current_group != pgid:
        raise SpecError("refusing to stop a process outside its recorded group")
    try:
        os.killpg(pgid, requested_signal)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise SpecError("cannot signal the recorded process group") from exc


def _load_valid_terminal(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    terminal = _load_object(Path(run["terminal_path"]), "adapter terminal receipt")
    if (
        terminal.get("version") != 1
        or terminal.get("adapter_id") != spec["execution"]["adapter"]["id"]
        or terminal.get("run_id") != run["run_id"]
        or terminal.get("trial_id") != run["trial_id"]
        or terminal.get("stage") != run["stage"]
        or terminal.get("seed") != run["seed"]
        or terminal.get("status") != "completed"
        or terminal.get("exit_code") != 0
    ):
        reason = terminal.get("failure_reason")
        suffix = f": {reason}" if isinstance(reason, str) and reason else ""
        raise SpecError(f"adapter terminal receipt is not successful{suffix}")
    checkpoint = terminal.get("checkpoint")
    if (
        checkpoint is None
        and spec["execution"]["adapter"]["require_checkpoint"]
    ):
        raise SpecError("adapter terminal receipt is missing required checkpoint")
    if checkpoint is not None:
        if (
            not isinstance(checkpoint, dict)
            or not isinstance(checkpoint.get("path"), str)
            or not isinstance(checkpoint.get("sha256"), str)
        ):
            raise SpecError("adapter terminal checkpoint evidence is invalid")
        checkpoint_path = Path(checkpoint["path"])
        if (
            not checkpoint_path.is_absolute()
            or not checkpoint_path.is_file()
            or checkpoint_path.is_symlink()
            or _sha256(checkpoint_path) != checkpoint["sha256"]
        ):
            raise SpecError("adapter terminal checkpoint hash does not match")
    return terminal


def _load_valid_reproducibility(
    spec: dict[str, Any],
    state: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any] | None:
    contract = spec["execution"].get("reproducibility")
    if contract is None or not contract["enabled"]:
        return None
    path = Path(run["reproducibility_path"])
    recorded_hash = run.get("reproducibility_sha256")
    if (
        not isinstance(recorded_hash, str)
        or not path.is_file()
        or path.is_symlink()
        or _sha256(path) != recorded_hash
    ):
        raise SpecError("reproducibility manifest is missing or changed")
    manifest = _load_object(path, "reproducibility manifest")
    if (
        manifest.get("version") != 1
        or manifest.get("run_id") != run["run_id"]
        or manifest.get("trial_id") != run["trial_id"]
        or manifest.get("stage") != run["stage"]
        or manifest.get("seed") != run["seed"]
        or manifest.get("session_sha256") != state["session_sha256"]
        or manifest.get("plan_sha256") != state["plan_sha256"]
        or manifest.get("executor_argv") != run["argv"]
        or manifest.get("algorithm") != spec["algorithm"]
    ):
        raise SpecError("reproducibility manifest identity is invalid")
    return manifest


def _bootstrap_baseline(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> None:
    effective = spec["execution"]["effective_config"]
    baseline_path = Path(effective["baseline_path"])
    if baseline_path.exists():
        return
    if not effective.get("allow_baseline_bootstrap", False):
        return
    if run["trial_id"] != "baseline" or run["overrides"]:
        raise SpecError("only the unchanged baseline trial may bootstrap config")
    source_path = Path(run["effective_config_path"])
    if not source_path.is_file() or source_path.is_symlink():
        raise SpecError("baseline bootstrap source is not a regular file")
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            baseline_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError:
        return
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(source_path.read_bytes())
            target.flush()
            os.fsync(target.fileno())
    except BaseException:
        try:
            baseline_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _refresh_quality_report(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any] | None:
    summary_path = Path(run["summary_path"])
    if not summary_path.exists():
        return None
    if not summary_path.is_file() or summary_path.is_symlink():
        raise SpecError("training summary must be a regular file")
    quality = detect_anomalies(
        spec,
        summary := _load_object(summary_path, "training summary"),
    )
    live_evidence = summary.get("live_evidence")
    rsl_run_dir = (
        live_evidence.get("rsl_rl_run_dir")
        if isinstance(live_evidence, dict)
        else None
    )
    quality["progress_evidence"] = {
        "summary_last_progress": summary.get("last_progress"),
        "summary_updated_at": (
            live_evidence.get("updated_at")
            if isinstance(live_evidence, dict)
            else None
        ),
        "tensorboard": tensorboard_progress(
            Path(rsl_run_dir) if isinstance(rsl_run_dir, str) else None
        ),
    }
    run["quality_report"] = quality
    return quality


def reconcile(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile recorded child identity, quality evidence, and result files."""
    for run in state["runs"].values():
        if run["status"] not in ACTIVE_STATUSES:
            continue
        previous_status = run["status"]
        if previous_status == "launching":
            try:
                receipt = _load_launch_receipt(run)
            except SpecError as exc:
                duplicate_pid = _find_process_with_exact_token(run["run_id"])
                if duplicate_pid is not None:
                    run["status"] = "failed"
                    run["finished_at"] = time.time()
                    run["failure_reason"] = (
                        "orphaned launch has the run ID but no valid receipt; "
                        f"refusing process control: {exc}"
                    )
                    state["stage"] = "blocked"
                else:
                    _record_launch_failure(
                        spec,
                        state,
                        run,
                        f"reserved launch did not produce a valid receipt: {exc}",
                        None,
                    )
                continue
            _restore_launch_receipt(run, receipt)
            previous_status = "running"
        if _process_is_exact(run):
            quality = _refresh_quality_report(spec, run)
            if run["status"] == "running":
                stop_status: str | None = None
                stop_reason: str | None = None
                if _campaign_timed_out(spec, state):
                    stop_status = "stopping_campaign_timeout"
                    stop_reason = "approved_campaign_timeout"
                elif (
                    isinstance(run.get("launched_at"), (int, float))
                    and time.time() - float(run["launched_at"])
                    >= spec["tuning"]["trial_timeout_minutes"] * 60
                ):
                    stop_status = "stopping_trial_timeout"
                    stop_reason = "approved_trial_timeout"
                elif quality is not None and quality["stop_trial"]:
                    stop_status = "stopping_quality_rule"
                    stop_reason = "approved_quality_stop_rule"
                if stop_status is not None:
                    _stop_exact_process(run)
                    run["status"] = stop_status
                    run["stop_reason"] = stop_reason
                    run["stop_requested_at"] = time.time()
                    run["failure_reason"] = stop_reason
            else:
                limits = spec["execution"].get("resource_limits")
                grace = limits["stop_grace_seconds"] if limits is not None else 30
                requested_at = run.get("stop_requested_at")
                if (
                    run["status"] != "stopping_forced"
                    and isinstance(requested_at, (int, float))
                    and time.time() - float(requested_at) >= grace
                ):
                    _stop_exact_process(run, signal.SIGKILL)
                    run["status"] = "stopping_forced"
            continue
        if previous_status != "running":
            _ACTIVE_CHILDREN.pop(run["pid"], None)
            run["status"] = "failed"
            run["finished_at"] = time.time()
            continue
        config_path = Path(run["effective_config_path"])
        _ACTIVE_CHILDREN.pop(run["pid"], None)
        quality = _refresh_quality_report(spec, run)
        if quality is not None and quality["stop_trial"]:
            run["status"] = "failed"
            run["finished_at"] = time.time()
            run["failure_reason"] = (
                "completed run violates an approved quality stop rule"
            )
            continue
        try:
            _load_valid_reproducibility(spec, state, run)
            if spec["execution"].get("adapter") is not None:
                run["terminal_receipt"] = _load_object(
                    Path(run["terminal_path"]),
                    "adapter terminal receipt",
                )
                _load_valid_terminal(spec, run)
            _load_valid_result(run)
            _bootstrap_baseline(spec, run)
            config_report = validate_effective_config(
                spec,
                Path(spec["execution"]["effective_config"]["baseline_path"]),
                config_path,
                run["overrides"],
                _runtime_config_values(spec, run),
            )
        except SpecError as exc:
            if (
                run["attempts"]
                <= spec["execution"]["max_retries_per_run"]
            ):
                run["status"] = "pending"
                run["failure_reason"] = str(exc)
                run["pid"] = None
                run["process_group"] = None
                run["process_start_ticks"] = None
                run["launched_at"] = None
                run["stop_reason"] = None
                run["stop_requested_at"] = None
            else:
                run["status"] = "failed"
                run["finished_at"] = time.time()
                run["failure_reason"] = str(exc)
            continue
        run["status"] = "completed"
        run["finished_at"] = time.time()
        run["config_report"] = config_report
        run["failure_reason"] = None
        run["stop_reason"] = None
        run["stop_requested_at"] = None
    _append_confirmation_runs(spec, plan, state)
    if _campaign_timed_out(spec, state):
        for run in state["runs"].values():
            if run["status"] == "pending":
                run["status"] = "failed"
                run["finished_at"] = time.time()
                run["failure_reason"] = "approved_campaign_timeout"
    if state["stage"] != "blocked":
        if state["runs"] and all(
            run["status"] == "completed" for run in state["runs"].values()
        ):
            state["stage"] = "completed"
        elif any(run["status"] == "failed" for run in state["runs"].values()):
            state["stage"] = "blocked"
    state["updated_at"] = time.time()
    return state


def launch_next(
    spec: dict[str, Any],
    state: dict[str, Any],
    persist: PersistCallback | None = None,
) -> dict[str, Any]:
    """Launch one pending run after exact state, GPU, and lock checks."""
    if any(run["status"] in ACTIVE_STATUSES for run in state["runs"].values()):
        raise SpecError("an execution-state run is already active")
    if _campaign_timed_out(spec, state):
        raise SpecError("approved campaign timeout has elapsed")
    pending = next(
        (run for run in state["runs"].values() if run["status"] == "pending"),
        None,
    )
    if pending is None:
        raise SpecError("no pending run is available")
    duplicate_pid = _find_process_with_exact_token(pending["run_id"])
    if duplicate_pid is not None:
        raise SpecError(
            f"run ID {pending['run_id']} is already present in process {duplicate_pid}"
        )
    pending["resource_report"] = _resource_preflight(spec)
    gpu_index = spec["execution"]["gpu_index"]
    if spec["execution"]["require_idle_gpu"] and not _gpu_idle(gpu_index):
        raise SpecError("approved GPU is not idle")
    lock_path = gpu_lock_path(gpu_index)
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise SpecError("another skill execution holds the approved GPU lock") from exc
    os.set_inheritable(lock_stream.fileno(), True)
    run_dir = Path(pending["run_dir"])
    if run_dir.exists() and (not run_dir.is_dir() or run_dir.is_symlink()):
        lock_stream.close()
        raise SpecError("run directory must be a regular directory")
    attempt = pending["attempts"] + 1
    attempt_dir = run_dir / f"attempt-{attempt}"
    if attempt_dir.exists():
        lock_stream.close()
        raise SpecError("attempt directory already exists; refusing to reuse outputs")
    pending["log_path"] = str(attempt_dir / "training.log")
    pending["result_path"] = str(attempt_dir / "result.json")
    pending["summary_path"] = str(attempt_dir / "training_summary.json")
    pending["effective_config_path"] = str(
        attempt_dir / "effective_config.json"
    )
    pending["terminal_path"] = str(attempt_dir / "terminal.json")
    pending["adapter_contract_path"] = str(
        attempt_dir / "adapter_contract.json"
    )
    pending["launch_receipt_path"] = str(attempt_dir / "launch_receipt.json")
    pending["reproducibility_path"] = str(
        attempt_dir / "reproducibility.json"
    )
    command = _render_command(spec, pending)
    pending["status"] = "launching"
    pending["attempts"] = attempt
    pending["reserved_at"] = time.time()
    pending["finished_at"] = None
    pending["pid"] = None
    pending["process_group"] = None
    pending["process_start_ticks"] = None
    pending["launched_at"] = None
    pending["failure_reason"] = None
    pending["stop_reason"] = None
    pending["stop_requested_at"] = None
    pending["terminal_receipt"] = None
    pending["launch_receipt_sha256"] = None
    pending["reproducibility_sha256"] = None
    pending["argv"] = command
    state["updated_at"] = time.time()
    if persist is not None:
        try:
            persist(state, "reserve-attempt")
        except BaseException:
            lock_stream.close()
            raise

    log_stream = None
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_dir.mkdir()
        if spec["execution"].get("adapter") is not None:
            _atomic_write(
                Path(pending["adapter_contract_path"]),
                _adapter_contract(spec, pending),
            )
        reproducibility = _build_reproducibility_manifest(
            spec,
            state,
            pending,
            command,
        )
        if reproducibility is not None:
            reproducibility_path = Path(pending["reproducibility_path"])
            _atomic_write(reproducibility_path, reproducibility)
            pending["reproducibility_sha256"] = _sha256(
                reproducibility_path
            )
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
    except (OSError, SpecError, ValueError) as exc:
        if log_stream is not None:
            log_stream.close()
        lock_stream.close()
        return _record_launch_failure(
            spec,
            state,
            pending,
            f"failed to launch exact authorized argv: {exc}",
            persist,
        )
    log_stream.close()
    lock_stream.close()
    start_ticks = _process_start_ticks(process.pid)
    if start_ticks is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return _record_launch_failure(
            spec,
            state,
            pending,
            "launched process identity could not be recorded",
            persist,
        )
    launched_at = time.time()
    pending["pid"] = process.pid
    pending["process_group"] = process.pid
    pending["process_start_ticks"] = start_ticks
    pending["launched_at"] = launched_at
    receipt = {
        "version": 1,
        "run_id": pending["run_id"],
        "trial_id": pending["trial_id"],
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
    except SpecError as exc:
        try:
            _stop_exact_process(pending)
        except SpecError:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return _record_launch_failure(
            spec,
            state,
            pending,
            f"launch receipt could not be recorded: {exc}",
            persist,
        )
    pending["status"] = "running"
    _ACTIVE_CHILDREN[process.pid] = process
    state["updated_at"] = time.time()
    if persist is not None:
        persist(state, "launch-started")
    return state


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    active = []
    for run in state["runs"].values():
        counts[run["status"]] += 1
        if run["status"] in ACTIVE_STATUSES:
            active.append(
                {
                    "run_id": run["run_id"],
                    "pid": run["pid"],
                    "status": run["status"],
                    "stop_reason": run.get("stop_reason"),
                }
            )
    return {
        "version": state["version"],
        "stage": state["stage"],
        "counts": dict(sorted(counts.items())),
        "active": active,
        "screening_selection": state["screening_selection"],
        "selection_failure_reason": state["selection_failure_reason"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Validated version-6 tune session JSON")
    parser.add_argument("plan", help="Version-4 staged trial plan JSON")
    parser.add_argument(
        "--action",
        choices={
            "initialize",
            "reconcile",
            "launch-next",
            "status",
            "recover-state",
        },
        required=True,
    )
    args = parser.parse_args()
    session_path = Path(args.session).resolve()
    plan_path = Path(args.plan).resolve()
    try:
        spec = load_and_validate(session_path)
        if spec.get("version") != 6:
            raise SpecError("trial execution requires session version 6")
        plan = _load_plan(plan_path)
        with execution_state_lock(spec):
            if args.action == "recover-state":
                _validate_plan_against_spec(spec, plan)
                state_path, state = recover_state_from_journal(
                    spec,
                    session_path,
                    plan_path,
                )
            else:
                state_path, state = initialize_state(
                    spec,
                    session_path,
                    plan,
                    plan_path,
                )
                if args.action == "reconcile":
                    state = reconcile(spec, plan, state)
                    _persist_state(spec, state_path, state, args.action)
                elif args.action == "launch-next":
                    state = reconcile(spec, plan, state)
                    _persist_state(
                        spec,
                        state_path,
                        state,
                        "reconcile-before-launch",
                    )

                    def persist_transition(
                        transition_state: dict[str, Any],
                        action: str,
                    ) -> None:
                        _persist_state(
                            spec,
                            state_path,
                            transition_state,
                            action,
                        )

                    state = launch_next(
                        spec,
                        state,
                        persist_transition,
                    )
                elif args.action == "initialize":
                    _persist_state(spec, state_path, state, args.action)
            summary = state_summary(state)
            summary["state_path"] = str(state_path)
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

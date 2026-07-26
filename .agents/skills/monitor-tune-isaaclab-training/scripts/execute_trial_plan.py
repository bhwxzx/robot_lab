#!/usr/bin/env python3
"""Idempotently launch and reconcile one authorized training run at a time."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_trial_plan import build_confirmation_runs, build_plan
from detect_training_anomalies import detect_anomalies
from rank_trials import select_confirmation_candidates
from validate_effective_config import validate_effective_config
from validate_session_spec import SpecError, load_and_validate


STATE_FILENAME = "execution_state.json"
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


def _state_path(spec: dict[str, Any]) -> Path:
    state_dir = Path(spec["execution"]["state_dir"])
    if state_dir.exists() and (not state_dir.is_dir() or state_dir.is_symlink()):
        raise SpecError("execution.state_dir must be a regular directory")
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / STATE_FILENAME


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
        "launched_at": None,
        "finished_at": None,
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "training.log"),
        "result_path": str(run_dir / "result.json"),
        "summary_path": str(run_dir / "training_summary.json"),
        "effective_config_path": str(run_dir / "effective_config.json"),
        "config_report": None,
        "quality_report": None,
        "failure_reason": None,
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


def _render_command(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> list[str]:
    replacements = {
        "effective_config_path": run["effective_config_path"],
        "gpu_index": str(spec["execution"]["gpu_index"]),
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


def _stop_exact_process(run: dict[str, Any]) -> None:
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
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        raise SpecError("cannot signal the recorded process group") from exc


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
        _load_object(summary_path, "training summary"),
    )
    run["quality_report"] = quality
    return quality


def reconcile(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile recorded child identity, quality evidence, and result files."""
    for run in state["runs"].values():
        if run["status"] not in {"running", "stopping_quality_rule"}:
            continue
        previous_status = run["status"]
        if _process_is_exact(run):
            quality = _refresh_quality_report(spec, run)
            if (
                quality is not None
                and quality["stop_trial"]
                and run["status"] == "running"
            ):
                _stop_exact_process(run)
                run["status"] = "stopping_quality_rule"
                run["failure_reason"] = "approved_quality_stop_rule"
            continue
        if previous_status == "stopping_quality_rule":
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
            _load_valid_result(run)
            config_report = validate_effective_config(
                spec,
                Path(spec["execution"]["effective_config"]["baseline_path"]),
                config_path,
                run["overrides"],
            )
        except SpecError as exc:
            if (
                previous_status != "stopping_quality_rule"
                and run["attempts"]
                <= spec["execution"]["max_retries_per_run"]
            ):
                run["status"] = "pending"
                run["failure_reason"] = str(exc)
                run["pid"] = None
                run["process_group"] = None
                run["process_start_ticks"] = None
            else:
                run["status"] = "failed"
                run["finished_at"] = time.time()
                run["failure_reason"] = str(exc)
            continue
        run["status"] = "completed"
        run["finished_at"] = time.time()
        run["config_report"] = config_report
        run["failure_reason"] = None
    _append_confirmation_runs(spec, plan, state)
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
) -> dict[str, Any]:
    """Launch one pending run after exact state, GPU, and lock checks."""
    if any(
        run["status"] in {"running", "stopping_quality_rule"}
        for run in state["runs"].values()
    ):
        raise SpecError("an execution-state run is already active")
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
    gpu_index = spec["execution"]["gpu_index"]
    if spec["execution"]["require_idle_gpu"] and not _gpu_idle(gpu_index):
        raise SpecError("approved GPU is not idle")
    lock_path = Path(spec["execution"]["state_dir"]) / f".gpu-{gpu_index}.lock"
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
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt = pending["attempts"] + 1
    attempt_dir = run_dir / f"attempt-{attempt}"
    if attempt_dir.exists():
        lock_stream.close()
        raise SpecError("attempt directory already exists; refusing to reuse outputs")
    attempt_dir.mkdir()
    pending["log_path"] = str(attempt_dir / "training.log")
    pending["result_path"] = str(attempt_dir / "result.json")
    pending["summary_path"] = str(attempt_dir / "training_summary.json")
    pending["effective_config_path"] = str(
        attempt_dir / "effective_config.json"
    )
    command = _render_command(spec, pending)
    log_stream = Path(pending["log_path"]).open("ab")
    try:
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
        log_stream.close()
        lock_stream.close()
        raise SpecError(f"failed to launch exact authorized argv: {exc}") from exc
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
        raise SpecError("launched process identity could not be recorded")
    pending["status"] = "running"
    pending["attempts"] = attempt
    pending["pid"] = process.pid
    pending["process_group"] = process.pid
    pending["process_start_ticks"] = start_ticks
    pending["launched_at"] = time.time()
    pending["failure_reason"] = None
    pending["argv"] = command
    _ACTIVE_CHILDREN[process.pid] = process
    state["updated_at"] = time.time()
    return state


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    active = []
    for run in state["runs"].values():
        counts[run["status"]] += 1
        if run["status"] in {"running", "stopping_quality_rule"}:
            active.append(
                {
                    "run_id": run["run_id"],
                    "pid": run["pid"],
                    "status": run["status"],
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
        choices={"initialize", "reconcile", "launch-next", "status"},
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
        state_path, state = initialize_state(
            spec,
            session_path,
            plan,
            plan_path,
        )
        if args.action in {"reconcile", "launch-next"}:
            state = reconcile(spec, plan, state)
        if args.action == "launch-next":
            state = launch_next(spec, state)
        if args.action != "status":
            _atomic_write(state_path, state)
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

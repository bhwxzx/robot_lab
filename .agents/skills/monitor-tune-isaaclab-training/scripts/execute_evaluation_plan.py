#!/usr/bin/env python3
"""Execute an approved policy-evaluation matrix transactionally, one cell at a time."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from execution_safety import gpu_lock_path
from execute_trial_plan import (
    _find_process_with_exact_token,
    _gpu_idle,
    _process_is_exact,
    _process_start_ticks,
    _stop_exact_process,
)
from validate_policy_evaluation import load_evaluation_plan
from validate_session_spec import SpecError, load_and_validate


STATE_FILENAME = "evaluation_state.json"
JOURNAL_FILENAME = "evaluation_events.jsonl"
STATE_LOCK_FILENAME = ".evaluation-state.lock"
ACTIVE_STATUSES = {
    "launching",
    "running",
    "stopping_timeout",
    "stopping_forced",
}
_ACTIVE_CHILDREN: dict[int, subprocess.Popen[bytes]] = {}
PersistCallback = Callable[[dict[str, Any], str], None]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha256(value: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"{label} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state_dir(spec: dict[str, Any]) -> Path:
    execution = spec["evaluation"].get("execution")
    if not isinstance(execution, dict):
        raise SpecError(
            "automated evaluation requires evaluation.execution authorization"
        )
    path = Path(execution["state_dir"])
    if path.exists() and (not path.is_dir() or path.is_symlink()):
        raise SpecError("evaluation.execution.state_dir must be a regular directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(spec: dict[str, Any]) -> Path:
    return _state_dir(spec) / STATE_FILENAME


def _journal_path(spec: dict[str, Any]) -> Path:
    return _state_dir(spec) / JOURNAL_FILENAME


@contextmanager
def evaluation_state_lock(spec: dict[str, Any]) -> Iterator[None]:
    stream = (_state_dir(spec) / STATE_LOCK_FILENAME).open(
        "a+",
        encoding="utf-8",
    )
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError(
                "another evaluation executor holds the state lock"
            ) from exc
        yield
    finally:
        stream.close()


def _read_journal(
    path: Path,
    allow_truncated_tail: bool = False,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if not path.is_file() or path.is_symlink():
        raise SpecError("evaluation journal must be a regular file")
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            if (
                allow_truncated_tail
                and line_number == len(lines)
                and not raw.endswith("\n")
            ):
                break
            raise SpecError(
                f"invalid evaluation journal JSON at line {line_number}: {exc.msg}"
            ) from exc
        if not isinstance(event, dict):
            raise SpecError(
                f"evaluation journal line {line_number} is not an object"
            )
        claimed_hash = event.get("event_sha256")
        unsigned = {
            key: value for key, value in event.items() if key != "event_sha256"
        }
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
                and not raw.endswith("\n")
            ):
                break
            raise SpecError(
                f"evaluation journal integrity failed at line {line_number}"
            )
        previous_hash = claimed_hash
        events.append(event)
    return events


def _rewrite_journal(path: Path, events: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(
            _canonical_bytes(event).decode("utf-8") + "\n"
            for event in events
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _persist_state(
    spec: dict[str, Any],
    state: dict[str, Any],
    action: str,
) -> None:
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
    _atomic_write(_state_path(spec), state)


def recover_state_from_journal(
    spec: dict[str, Any],
    session_path: Path,
    plan_path: Path,
) -> dict[str, Any]:
    journal_path = _journal_path(spec)
    events = _read_journal(journal_path, allow_truncated_tail=True)
    if not events:
        raise SpecError("evaluation journal is empty")
    canonical = "".join(
        _canonical_bytes(event).decode("utf-8") + "\n" for event in events
    )
    if journal_path.read_text(encoding="utf-8") != canonical:
        _rewrite_journal(journal_path, events)
    state = events[-1]["state"]
    if (
        state.get("session_sha256") != _sha256(session_path)
        or state.get("plan_sha256") != _sha256(plan_path)
    ):
        raise SpecError(
            "evaluation journal is bound to a different session or plan"
        )
    _atomic_write(_state_path(spec), state)
    return state


def _validate_plan_against_spec(
    spec: dict[str, Any],
    plan: dict[str, Any],
) -> None:
    evaluation = spec["evaluation"]
    if plan.get("algorithm") != spec["algorithm"]:
        raise SpecError("evaluation plan algorithm does not match the session")
    if plan.get("training_run_id") != spec["training"]["run_id"]:
        raise SpecError("evaluation plan training run does not match the session")
    if plan.get("gates") != evaluation["gates"]:
        raise SpecError("evaluation plan gates do not match the session")
    if plan.get("minimum_reviewed_videos") != evaluation["visual_review"][
        "minimum_reviewed_videos"
    ]:
        raise SpecError("evaluation plan visual-review contract changed")
    output_root = Path(evaluation["output_dir"])
    artifact_specs = {
        artifact["kind"]: artifact for artifact in evaluation["artifacts"]
    }
    scenario_specs = {
        scenario["id"]: scenario for scenario in evaluation["scenarios"]
    }
    candidate_ids = plan.get("candidate_ids")
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise SpecError("evaluation plan candidate IDs are invalid")
    run_ids: set[str] = set()
    matrix_cells: set[tuple[str, str, str, int]] = set()
    for index, run in enumerate(plan["runs"]):
        path = f"evaluation plan runs[{index}]"
        if not isinstance(run, dict):
            raise SpecError(f"{path} must be an object")
        run_id = run.get("run_id")
        command = run.get("command")
        if not isinstance(run_id, str) or not run_id:
            raise SpecError(f"{path}.run_id is invalid")
        if run_id in run_ids:
            raise SpecError(f"duplicate evaluation run ID: {run_id}")
        run_ids.add(run_id)
        candidate_id = run.get("candidate_id")
        artifact_kind = run.get("artifact")
        scenario_id = run.get("scenario_id")
        seed = run.get("seed")
        if (
            candidate_id not in candidate_ids
            or artifact_kind not in artifact_specs
            or scenario_id not in scenario_specs
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise SpecError(f"{path} matrix identity is invalid")
        scenario = scenario_specs[scenario_id]
        artifact = artifact_specs[artifact_kind]
        cell = (candidate_id, artifact_kind, scenario_id, seed)
        if cell in matrix_cells:
            raise SpecError(f"duplicate evaluation matrix cell: {cell}")
        matrix_cells.add(cell)
        expected_run_id = (
            f"{candidate_id}__{artifact_kind}__{scenario_id}__seed-{seed}"
        )
        if run_id != expected_run_id:
            raise SpecError(f"{path}.run_id is not deterministic")
        if (
            seed not in scenario["seeds"]
            or run.get("duration_steps") != scenario["duration_steps"]
            or run.get("overrides") != scenario["overrides"]
            or run.get("command_schedule") != scenario["command_schedule"]
            or run.get("scenario_category") != scenario["category"]
            or run.get("scenario_required") != scenario["required"]
            or run.get("artifact_required") != artifact["required"]
            or run.get("video_required")
            != bool(artifact["required"] and scenario["required"] and scenario["video"])
        ):
            raise SpecError(f"{path} differs from the approved scenario/artifact")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(token, str) and token for token in command)
        ):
            raise SpecError(f"{path}.command must be a non-empty argv array")
        if command.count(run_id) < 1:
            raise SpecError(
                f"{path}.command must contain the run ID as a standalone token"
            )
        for field in (
            "artifact_path",
            "checkpoint_path",
            "result_path",
            "video_path",
        ):
            value = run.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise SpecError(f"{path}.{field} must be absolute")
        for field in ("result_path", "video_path"):
            try:
                Path(run[field]).relative_to(output_root)
            except ValueError as exc:
                raise SpecError(
                    f"{path}.{field} must be inside evaluation.output_dir"
                ) from exc
        expected_run_dir = (
            output_root
            / candidate_id
            / artifact_kind
            / scenario_id
            / f"seed-{seed}"
        )
        if (
            Path(run["result_path"]) != expected_run_dir / "result.json"
            or Path(run["video_path"]) != expected_run_dir / "motion.mp4"
            or Path(run.get("run_dir", "")) != expected_run_dir
        ):
            raise SpecError(f"{path} output paths are not deterministic")
        if artifact_kind == "native" and (
            run["artifact_path"] != run["checkpoint_path"]
            or run["artifact_sha256"] != run["checkpoint_sha256"]
        ):
            raise SpecError(f"{path} native artifact must equal its checkpoint")
        values = {
            "artifact_kind": artifact_kind,
            "artifact_path": run["artifact_path"],
            "artifact_sha256": run["artifact_sha256"],
            "candidate_id": candidate_id,
            "checkpoint_path": run["checkpoint_path"],
            "checkpoint_sha256": run["checkpoint_sha256"],
            "command_schedule_json": json.dumps(
                scenario["command_schedule"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "duration_steps": scenario["duration_steps"],
            "executor_run_id": run_id,
            "gpu_index": evaluation["gpu_index"],
            "result_path": run["result_path"],
            "require_idle_gpu_flag": "--require_idle_gpu",
            "run_id": run_id,
            "scenario_id": scenario_id,
            "scenario_overrides_json": json.dumps(
                scenario["overrides"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "seed": seed,
            "video_path": run["video_path"],
        }
        try:
            approved_command = [
                token.format_map(values) for token in artifact["command"]
            ]
        except (KeyError, ValueError) as exc:
            raise SpecError(
                f"cannot re-render approved evaluation command: {exc}"
            ) from exc
        if command != approved_command:
            raise SpecError(f"{path}.command differs from the approved template")
    expected_cells = {
        (candidate_id, artifact_kind, scenario_id, seed)
        for candidate_id in candidate_ids
        for artifact_kind in artifact_specs
        for scenario_id, scenario in scenario_specs.items()
        for seed in scenario["seeds"]
    }
    if matrix_cells != expected_cells:
        raise SpecError("evaluation plan does not contain the complete matrix")


def _run_record(run: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(run.get("run_dir", Path(run["result_path"]).parent))
    return {
        "run_id": run["run_id"],
        "candidate_id": run["candidate_id"],
        "artifact": run["artifact"],
        "scenario_id": run["scenario_id"],
        "seed": run["seed"],
        "status": "pending",
        "attempts": 0,
        "run_dir": str(run_dir),
        "canonical_result_path": run["result_path"],
        "canonical_video_path": run["video_path"],
        "attempt_dir": None,
        "raw_result_path": None,
        "raw_video_path": None,
        "log_path": None,
        "launch_receipt_path": None,
        "launch_receipt_sha256": None,
        "result_sha256": None,
        "video_sha256": None,
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
    }


def initialize_state(
    spec: dict[str, Any],
    session_path: Path,
    plan: dict[str, Any],
    plan_path: Path,
) -> dict[str, Any]:
    _validate_plan_against_spec(spec, plan)
    state_path = _state_path(spec)
    session_hash = _sha256(session_path)
    plan_hash = _sha256(plan_path)
    if state_path.exists():
        state = _load_object(state_path, "evaluation state")
        if (
            state.get("session_sha256") != session_hash
            or state.get("plan_sha256") != plan_hash
        ):
            raise SpecError(
                "existing evaluation state is bound to another session or plan"
            )
        return state
    for run in plan["runs"]:
        for output in (run["result_path"], run["video_path"]):
            if Path(output).exists():
                raise SpecError(
                    "canonical evaluation output already exists; choose a new "
                    f"output directory: {output}"
                )
    now = time.time()
    state = {
        "version": 1,
        "session_path": str(session_path),
        "session_sha256": session_hash,
        "plan_path": str(plan_path),
        "plan_sha256": plan_hash,
        "algorithm": spec["algorithm"],
        "training_run_id": spec["training"]["run_id"],
        "created_at": now,
        "updated_at": now,
        "stage": "executing",
        "runs": {
            run["run_id"]: _run_record(run) for run in plan["runs"]
        },
    }
    _atomic_write(state_path, state)
    return state


def _resource_preflight(spec: dict[str, Any]) -> dict[str, Any]:
    execution = spec["evaluation"]["execution"]
    free_gb = shutil.disk_usage(_state_dir(spec)).free / 1_000_000_000
    if free_gb < float(execution["min_free_disk_gb"]):
        raise SpecError(
            f"free disk {free_gb:.2f} GB is below approved minimum "
            f"{execution['min_free_disk_gb']} GB"
        )
    gpu_index = spec["evaluation"]["gpu_index"]
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
        raise SpecError(f"cannot verify approved GPU health: {exc}") from exc
    if temperature > execution["max_gpu_temperature_c"]:
        raise SpecError(
            f"GPU temperature {temperature} C exceeds approved maximum "
            f"{execution['max_gpu_temperature_c']} C"
        )
    return {
        "checked_at": time.time(),
        "free_disk_gb": free_gb,
        "gpu_temperature_c": temperature,
    }


def _verify_artifacts(expected: dict[str, Any]) -> None:
    for label, path_value, hash_value in (
        (
            "checkpoint",
            expected["checkpoint_path"],
            expected["checkpoint_sha256"],
        ),
        (
            "artifact",
            expected["artifact_path"],
            expected["artifact_sha256"],
        ),
    ):
        path = Path(path_value)
        if not path.is_file() or path.is_symlink():
            raise SpecError(f"approved {label} is not a regular file: {path}")
        if _sha256(path) != hash_value:
            raise SpecError(f"approved {label} hash changed: {path}")


def _attempt_command(
    expected: dict[str, Any],
    raw_result_path: Path,
    raw_video_path: Path,
) -> list[str]:
    replacements = {
        expected["result_path"]: str(raw_result_path),
        expected["video_path"]: str(raw_video_path),
    }
    counts = Counter({key: 0 for key in replacements})
    command: list[str] = []
    for token in expected["command"]:
        rendered = token
        for source, target in replacements.items():
            occurrences = rendered.count(source)
            counts[source] += occurrences
            rendered = rendered.replace(source, target)
        command.append(rendered)
    if any(count != 1 for count in counts.values()):
        raise SpecError(
            "evaluation command must contain each canonical result/video path "
            "exactly once"
        )
    return command


def _load_launch_receipt(run: dict[str, Any]) -> dict[str, Any]:
    path = Path(run["launch_receipt_path"])
    if not path.is_file() or path.is_symlink():
        raise SpecError("evaluation launch receipt is missing")
    receipt = _load_object(path, "evaluation launch receipt")
    recorded_hash = run.get("launch_receipt_sha256")
    if isinstance(recorded_hash, str) and _sha256(path) != recorded_hash:
        raise SpecError("evaluation launch receipt hash changed")
    if (
        receipt.get("version") != 1
        or receipt.get("run_id") != run["run_id"]
        or receipt.get("attempt") != run["attempts"]
        or receipt.get("argv") != run["argv"]
        or receipt.get("process_group") != receipt.get("pid")
        or not isinstance(receipt.get("pid"), int)
        or not isinstance(receipt.get("process_start_ticks"), int)
        or not isinstance(receipt.get("launched_at"), (int, float))
    ):
        raise SpecError("evaluation launch receipt identity is invalid")
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


def _record_failure(
    spec: dict[str, Any],
    state: dict[str, Any],
    run: dict[str, Any],
    reason: str,
    persist: PersistCallback | None = None,
) -> dict[str, Any]:
    run["failure_reason"] = reason
    run["finished_at"] = time.time()
    if run["attempts"] <= spec["evaluation"]["execution"][
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
        persist(state, "evaluation-attempt-failed")
    return state


def _validate_finite_json(value: Any, path: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise SpecError(f"{path} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise SpecError(f"{path} contains a non-string key")
            _validate_finite_json(item, f"{path}.{key}")
        return
    raise SpecError(f"{path} contains an unsupported JSON value")


def _load_raw_result(
    expected: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    path = Path(run["raw_result_path"])
    if not path.is_file() or path.is_symlink():
        raise SpecError("evaluation child did not produce a regular result")
    result = _load_object(path, "raw evaluation result")
    for key in ("run_id", "candidate_id", "artifact", "scenario_id", "seed"):
        if result.get(key) != expected[key]:
            raise SpecError(f"raw evaluation result {key} is invalid")
    if result.get("version") != 1 or result.get("status") != "completed":
        raise SpecError("raw evaluation result is not completed version 1")
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise SpecError("raw evaluation result metrics must be non-empty")
    _validate_finite_json(metrics, "raw evaluation metrics")
    motion_evidence = result.get("motion_evidence")
    if motion_evidence is not None:
        if not isinstance(motion_evidence, dict):
            raise SpecError("motion_evidence must be an object")
        _validate_finite_json(motion_evidence, "motion_evidence")
    return result


def _selected(selectors: list[str], value: str) -> bool:
    return "*" in selectors or value in selectors


def _gate_observations(
    spec: dict[str, Any],
    expected: dict[str, Any],
    metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    operators = {
        "<=": lambda actual, limit: actual <= limit,
        ">=": lambda actual, limit: actual >= limit,
        "<": lambda actual, limit: actual < limit,
        ">": lambda actual, limit: actual > limit,
    }
    for gate in spec["evaluation"]["gates"]:
        if not (
            _selected(gate["artifacts"], expected["artifact"])
            and _selected(gate["scenarios"], expected["scenario_id"])
        ):
            continue
        value = metrics.get(gate["metric"])
        passed = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and operators[gate["op"]](float(value), float(gate["value"]))
        )
        observations.append(
            {
                "metric": gate["metric"],
                "actual": float(value) if isinstance(value, (int, float)) else None,
                "op": gate["op"],
                "limit": float(gate["value"]),
                "single_run_observation_passed": passed,
                "final_aggregation": gate["aggregation"],
            }
        )
    return observations


def _load_canonical_result(
    expected: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any] | None:
    path = Path(run["canonical_result_path"])
    if not path.exists():
        return None
    result = _load_object(path, "canonical evaluation result")
    evidence = result.get("execution_evidence")
    identity_matches = all(
        result.get(key) == expected[key]
        for key in ("run_id", "candidate_id", "artifact", "scenario_id", "seed")
    )
    if (
        not identity_matches
        or result.get("status") != "completed"
        or not isinstance(evidence, dict)
        or evidence.get("attempt") != run["attempts"]
    ):
        raise SpecError("canonical evaluation result identity is invalid")
    recorded_result_hash = run.get("result_sha256")
    if (
        isinstance(recorded_result_hash, str)
        and _sha256(path) != recorded_result_hash
    ):
        raise SpecError("canonical evaluation result hash changed")
    video_hash = evidence.get("video_sha256")
    video_path = result.get("video_path")
    if video_hash is not None:
        path_value = Path(video_path) if isinstance(video_path, str) else None
        if (
            path_value is None
            or not path_value.is_file()
            or _sha256(path_value) != video_hash
        ):
            raise SpecError("canonical evaluation video hash is invalid")
    if not isinstance(result.get("metrics"), dict) or not result["metrics"]:
        raise SpecError("canonical evaluation metrics must be non-empty")
    _validate_finite_json(result["metrics"], "canonical metrics")
    return result


def _promote_result(
    spec: dict[str, Any],
    expected: dict[str, Any],
    run: dict[str, Any],
) -> None:
    _verify_artifacts(expected)
    existing = _load_canonical_result(expected, run)
    if existing is not None:
        run["result_sha256"] = _sha256(Path(run["canonical_result_path"]))
        evidence = existing["execution_evidence"]
        run["video_sha256"] = evidence.get("video_sha256")
        return
    result = _load_raw_result(expected, run)
    raw_video = Path(run["raw_video_path"])
    canonical_video = Path(run["canonical_video_path"])
    video_hash: str | None = None
    minimum_bytes = spec["evaluation"]["execution"]["minimum_video_bytes"]
    if raw_video.exists():
        if (
            not raw_video.is_file()
            or raw_video.is_symlink()
            or raw_video.stat().st_size < minimum_bytes
        ):
            raise SpecError("evaluation video is incomplete or too small")
        if canonical_video.exists():
            raise SpecError("both raw and canonical evaluation videos exist")
        canonical_video.parent.mkdir(parents=True, exist_ok=True)
        os.replace(raw_video, canonical_video)
        video_hash = _sha256(canonical_video)
    elif canonical_video.exists():
        if (
            not canonical_video.is_file()
            or canonical_video.is_symlink()
            or canonical_video.stat().st_size < minimum_bytes
        ):
            raise SpecError("canonical evaluation video is incomplete")
        video_hash = _sha256(canonical_video)
    elif expected["video_required"]:
        raise SpecError("required evaluation video is missing")

    normalized = dict(result)
    normalized["video_path"] = str(canonical_video) if video_hash else ""
    normalized["execution_evidence"] = {
        "attempt": run["attempts"],
        "raw_result_path": run["raw_result_path"],
        "raw_result_sha256": _sha256(Path(run["raw_result_path"])),
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "artifact_sha256": expected["artifact_sha256"],
        "video_sha256": video_hash,
        "approved_gate_observations": _gate_observations(
            spec,
            expected,
            result["metrics"],
        ),
    }
    canonical_result = Path(run["canonical_result_path"])
    if canonical_result.exists():
        raise SpecError("canonical evaluation result unexpectedly exists")
    _atomic_write(canonical_result, normalized)
    run["result_sha256"] = _sha256(canonical_result)
    run["video_sha256"] = video_hash


def _update_stage(state: dict[str, Any]) -> None:
    statuses = {run["status"] for run in state["runs"].values()}
    if "failed" in statuses:
        state["stage"] = "blocked"
    elif statuses == {"completed"}:
        state["stage"] = "awaiting_visual_review"
    else:
        state["stage"] = "executing"


def reconcile(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    expected_runs = {run["run_id"]: run for run in plan["runs"]}
    now = time.time()
    for run in state["runs"].values():
        if run["status"] not in ACTIVE_STATUSES:
            continue
        previous_status = run["status"]
        if previous_status == "launching":
            try:
                receipt = _load_launch_receipt(run)
            except SpecError as exc:
                orphan_pid = _find_process_with_exact_token(run["run_id"])
                if orphan_pid is not None:
                    run["status"] = "failed"
                    run["finished_at"] = now
                    run["failure_reason"] = (
                        "orphaned evaluation launch has no valid receipt; "
                        f"refusing process control: {exc}"
                    )
                    state["stage"] = "blocked"
                else:
                    _record_failure(
                        spec,
                        state,
                        run,
                        f"reserved evaluation launch has no valid receipt: {exc}",
                    )
                continue
            _restore_launch_receipt(run, receipt)
            previous_status = "running"
        if _process_is_exact(run):
            elapsed = now - float(run["launched_at"])
            timeout = spec["evaluation"]["run_timeout_minutes"] * 60
            if run["status"] == "running" and elapsed >= timeout:
                _stop_exact_process(run)
                run["status"] = "stopping_timeout"
                run["stop_reason"] = "approved_evaluation_timeout"
                run["stop_requested_at"] = now
                run["failure_reason"] = run["stop_reason"]
            elif run["status"] in {"stopping_timeout", "stopping_forced"}:
                grace = spec["evaluation"]["execution"]["stop_grace_seconds"]
                requested = run.get("stop_requested_at")
                if (
                    run["status"] != "stopping_forced"
                    and isinstance(requested, (int, float))
                    and now - float(requested) >= grace
                ):
                    _stop_exact_process(run, signal.SIGKILL)
                    run["status"] = "stopping_forced"
            continue

        pid = run.get("pid")
        if isinstance(pid, int):
            _ACTIVE_CHILDREN.pop(pid, None)
        expected = expected_runs[run["run_id"]]
        if previous_status != "running":
            _record_failure(
                spec,
                state,
                run,
                run.get("failure_reason") or "evaluation process stopped",
            )
            continue
        try:
            _promote_result(spec, expected, run)
        except (OSError, SpecError, ValueError) as exc:
            _record_failure(spec, state, run, str(exc))
            continue
        run["status"] = "completed"
        run["finished_at"] = now
        run["failure_reason"] = None
    state["updated_at"] = time.time()
    _update_stage(state)
    return state


def launch_next(
    spec: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    persist: PersistCallback | None = None,
) -> dict[str, Any]:
    if any(run["status"] in ACTIVE_STATUSES for run in state["runs"].values()):
        raise SpecError("an evaluation run is already active")
    pending = next(
        (run for run in state["runs"].values() if run["status"] == "pending"),
        None,
    )
    if pending is None:
        raise SpecError("no pending evaluation run is available")
    expected = next(
        run for run in plan["runs"] if run["run_id"] == pending["run_id"]
    )
    if _find_process_with_exact_token(pending["run_id"]) is not None:
        raise SpecError(f"evaluation run ID is already active: {pending['run_id']}")
    _verify_artifacts(expected)
    pending["resource_report"] = _resource_preflight(spec)
    gpu_index = spec["evaluation"]["gpu_index"]
    if spec["evaluation"]["require_idle_gpu"] and not _gpu_idle(gpu_index):
        raise SpecError("approved evaluation GPU is not idle")

    lock_stream = gpu_lock_path(gpu_index).open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.close()
        raise SpecError("another skill executor holds the approved GPU") from exc
    if spec["evaluation"]["require_idle_gpu"] and not _gpu_idle(gpu_index):
        lock_stream.close()
        raise SpecError("approved evaluation GPU became busy before launch")
    os.set_inheritable(lock_stream.fileno(), True)

    attempt = pending["attempts"] + 1
    run_dir = Path(pending["run_dir"])
    attempt_dir = run_dir / f"attempt-{attempt}"
    if attempt_dir.exists():
        lock_stream.close()
        raise SpecError("evaluation attempt directory already exists")
    raw_result = attempt_dir / "result.json"
    raw_video = attempt_dir / "motion.mp4"
    command = _attempt_command(expected, raw_result, raw_video)
    pending.update(
        {
            "status": "launching",
            "attempts": attempt,
            "attempt_dir": str(attempt_dir),
            "raw_result_path": str(raw_result),
            "raw_video_path": str(raw_video),
            "log_path": str(attempt_dir / "evaluation.log"),
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
        try:
            persist(state, "reserve-evaluation-attempt")
        except BaseException:
            lock_stream.close()
            raise

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
    except (OSError, SpecError, ValueError) as exc:
        if log_stream is not None:
            log_stream.close()
        lock_stream.close()
        return _record_failure(
            spec,
            state,
            pending,
            f"failed to launch approved evaluation argv: {exc}",
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
        return _record_failure(
            spec,
            state,
            pending,
            "evaluation process identity could not be recorded",
            persist,
        )
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
    except (OSError, SpecError, ValueError) as exc:
        try:
            _stop_exact_process(pending)
        except SpecError:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        return _record_failure(
            spec,
            state,
            pending,
            f"evaluation launch receipt could not be recorded: {exc}",
            persist,
        )
    pending["status"] = "running"
    _ACTIVE_CHILDREN[process.pid] = process
    state["updated_at"] = time.time()
    if persist is not None:
        persist(state, "evaluation-launch-started")
    return state


def state_summary(state: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(run["status"] for run in state["runs"].values())
    return {
        "version": state["version"],
        "stage": state["stage"],
        "counts": dict(sorted(counts.items())),
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
        "completed_result_paths": [
            run["canonical_result_path"]
            for run in state["runs"].values()
            if run["status"] == "completed"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Approved session JSON")
    parser.add_argument("plan", help="Version-1 evaluation plan JSON")
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
        evaluation = spec.get("evaluation")
        if (
            not isinstance(evaluation, dict)
            or not evaluation.get("enabled")
            or not isinstance(evaluation.get("execution"), dict)
        ):
            raise SpecError(
                "session must authorize automated evaluation execution"
            )
        plan = load_evaluation_plan(plan_path)
        with evaluation_state_lock(spec):
            if args.action == "recover-state":
                _validate_plan_against_spec(spec, plan)
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
                    _persist_state(spec, state, "initialize-evaluation")
                elif args.action == "reconcile":
                    state = reconcile(spec, plan, state)
                    _persist_state(spec, state, "reconcile-evaluation")
                elif args.action == "launch-next":
                    state = reconcile(spec, plan, state)
                    _persist_state(
                        spec,
                        state,
                        "reconcile-before-evaluation-launch",
                    )

                    def persist_transition(
                        transition_state: dict[str, Any],
                        action: str,
                    ) -> None:
                        _persist_state(spec, transition_state, action)

                    state = launch_next(
                        spec,
                        plan,
                        state,
                        persist_transition,
                    )
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

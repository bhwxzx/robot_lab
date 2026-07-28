#!/usr/bin/env python3
"""Advance an approved training-to-policy-evaluation handoff by one transition."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from build_evaluation_plan import _load_candidates, build_plan
from execute_evaluation_plan import (
    ACTIVE_STATUSES,
    _load_object as load_evaluation_object,
    _persist_state as persist_evaluation_state,
    _state_path as evaluation_state_path,
    evaluation_state_lock,
    initialize_state as initialize_evaluation_state,
    launch_next as launch_evaluation,
    reconcile as reconcile_evaluation,
    state_summary as evaluation_summary,
)
from validate_session_spec import SpecError, load_and_validate


STATE_NAME = "evaluation_handoff_state.json"
JOURNAL_NAME = "evaluation_handoff_events.jsonl"
LOCK_NAME = ".evaluation-handoff.lock"


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
            raise SpecError(f"immutable handoff artifact changed: {path}")
        return
    _atomic_write(path, value)


def _root(spec: dict[str, Any]) -> Path:
    return Path(spec["evaluation"]["output_dir"]) / ".handoff"


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    stream = (root / LOCK_NAME).open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpecError("evaluation handoff controller is already active") from exc
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"invalid handoff journal JSON at line {line_number}"
            ) from exc
        if (
            not isinstance(event, dict)
            or event.get("sequence") != line_number
            or event.get("previous_event_sha256") != previous
            or event.get("event_sha256")
            != _object_sha256(
                {
                    key: value
                    for key, value in event.items()
                    if key != "event_sha256"
                }
            )
        ):
            raise SpecError("evaluation handoff journal chain is invalid")
        previous = event["event_sha256"]
        events.append(event)
    return events


def _persist(root: Path, state: dict[str, Any], action: str) -> None:
    journal_path = root / JOURNAL_NAME
    events = _read_events(journal_path)
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
    event["event_sha256"] = _object_sha256(
        {
            key: value
            for key, value in event.items()
            if key != "event_sha256"
        }
    )
    with journal_path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_bytes(event).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _atomic_write(root / STATE_NAME, state)


def _new_state(
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
    worker_id: str | None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "session_sha256": _file_sha256(session_path),
        "training_ranking_sha256": _file_sha256(ranking_path),
        "checkpoint_inventory_sha256": _file_sha256(inventory_path),
        "worker_id": worker_id,
        "candidate_manifest_path": None,
        "candidate_manifest_sha256": None,
        "evaluation_plan_path": None,
        "evaluation_plan_sha256": None,
        "last_action": "initialized",
        "updated_at": time.time(),
    }


def _load_state(
    root: Path,
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
    worker_id: str | None,
) -> dict[str, Any] | None:
    path = root / STATE_NAME
    if not path.exists():
        return None
    state = _load_json(path, "evaluation handoff state")
    events = _read_events(root / JOURNAL_NAME)
    if (
        not events
        or events[-1].get("state") != state
        or events[-1].get("state_sha256") != _object_sha256(state)
        or state.get("version") != 1
        or state.get("session_sha256") != _file_sha256(session_path)
        or state.get("training_ranking_sha256") != _file_sha256(ranking_path)
        or state.get("checkpoint_inventory_sha256")
        != _file_sha256(inventory_path)
        or state.get("worker_id") != worker_id
    ):
        raise SpecError("evaluation handoff state binding is invalid")
    for path_key, hash_key in (
        ("candidate_manifest_path", "candidate_manifest_sha256"),
        ("evaluation_plan_path", "evaluation_plan_sha256"),
    ):
        bound_path = state.get(path_key)
        bound_hash = state.get(hash_key)
        if bound_path is not None and (
            not isinstance(bound_hash, str)
            or _file_sha256(Path(bound_path)) != bound_hash
        ):
            raise SpecError("evaluation handoff artifact binding is invalid")
    return state


def _validate_sources(
    spec: dict[str, Any],
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ranking = _load_json(ranking_path, "training ranking")
    inventory = _load_json(inventory_path, "checkpoint inventory")
    if (
        ranking.get("algorithm") != spec["algorithm"]
        or ranking.get("final_selection") is not None
        or ranking.get("selection_status") != "awaiting_policy_evaluation"
    ):
        raise SpecError(
            "training ranking is not awaiting the approved policy evaluation"
        )
    if (
        inventory.get("version") != 1
        or inventory.get("session_sha256") != _file_sha256(session_path)
        or inventory.get("training_ranking_sha256") != _file_sha256(ranking_path)
        or inventory.get("training_run_id") != spec["training"]["run_id"]
        or inventory.get("algorithm") != spec["algorithm"]
        or not isinstance(inventory.get("entries"), list)
    ):
        raise SpecError("checkpoint inventory binding is invalid")
    return ranking, inventory


def _selected_candidates(
    spec: dict[str, Any],
    ranking: dict[str, Any],
    inventory: dict[str, Any],
    worker_id: str | None,
) -> list[dict[str, Any]]:
    contract = spec["evaluation_handoff"]
    ranked = ranking.get("ranking")
    if not isinstance(ranked, list):
        raise SpecError("training ranking entries are invalid")
    selected = [
        item
        for item in ranked
        if isinstance(item, dict) and item.get("pareto_optimal") is True
    ][: contract["top_k"]]
    if len(selected) != contract["top_k"]:
        raise SpecError("training ranking has too few Pareto candidates")
    expected_worker = contract["evaluation_worker_id"]
    if worker_id != expected_worker:
        raise SpecError("this is not the approved evaluation worker")
    checkpoint_seed = contract["checkpoint_seed"]
    selected_candidates: list[dict[str, Any]] = []
    for ranked_item in selected:
        trial_id = ranked_item.get("trial_id")
        matches = [
            entry
            for entry in inventory["entries"]
            if isinstance(entry, dict)
            and entry.get("trial_id") == trial_id
            and entry.get("seed") == checkpoint_seed
            and entry.get("worker_id") == expected_worker
        ]
        if not matches:
            raise SpecError(
                f"candidate {trial_id} has no checkpoint on the approved "
                "evaluation worker"
            )
        matches.sort(
            key=lambda item: (
                item.get("checkpoint_step", -1),
                item.get("run_id", ""),
            ),
            reverse=True,
        )
        checkpoint = matches[0]
        if (
            len(matches) > 1
            and matches[1].get("checkpoint_step")
            == checkpoint.get("checkpoint_step")
            and (
                matches[1].get("checkpoint_path")
                != checkpoint.get("checkpoint_path")
                or matches[1].get("checkpoint_sha256")
                != checkpoint.get("checkpoint_sha256")
            )
        ):
            raise SpecError(f"candidate {trial_id} has ambiguous newest checkpoints")
        checkpoint_path = Path(checkpoint.get("checkpoint_path", ""))
        checkpoint_hash = checkpoint.get("checkpoint_sha256")
        if (
            not checkpoint_path.is_absolute()
            or not checkpoint_path.is_file()
            or checkpoint_path.is_symlink()
            or not isinstance(checkpoint_hash, str)
            or _file_sha256(checkpoint_path) != checkpoint_hash
        ):
            raise SpecError(f"candidate {trial_id} checkpoint hash changed")
        values = {
            "candidate_id": trial_id,
            "checkpoint_dir": str(checkpoint_path.parent),
            "checkpoint_path": str(checkpoint_path),
            "rsl_rl_run_dir": checkpoint.get("rsl_rl_run_dir"),
            "seed": checkpoint_seed,
            "trial_id": trial_id,
        }
        artifacts: dict[str, str] = {}
        artifact_hashes: dict[str, str] = {}
        for kind, template in contract["artifact_path_templates"].items():
            try:
                artifact_path = Path(template.format_map(values))
            except (KeyError, ValueError, TypeError) as exc:
                raise SpecError(
                    f"cannot render {kind} artifact path for {trial_id}"
                ) from exc
            if (
                not artifact_path.is_absolute()
                or not artifact_path.is_file()
                or artifact_path.is_symlink()
            ):
                raise SpecError(
                    f"candidate {trial_id} {kind} artifact is unavailable"
                )
            artifacts[kind] = str(artifact_path)
            artifact_hashes[kind] = _file_sha256(artifact_path)
        selected_candidates.append(
            {
                "candidate_id": trial_id,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "artifacts": artifacts,
                "artifact_sha256": artifact_hashes,
            }
        )
    return selected_candidates


def _decision(
    spec: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    plan_path_value = state.get("evaluation_plan_path")
    if plan_path_value is None:
        return {
            "next_action": "prepare_evaluation",
            "reason": "candidate_manifest_and_plan_absent",
        }
    plan_path = Path(plan_path_value)
    state_path = evaluation_state_path(spec)
    if not state_path.exists():
        return {
            "next_action": "initialize_evaluation",
            "reason": "evaluation_state_absent",
            "evaluation_plan_path": str(plan_path),
        }
    evaluation_state = load_evaluation_object(
        state_path,
        "evaluation state",
    )
    if (
        evaluation_state.get("session_sha256") != state["session_sha256"]
        or evaluation_state.get("plan_sha256")
        != state["evaluation_plan_sha256"]
    ):
        raise SpecError("evaluation executor state differs from handoff bindings")
    summary = evaluation_summary(evaluation_state)
    runs = list(evaluation_state["runs"].values())
    if any(run["status"] in ACTIVE_STATUSES for run in runs):
        action, reason = "reconcile", "evaluation_child_active"
    elif any(run["status"] == "failed" for run in runs):
        action, reason = "blocked", "evaluation_contains_failed_run"
    elif any(run["status"] == "pending" for run in runs):
        action, reason = "launch_next", "pending_evaluation_cell_available"
    elif evaluation_state.get("stage") == "awaiting_visual_review":
        action, reason = (
            "awaiting_visual_review",
            "automatic_evaluation_complete",
        )
    else:
        action, reason = "reconcile", "evaluation_stage_requires_reconcile"
    return {
        "next_action": action,
        "reason": reason,
        "evaluation": summary,
        "evaluation_plan_path": str(plan_path),
    }


def _advance(
    spec: dict[str, Any],
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
    root: Path,
    state: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    action = decision["next_action"]
    if action in {"blocked", "awaiting_visual_review"}:
        return {"action_taken": "none", **decision}
    if action == "prepare_evaluation":
        ranking, inventory = _validate_sources(
            spec,
            session_path,
            ranking_path,
            inventory_path,
        )
        candidates = _selected_candidates(
            spec,
            ranking,
            inventory,
            state["worker_id"],
        )
        manifest = {"candidates": candidates}
        manifest_path = root / "candidate_manifest.json"
        _write_immutable(manifest_path, manifest)
        selected_artifacts = {
            artifact["kind"] for artifact in spec["evaluation"]["artifacts"]
        }
        normalized = _load_candidates(manifest_path, selected_artifacts)
        plan = build_plan(spec, normalized)
        plan_path = root / "evaluation_plan.json"
        _write_immutable(plan_path, plan)
        state["candidate_manifest_path"] = str(manifest_path)
        state["candidate_manifest_sha256"] = _file_sha256(manifest_path)
        state["evaluation_plan_path"] = str(plan_path)
        state["evaluation_plan_sha256"] = _file_sha256(plan_path)
    else:
        plan_path = Path(state["evaluation_plan_path"])
        plan = _load_json(plan_path, "evaluation plan")
        with evaluation_state_lock(spec):
            evaluation_state = initialize_evaluation_state(
                spec,
                session_path,
                plan,
                plan_path,
            )
            if action == "initialize_evaluation":
                persist_evaluation_state(
                    spec,
                    evaluation_state,
                    "handoff-initialize-evaluation",
                )
            elif action == "reconcile":
                evaluation_state = reconcile_evaluation(
                    spec,
                    plan,
                    evaluation_state,
                )
                persist_evaluation_state(
                    spec,
                    evaluation_state,
                    "handoff-reconcile-evaluation",
                )
            elif action == "launch_next":
                evaluation_state = reconcile_evaluation(
                    spec,
                    plan,
                    evaluation_state,
                )
                persist_evaluation_state(
                    spec,
                    evaluation_state,
                    "handoff-reconcile-before-launch",
                )

                def persist_transition(
                    value: dict[str, Any],
                    transition: str,
                ) -> None:
                    persist_evaluation_state(
                        spec,
                        value,
                        f"handoff-{transition}",
                    )

                launch_evaluation(
                    spec,
                    plan,
                    evaluation_state,
                    persist_transition,
                )
            else:
                raise SpecError(f"unsupported evaluation handoff action: {action}")
    state["last_action"] = action
    state["updated_at"] = time.time()
    _persist(root, state, action)
    return {"action_taken": action, **_decision(spec, state)}


def inspect_or_advance(
    session_path: Path,
    ranking_path: Path,
    inventory_path: Path,
    *,
    execute: bool,
    worker_id: str | None = None,
) -> dict[str, Any]:
    spec = load_and_validate(session_path)
    contract = spec.get("evaluation_handoff")
    if not isinstance(contract, dict) or not contract.get("enabled"):
        raise SpecError("session does not enable evaluation_handoff")
    if execute and contract["mode"] != "execute":
        raise SpecError("evaluation handoff advance requires mode=execute")
    if worker_id != contract["evaluation_worker_id"]:
        raise SpecError("this is not the approved evaluation worker")
    _validate_sources(spec, session_path, ranking_path, inventory_path)
    root = _root(spec)
    state = (
        _load_state(
            root,
            session_path,
            ranking_path,
            inventory_path,
            worker_id,
        )
        if root.exists()
        else None
    )
    if state is None:
        state = _new_state(
            session_path,
            ranking_path,
            inventory_path,
            worker_id,
        )
        if not execute:
            return {
                "mode": "shadow",
                "next_action": "initialize_handoff",
                "reason": "handoff_state_absent",
                "would_write": str(root / STATE_NAME),
            }
        root.mkdir(parents=True, exist_ok=True)
        with _lock(root):
            _persist(root, state, "initialize-handoff")
        return {
            "mode": "execute",
            "action_taken": "initialize_handoff",
            "state_path": str(root / STATE_NAME),
        }
    with _lock(root):
        decision = _decision(spec, state)
        result = (
            _advance(
                spec,
                session_path,
                ranking_path,
                inventory_path,
                root,
                state,
                decision,
            )
            if execute
            else {"action_taken": "none", **decision}
        )
    return {
        "mode": "execute" if execute else "shadow",
        "state_path": str(root / STATE_NAME),
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("training_ranking")
    parser.add_argument("checkpoint_inventory")
    parser.add_argument(
        "--worker-id",
        help="Required local worker identity for version-7 evaluation",
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
            Path(args.training_ranking).resolve(),
            Path(args.checkpoint_inventory).resolve(),
            execute=args.action == "advance",
            worker_id=args.worker_id,
        )
    except (OSError, SpecError) as exc:
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

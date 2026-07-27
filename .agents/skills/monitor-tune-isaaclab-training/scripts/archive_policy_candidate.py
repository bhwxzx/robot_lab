#!/usr/bin/env python3
"""Archive one simulation-qualified JIT/ONNX policy pair with evidence."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo

from algorithm_profiles import load_registry, resolve_profile
from build_trial_plan import build_plan as build_trial_plan
from inspect_policy_storage import StorageError, inspect_storage
from rank_trials import rank
from validate_policy_evaluation import (
    evaluate_results,
    load_evaluation_plan,
    load_evaluation_results,
)
from validate_session_spec import SpecError, load_and_validate


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SpecError("value is not finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


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
    _canonical_sha256(value)
    return value


def _load_training_runs(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"training results do not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid training results JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
        raise SpecError("training results must contain a runs array")
    return value["runs"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SpecError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _distributed_lease(spec: dict[str, Any]) -> dict[str, Any] | None:
    archive = spec.get("archive")
    if not isinstance(archive, dict):
        return None
    lease = archive.get("distributed_lease")
    return lease if isinstance(lease, dict) and lease.get("enabled") else None


def _policy_storage_preflight(
    spec: dict[str, Any],
    worker_id: str,
    *,
    expected_base_commit: str | None = None,
) -> dict[str, Any]:
    lease = _distributed_lease(spec)
    if lease is None:
        raise SpecError("distributed policy archive lease is not enabled")
    if worker_id not in lease["authorized_worker_ids"]:
        raise SpecError("worker is not authorized for policy archive leases")
    storage_root = Path(lease["worker_storage_roots"][worker_id]).resolve()
    inventory = inspect_storage(storage_root, hash_artifacts=True)
    if not inventory["git_clean"]:
        raise SpecError("policy storage Git worktree is not clean")
    actual_origin = _git(storage_root, "remote", "get-url", "origin")
    if (
        actual_origin != lease["storage_remote_url"]
        and os.environ.get("POLICY_ARCHIVE_ALLOW_TEST_REMOTE") != "1"
    ):
        raise SpecError(
            "policy storage origin does not match the approved lease remote"
        )
    current_branch = _git(storage_root, "branch", "--show-current")
    if current_branch != lease["storage_branch"]:
        raise SpecError(
            "policy storage branch does not match the approved lease branch"
        )
    head = _git(storage_root, "rev-parse", "HEAD")
    remote_lines = _git(
        storage_root,
        "ls-remote",
        "--heads",
        "origin",
        f"refs/heads/{lease['storage_branch']}",
    ).splitlines()
    remote_commits = {
        line.split()[0]
        for line in remote_lines
        if len(line.split()) == 2
        and line.split()[1] == f"refs/heads/{lease['storage_branch']}"
    }
    if len(remote_commits) != 1:
        raise SpecError(
            "approved policy storage remote branch does not resolve exactly once"
        )
    remote_commit = next(iter(remote_commits))
    if head != remote_commit:
        raise SpecError(
            "policy storage local HEAD must equal the approved remote branch "
            "before requesting or using a lease"
        )
    if expected_base_commit is not None and head != expected_base_commit:
        raise SpecError(
            "policy storage base commit changed after the archive lease request"
        )
    return {
        "root": str(storage_root),
        "origin": lease["storage_remote_url"],
        "branch": lease["storage_branch"],
        "base_commit": head,
        "inventory": inventory,
    }


def _source_git_state(cwd: Path) -> dict[str, Any]:
    if not cwd.is_dir():
        return {"available": False}
    result = subprocess.run(
        ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"available": False}
    root = Path(result.stdout.strip()).resolve()
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "available": True,
        "root": str(root),
        "commit": _git(root, "rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def _task_from_command(command: list[str]) -> str:
    for index, token in enumerate(command):
        if token.startswith("--task="):
            return token.split("=", 1)[1]
        if token == "--task" and index + 1 < len(command):
            return command[index + 1]
    return "unknown"


def _artifact_pair(
    plan: dict[str, Any],
    candidate_id: str,
) -> dict[str, dict[str, str]]:
    records: dict[str, set[tuple[str, str]]] = {
        "jit": set(),
        "onnx": set(),
    }
    required_seen = {"jit": False, "onnx": False}
    for run in plan["runs"]:
        kind = run.get("artifact")
        if run.get("candidate_id") != candidate_id or kind not in records:
            continue
        path = run.get("artifact_path")
        digest = run.get("artifact_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise SpecError("evaluation plan artifact record is incomplete")
        records[kind].add((path, digest))
        required_seen[kind] = required_seen[kind] or bool(
            run.get("artifact_required")
        )

    pair: dict[str, dict[str, str]] = {}
    expected_suffix = {"jit": ".pt", "onnx": ".onnx"}
    for kind in ("jit", "onnx"):
        if len(records[kind]) != 1 or not required_seen[kind]:
            raise SpecError(
                f"candidate {candidate_id} must have one required {kind} artifact"
            )
        source_text, expected_hash = next(iter(records[kind]))
        source = Path(source_text)
        if (
            not source.is_absolute()
            or not source.is_file()
            or source.is_symlink()
        ):
            raise SpecError(
                f"{kind} artifact must be an existing absolute regular file"
            )
        if source.suffix.lower() != expected_suffix[kind]:
            raise SpecError(
                f"{kind} artifact must use {expected_suffix[kind]} format"
            )
        if _sha256(source) != expected_hash:
            raise SpecError(f"{kind} artifact SHA-256 changed after evaluation")
        pair[kind] = {
            "source_path": str(source),
            "sha256": expected_hash,
        }
    return pair


def _validate_selected_plan_coverage(
    spec: dict[str, Any],
    plan: dict[str, Any],
    candidate_id: str,
) -> None:
    artifact_specs = {
        artifact["kind"]: artifact
        for artifact in spec["evaluation"]["artifacts"]
    }
    scenario_specs = {
        scenario["id"]: scenario
        for scenario in spec["evaluation"]["scenarios"]
    }
    expected = {
        (artifact, scenario["id"], seed)
        for artifact in artifact_specs
        for scenario in scenario_specs.values()
        for seed in scenario["seeds"]
    }
    actual: set[tuple[str, str, int]] = set()
    for run in plan["runs"]:
        if run.get("candidate_id") != candidate_id:
            continue
        key = (
            run.get("artifact"),
            run.get("scenario_id"),
            run.get("seed"),
        )
        if key in actual:
            raise SpecError("evaluation plan contains a duplicate selected run")
        if key not in expected:
            raise SpecError("evaluation plan contains an unexpected selected run")
        actual.add(key)
        artifact = artifact_specs[key[0]]
        scenario = scenario_specs[key[1]]
        expected_video = bool(
            artifact["required"]
            and scenario["required"]
            and scenario["video"]
        )
        comparisons = {
            "artifact_required": artifact["required"],
            "scenario_required": scenario["required"],
            "duration_steps": scenario["duration_steps"],
            "overrides": scenario["overrides"],
            "command_schedule": scenario["command_schedule"],
            "video_required": expected_video,
        }
        for field, expected_value in comparisons.items():
            if run.get(field) != expected_value:
                raise SpecError(
                    f"evaluation plan {field} does not match the session"
                )
    if actual != expected:
        missing = sorted(expected - actual, key=repr)
        raise SpecError(
            f"evaluation plan is missing selected run cells: {missing}"
        )


def _duplicate_archive(
    inventory: dict[str, Any],
    pair: dict[str, dict[str, str]],
) -> str | None:
    expected = {
        "policy.pt": pair["jit"]["sha256"],
        "policy.onnx": pair["onnx"]["sha256"],
    }
    for entry in inventory["entries"]:
        if entry.get("sha256") == expected:
            return str(entry["relative_dir"])
    return None


def _description(manifest: dict[str, Any]) -> str:
    final = manifest["final_selection"]
    scenarios = ", ".join(manifest["evaluation"]["scenarios"])
    visual_notes = " | ".join(
        review["notes"] for review in manifest["evaluation"]["visual_reviews"]
    )
    metrics = json.dumps(
        final.get("mean_metrics", {}),
        ensure_ascii=False,
        sort_keys=True,
    )
    parameter_values = json.dumps(
        manifest["tuning_overrides"],
        ensure_ascii=False,
        sort_keys=True,
    )
    source_note = (
        "训练时源码含未提交修改，commit 单独不足以完全复现"
        if manifest["training_source_git"]["dirty"]
        else "训练时源码工作区已记录为干净"
    )
    notes = manifest["description_notes"].strip() or "无"
    return (
        f"策略类型: {manifest['task']}\n"
        f"算法: {manifest['algorithm']['name']}\n"
        f"Runner: {manifest['algorithm']['runner_class']}\n"
        f"算法画像: {manifest['algorithm']['profile_id']} "
        f"v{manifest['algorithm']['profile_version']}\n"
        f"训练运行: {manifest['training_run_id']}\n"
        f"训练源码提交: {manifest['training_source_git']['commit']}\n"
        f"训练源码含未提交修改: "
        f"{manifest['training_source_git']['dirty']}\n"
        f"训练源码复现提示: {source_note}\n"
        f"候选策略: {manifest['candidate_id']}\n"
        f"历史输入契约: {manifest['history_contract']}\n"
        f"调优参数: {parameter_values}\n"
        f"最终均值指标: {metrics}\n"
        f"闭环评估场景: {scenarios}\n"
        f"视觉审查: {visual_notes or '已通过，未提供额外摘要'}\n"
        f"JIT SHA-256: {manifest['artifacts']['jit']['sha256']}\n"
        f"ONNX SHA-256: {manifest['artifacts']['onnx']['sha256']}\n"
        f"归档时间: {manifest['archived_at']}\n"
        f"附加说明: {notes}\n"
        "仿真结论: simulation_qualified_hardware_candidate\n"
        "硬件状态: 仅可进入受监督实物测试；未经实物验证，"
        "不代表 hardware-ready。\n"
    )


def _build_manifest(
    spec: dict[str, Any],
    ranked: dict[str, Any],
    report: dict[str, Any],
    plan: dict[str, Any],
    results: dict[str, Any],
    pair: dict[str, dict[str, str]],
    archived_at: datetime,
    archive_id: str,
    storage_root: Path,
    storage_base_commit: str,
    lease_binding: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_id = ranked["final_selection"]["trial_id"]
    profile = resolve_profile(
        load_registry(),
        spec["algorithm"]["profile_id"],
    )
    candidate_reviews = [
        review
        for review in results["visual_reviews"]
        if review.get("candidate_id") == candidate_id
    ]
    scenario_ids = sorted(
        {
            run["scenario_id"]
            for run in plan["runs"]
            if run["candidate_id"] == candidate_id
            and run["scenario_required"]
        }
    )
    candidate_report = next(
        item
        for item in report["candidate_results"]
        if item["candidate_id"] == candidate_id
    )
    trial_plan = build_trial_plan(spec)
    planned_trial = next(
        (
            trial
            for trial in trial_plan["trials"]
            if trial["trial_id"] == candidate_id
        ),
        None,
    )
    if planned_trial is None:
        raise SpecError("final selection is not in the authorized trial plan")
    return {
        "version": 1,
        "archive_id": archive_id,
        "archived_at": archived_at.isoformat(),
        "candidate_id": candidate_id,
        "selection_status": ranked["selection_status"],
        "hardware_ready": False,
        "hardware_status": "supervised_real_robot_testing_required",
        "task": _task_from_command(spec["training"]["command"]),
        "training_run_id": spec["training"]["run_id"],
        "algorithm": spec["algorithm"],
        "history_contract": profile["evaluation_capabilities"][
            "history_contract"
        ],
        "tuning_overrides": planned_trial["overrides"],
        "final_selection": ranked["final_selection"],
        "evaluation": {
            "status": candidate_report["status"],
            "scenarios": scenario_ids,
            "gates": plan["gates"],
            "parity_expectations": [
                expectation
                for expectation in plan["parity_expectations"]
                if expectation["candidate_id"] == candidate_id
            ],
            "visual_reviews": candidate_reviews,
        },
        "artifacts": pair,
        "description_notes": spec["archive"]["description_notes"],
        "training_source_git": {
            "commit": spec["training"]["source_git_commit"],
            "dirty": spec["training"]["source_git_dirty"],
        },
        "source_repository_observed_at_archive": _source_git_state(
            Path(spec["training"]["cwd"])
        ),
        "storage_repository": {
            "root": str(storage_root),
            "base_commit": storage_base_commit,
            "git_action": "none",
        },
        "distributed_archive_lease": lease_binding,
    }


def _qualification_context(
    spec: dict[str, Any],
    training_runs: list[dict[str, Any]],
    plan: dict[str, Any],
    results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, dict[str, str]]]:
    for planned_candidate_id in plan["candidate_ids"]:
        _validate_selected_plan_coverage(
            spec,
            plan,
            planned_candidate_id,
        )
    report = evaluate_results(spec, plan, results)
    ranked = rank(spec, training_runs, evaluation_report=report)
    if (
        ranked.get("selection_status")
        != "simulation_qualified_hardware_candidate"
        or not isinstance(ranked.get("final_selection"), dict)
        or ranked.get("hardware_ready") is not False
    ):
        raise SpecError(
            "no simulation-qualified final selection is available to archive"
        )
    candidate_id = ranked["final_selection"].get("trial_id")
    if candidate_id not in report["simulation_qualified_candidates"]:
        raise SpecError("final selection is not simulation-qualified")
    pair = _artifact_pair(plan, candidate_id)
    return report, ranked, candidate_id, pair


def build_distributed_archive_request(
    spec: dict[str, Any],
    training_runs: list[dict[str, Any]],
    plan: dict[str, Any],
    results: dict[str, Any],
    worker_id: str,
) -> dict[str, Any]:
    """Build one hash-bound request without changing either Git repository."""
    if spec.get("version") != 7 or _distributed_lease(spec) is None:
        raise SpecError(
            "distributed archive requests require a version-7 lease-enabled session"
        )
    _, _, candidate_id, pair = _qualification_context(
        spec,
        training_runs,
        plan,
        results,
    )
    storage = _policy_storage_preflight(spec, worker_id)
    identity = {
        "schema_version": 1,
        "event": "policy_archive_requested",
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _canonical_sha256(spec),
        "worker_id": worker_id,
        "candidate_id": candidate_id,
        "artifacts": {
            "jit": {"sha256": pair["jit"]["sha256"]},
            "onnx": {"sha256": pair["onnx"]["sha256"]},
        },
        "storage_remote_url": storage["origin"],
        "storage_branch": storage["branch"],
        "storage_base_commit": storage["base_commit"],
    }
    return {
        **identity,
        "request_id": _canonical_sha256(identity),
    }


def _validate_lease_grant(
    spec: dict[str, Any],
    worker_id: str,
    grant: dict[str, Any],
    candidate_id: str,
    pair: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if grant.get("event") != "policy_archive_granted":
        raise SpecError("lease grant event is invalid")
    request = grant.get("request")
    if not isinstance(request, dict):
        raise SpecError("lease grant must contain the exact archive request")
    if grant.get("request_sha256") != _canonical_sha256(request):
        raise SpecError("lease grant request hash does not match")
    expected = {
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _canonical_sha256(spec),
        "worker_id": worker_id,
        "candidate_id": candidate_id,
        "storage_remote_url": spec["archive"]["distributed_lease"][
            "storage_remote_url"
        ],
        "storage_branch": spec["archive"]["distributed_lease"][
            "storage_branch"
        ],
    }
    for field, value in expected.items():
        if request.get(field) != value:
            raise SpecError(f"lease grant request {field} does not match")
    expected_hashes = {
        "jit": pair["jit"]["sha256"],
        "onnx": pair["onnx"]["sha256"],
    }
    for kind, digest in expected_hashes.items():
        record = request.get("artifacts", {}).get(kind)
        if not isinstance(record, dict) or record.get("sha256") != digest:
            raise SpecError(f"lease grant {kind} artifact hash does not match")
    lease_id = grant.get("lease_id")
    if (
        not isinstance(lease_id, str)
        or lease_id != request.get("request_id")
    ):
        raise SpecError("lease grant ID does not match the request")
    storage = _policy_storage_preflight(
        spec,
        worker_id,
        expected_base_commit=request.get("storage_base_commit"),
    )
    return request, storage


def archive_candidate(
    spec: dict[str, Any],
    training_runs: list[dict[str, Any]],
    plan: dict[str, Any],
    results: dict[str, Any],
    now: datetime | None = None,
    *,
    worker_id: str | None = None,
    lease_grant: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate promotion evidence and atomically archive the final pair."""
    archive = spec.get("archive")
    if spec.get("version") not in {4, 5, 6, 7}:
        raise SpecError("policy archive requires session version 4, 5, 6, or 7")
    if not isinstance(archive, dict) or not archive.get("enabled"):
        raise SpecError("session policy archive is not enabled")

    report, ranked, candidate_id, pair = _qualification_context(
        spec,
        training_runs,
        plan,
        results,
    )

    lease_binding: dict[str, Any] | None = None
    if _distributed_lease(spec) is not None:
        if worker_id is None or lease_grant is None:
            raise SpecError(
                "distributed policy archive requires worker ID and lease grant"
            )
        request, storage = _validate_lease_grant(
            spec,
            worker_id,
            lease_grant,
            candidate_id,
            pair,
        )
        storage_root = Path(storage["root"])
        lease_binding = {
            "lease_id": lease_grant["lease_id"],
            "worker_id": worker_id,
            "request_sha256": lease_grant["request_sha256"],
            "storage_base_commit": request["storage_base_commit"],
        }
    else:
        if worker_id is not None or lease_grant is not None:
            raise SpecError(
                "worker ID and lease grant are only valid for distributed archive"
            )
        storage_root = Path(archive["storage_root"])

    inventory = inspect_storage(storage_root, hash_artifacts=True)
    if archive["require_clean_git_worktree"] and not inventory["git_clean"]:
        raise SpecError("policy storage Git worktree is not clean")
    duplicate = _duplicate_archive(inventory, pair)
    if duplicate is not None:
        raise SpecError(
            f"the exact JIT/ONNX pair is already archived at {duplicate}"
        )

    resolved_root = storage_root.resolve()
    collection = resolved_root.joinpath(
        *PurePosixPath(archive["collection"]).parts
    )
    if (
        not collection.is_dir()
        or collection.is_symlink()
        or collection.resolve().parent != resolved_root.joinpath(
            *PurePosixPath(archive["collection"]).parts[:-1]
        ).resolve()
    ):
        raise SpecError(
            "authorized archive collection must be an existing regular directory"
        )
    if resolved_root not in collection.resolve().parents:
        raise SpecError("archive collection escapes storage_root")

    timezone = ZoneInfo(archive["timezone"])
    if now is None:
        archived_at = datetime.now(timezone)
    elif now.tzinfo is None:
        archived_at = now.replace(tzinfo=timezone)
    else:
        archived_at = now.astimezone(timezone)
    archive_id = archived_at.strftime("%Y-%m-%d-%H-%M-%S")
    destination = collection / archive_id
    temporary = collection / f".archive-tmp-{archive_id}-{os.getpid()}"
    if destination.exists() or temporary.exists():
        raise SpecError(f"archive destination already exists: {destination}")

    git_dir = resolved_root / ".git"
    if not git_dir.is_dir():
        raise SpecError("policy storage .git directory is missing")
    lock_fd = os.open(git_dir, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise SpecError("another policy archive operation is active") from exc

    try:
        locked_inventory = inspect_storage(
            resolved_root,
            hash_artifacts=True,
        )
        if not locked_inventory["git_clean"]:
            raise SpecError(
                "policy storage changed after preflight; archive aborted"
            )
        temporary.mkdir(mode=0o775)
        shutil.copy2(pair["jit"]["source_path"], temporary / "policy.pt")
        shutil.copy2(pair["onnx"]["source_path"], temporary / "policy.onnx")
        copied_hashes = {
            "jit": _sha256(temporary / "policy.pt"),
            "onnx": _sha256(temporary / "policy.onnx"),
        }
        for kind, digest in copied_hashes.items():
            if digest != pair[kind]["sha256"]:
                raise SpecError(f"copied {kind} artifact hash does not match")

        manifest = _build_manifest(
            spec,
            ranked,
            report,
            plan,
            results,
            pair,
            archived_at,
            archive_id,
            resolved_root,
            locked_inventory["git_commit"],
            lease_binding,
        )
        (temporary / "archive_manifest.json").write_text(
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "策略说明.txt").write_text(
            _description(manifest),
            encoding="utf-8",
        )
        temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    manifest_path = destination / "archive_manifest.json"
    receipt = {
        "version": 1,
        "status": "archived_simulation_qualified_hardware_candidate",
        "candidate_id": candidate_id,
        "archive_path": str(destination),
        "files": {
            "jit": str(destination / "policy.pt"),
            "onnx": str(destination / "policy.onnx"),
            "description": str(destination / "策略说明.txt"),
            "manifest": str(destination / "archive_manifest.json"),
        },
        "sha256": {
            "policy.pt": pair["jit"]["sha256"],
            "policy.onnx": pair["onnx"]["sha256"],
        },
        "hardware_ready": False,
        "hardware_status": "supervised_real_robot_testing_required",
        "git_action": "none",
    }
    if lease_binding is not None:
        receipt["distributed_archive_lease"] = lease_binding
        receipt["manifest_sha256"] = _sha256(manifest_path)
        receipt["storage_base_commit"] = inventory["git_commit"]
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "session",
        help="Validated version-4, version-5, version-6, or version-7 session JSON",
    )
    parser.add_argument("training_results", help="Per-trial training results")
    parser.add_argument("evaluation_plan", help="Evaluation plan JSON")
    parser.add_argument("evaluation_results", help="Evaluation results JSON")
    parser.add_argument(
        "--worker-id",
        help="Distributed worker requesting or using the archive lease",
    )
    parser.add_argument(
        "--lease-grant",
        help="Materialized active policy archive lease grant JSON",
    )
    parser.add_argument(
        "--prepare-lease-request",
        action="store_true",
        help="Build a request without copying policy artifacts",
    )
    parser.add_argument("--output", help="Optional absolute receipt JSON path")
    args = parser.parse_args()
    try:
        output_path = Path(args.output) if args.output else None
        if output_path is not None:
            if not output_path.is_absolute():
                raise SpecError("--output must be absolute")
            if output_path.exists():
                raise SpecError("--output already exists")
            if not output_path.parent.is_dir():
                raise SpecError("--output parent directory does not exist")
        spec = load_and_validate(args.session)
        training_runs = _load_training_runs(Path(args.training_results))
        evaluation_plan = load_evaluation_plan(Path(args.evaluation_plan))
        evaluation_results = load_evaluation_results(
            Path(args.evaluation_results)
        )
        if args.prepare_lease_request:
            if args.worker_id is None or args.lease_grant is not None:
                raise SpecError(
                    "--prepare-lease-request requires --worker-id and forbids "
                    "--lease-grant"
                )
            receipt = build_distributed_archive_request(
                spec,
                training_runs,
                evaluation_plan,
                evaluation_results,
                args.worker_id,
            )
        else:
            grant = (
                _load_object(Path(args.lease_grant), "lease grant")
                if args.lease_grant is not None
                else None
            )
            receipt = archive_candidate(
                spec,
                training_runs,
                evaluation_plan,
                evaluation_results,
                worker_id=args.worker_id,
                lease_grant=grant,
            )
        encoded = json.dumps(
            receipt,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        if output_path is not None:
            output_path.write_text(encoded + "\n", encoding="utf-8")
        else:
            print(encoded)
    except (OSError, SpecError, StorageError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Exchange immutable distributed tuning jobs and results through Git branches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from build_trial_plan import build_confirmation_runs, validate_trial_plan
from merge_historical_priors import (
    _validate_index as _validate_history_index,
    merge_history_indexes,
)
from rank_trials import select_confirmation_candidates
from validate_session_spec import SpecError, load_and_validate


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
EVENT_LIMIT_BYTES = 1_000_000


class MailboxError(RuntimeError):
    """Raised when the Git mailbox cannot preserve its safety contract."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MailboxError("mailbox data must be finite JSON") from exc
    data = encoded.encode("utf-8")
    if len(data) > EVENT_LIMIT_BYTES:
        raise MailboxError(
            f"mailbox JSON exceeds {EVENT_LIMIT_BYTES} bytes; publish metadata only"
        )
    return data


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MailboxError(f"JSON file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise MailboxError(
            f"invalid JSON in {source} at line {exc.lineno}: {exc.msg}"
        ) from exc


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _age_seconds(timestamp: Any) -> float | None:
    if not isinstance(timestamp, str):
        return None
    try:
        observed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        return None
    return max(0.0, (datetime.now(UTC) - observed.astimezone(UTC)).total_seconds())


def _git(
    repo: Path,
    *args: str,
    check: bool = True,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    command = ["git", "-C", str(repo), *args]
    result = subprocess.run(
        command,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise MailboxError(f"Git command failed: {' '.join(args)}: {detail}")
    return result


def _repo(path: str | Path) -> Path:
    repo = Path(path).resolve()
    if not repo.is_dir():
        raise MailboxError(f"coordination repository does not exist: {repo}")
    result = _git(repo, "rev-parse", "--show-toplevel")
    top = Path(result.stdout.decode().strip()).resolve()
    if top != repo:
        raise MailboxError("coordination repository path must be its Git worktree root")
    return repo


def _require_clean(repo: Path) -> None:
    status = _git(repo, "status", "--porcelain=v1").stdout.decode()
    if status:
        raise MailboxError(
            "coordination repository must be clean before a mailbox mutation"
        )


def _verify_remote(repo: Path, spec: dict[str, Any]) -> None:
    actual = _git(repo, "remote", "get-url", "origin").stdout.decode().strip()
    expected = spec["distributed"]["remote_url"]
    if actual == expected:
        return
    if os.environ.get("GIT_MAILBOX_ALLOW_TEST_REMOTE") == "1":
        return
    raise MailboxError(
        "origin URL does not exactly match distributed.remote_url; "
        "credentials must remain in a credential helper, not the session"
    )


def _fetch(repo: Path) -> None:
    _git(repo, "fetch", "--prune", "origin")


def _branch_exists(repo: Path, ref: str) -> bool:
    return _git(repo, "show-ref", "--verify", "--quiet", ref, check=False).returncode == 0


def _checkout_coordinator(repo: Path, branch: str) -> None:
    _require_clean(repo)
    remote_ref = f"refs/remotes/origin/{branch}"
    local_ref = f"refs/heads/{branch}"
    if _branch_exists(repo, local_ref):
        _git(repo, "checkout", branch)
        if _branch_exists(repo, remote_ref):
            _git(repo, "merge", "--ff-only", f"origin/{branch}")
        return
    if _branch_exists(repo, remote_ref):
        _git(repo, "checkout", "-b", branch, "--track", f"origin/{branch}")
        return
    _git(repo, "checkout", "-b", branch)


def _checkout_worker(repo: Path, worker: dict[str, Any], coordinator_branch: str) -> None:
    _require_clean(repo)
    branch = worker["branch"]
    local_ref = f"refs/heads/{branch}"
    remote_ref = f"refs/remotes/origin/{branch}"
    if _branch_exists(repo, local_ref):
        _git(repo, "checkout", branch)
        if _branch_exists(repo, remote_ref):
            _git(repo, "merge", "--ff-only", f"origin/{branch}")
        return
    if _branch_exists(repo, remote_ref):
        _git(repo, "checkout", "-b", branch, "--track", f"origin/{branch}")
        return
    coordinator_ref = f"refs/remotes/origin/{coordinator_branch}"
    if not _branch_exists(repo, coordinator_ref):
        raise MailboxError("coordinator branch has not been published")
    _git(repo, "checkout", "-b", branch, f"origin/{coordinator_branch}")


def _safe_relative(path: str) -> PurePosixPath:
    relative = PurePosixPath(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise MailboxError(f"unsafe mailbox path: {path}")
    return relative


def _read_ref_json(repo: Path, ref: str, relative: str) -> Any | None:
    safe = _safe_relative(relative)
    result = _git(repo, "show", f"{ref}:{safe.as_posix()}", check=False)
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise MailboxError(f"invalid JSON at {ref}:{safe}") from exc


def _list_ref(repo: Path, ref: str, prefix: str) -> list[str]:
    safe = _safe_relative(prefix)
    result = _git(repo, "ls-tree", "-r", "--name-only", ref, "--", safe.as_posix())
    return [line for line in result.stdout.decode().splitlines() if line]


def _write_immutable(repo: Path, relative: str, value: Any) -> bool:
    safe = _safe_relative(relative)
    target = repo.joinpath(*safe.parts)
    data = _canonical_bytes(value) + b"\n"
    if target.exists():
        if target.read_bytes() != data:
            raise MailboxError(f"immutable mailbox collision at {safe}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return True


def _commit_and_push(repo: Path, branch: str, paths: list[str], message: str) -> None:
    for path in paths:
        _git(repo, "add", "--", _safe_relative(path).as_posix())
    staged = _git(repo, "diff", "--cached", "--quiet", check=False)
    if staged.returncode == 1:
        _git(
            repo,
            "-c",
            "user.name=IsaacLab Git Mailbox",
            "-c",
            "user.email=git-mailbox@localhost",
            "commit",
            "-m",
            message,
        )
    elif staged.returncode != 0:
        raise MailboxError("could not inspect staged mailbox changes")
    _git(repo, "push", "-u", "origin", branch)


def _worker(spec: dict[str, Any], worker_id: str) -> dict[str, Any]:
    for worker in spec["distributed"]["workers"]:
        if worker["id"] == worker_id:
            return worker
    raise MailboxError(f"unknown worker: {worker_id}")


def _campaign_root(spec: dict[str, Any]) -> str:
    return f"campaigns/{spec['distributed']['campaign_id']}"


def _job_path(spec: dict[str, Any], worker_id: str, job_id: str) -> str:
    return f"{_campaign_root(spec)}/jobs/{worker_id}/{job_id}.json"


def _event_path(
    spec: dict[str, Any],
    category: str,
    worker_id: str,
    job_id: str,
    attempt: int,
    suffix: str = "",
) -> str:
    name = f"attempt-{attempt}{suffix}.json"
    return f"{_campaign_root(spec)}/{category}/{worker_id}/{job_id}/{name}"


def _load_plan(path: str | Path, spec: dict[str, Any]) -> dict[str, Any]:
    plan = _load_json(path)
    if not isinstance(plan, dict) or plan.get("version") not in {4, 5, 6}:
        raise MailboxError(
            "distributed publication requires a supported staged plan"
        )
    if plan.get("run_id") != spec["training"]["run_id"]:
        raise MailboxError("trial plan run_id does not match the session")
    try:
        validate_trial_plan(spec, plan)
    except SpecError as exc:
        raise MailboxError(
            "immutable mailbox collision or invalid deterministic trial plan"
        ) from exc
    runs = plan.get("runs")
    if not isinstance(runs, list):
        raise MailboxError("trial plan runs must be an array")
    run_ids: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise MailboxError(f"trial plan runs[{index}] must be an object")
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or not run_id
            or run_id in run_ids
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id)
        ):
            raise MailboxError(f"trial plan runs[{index}].run_id is invalid or duplicate")
        run_ids.add(run_id)
    return plan


def _history_index_path(spec: dict[str, Any], worker_id: str) -> str:
    return f"{_campaign_root(spec)}/history/indexes/{worker_id}.json"


def _adaptive_round_path(spec: dict[str, Any], round_number: int) -> str:
    return (
        f"{_campaign_root(spec)}/adaptive-rounds/"
        f"round-{round_number:03d}.json"
    )


def _multifidelity_decision_path(
    spec: dict[str, Any],
    decision: int,
) -> str:
    return (
        f"{_campaign_root(spec)}/multi-fidelity/"
        f"decision-{decision:03d}.json"
    )


def _plan_snapshot_path(spec: dict[str, Any], plan: dict[str, Any]) -> str:
    return f"{_campaign_root(spec)}/plans/{_sha256(plan)}.json"


def _assigned_worker(
    spec: dict[str, Any],
    run: dict[str, Any],
) -> dict[str, Any]:
    assignment_mode = spec["distributed"].get("assignment_mode", "by_seed")
    if assignment_mode == "by_trial":
        workers = spec["distributed"]["workers"]
        trial_id = run["trial_id"]
        if trial_id == "baseline":
            return _worker(spec, spec["distributed"]["coordinator_id"])
        match = re.fullmatch(r"trial-(\d+)", trial_id)
        if match is None:
            raise MailboxError(
                f"by_trial assignment requires trial-NNN identifiers, got {trial_id}"
            )
        return workers[(int(match.group(1)) - 1) % len(workers)]
    seed = run["seed"]
    matches = [
        worker
        for worker in spec["distributed"]["workers"]
        if seed in worker["assigned_seeds"]
    ]
    if len(matches) != 1:
        raise MailboxError(f"seed {seed} does not resolve to exactly one worker")
    return matches[0]


def _build_jobs(
    spec: dict[str, Any],
    plan: dict[str, Any],
    *,
    runs: list[dict[str, Any]] | None = None,
    include_calibration: bool = True,
    selection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    session_sha = _sha256(spec)
    plan_sha = _sha256(plan)
    jobs: list[dict[str, Any]] = []
    for run in plan["runs"] if runs is None else runs:
        worker = _assigned_worker(spec, run)
        job = {
            "schema_version": 1,
            "campaign_id": spec["distributed"]["campaign_id"],
            "job_id": run["run_id"],
            "kind": "trial",
            "worker_id": worker["id"],
            "worker_branch": worker["branch"],
            "source_git_commit": spec["training"]["source_git_commit"],
            "session_sha256": session_sha,
            "plan_sha256": plan_sha,
            "run": run,
        }
        if selection is not None:
            job["selection_sha256"] = _sha256(selection)
            job["selection"] = selection
        jobs.append(job)
    if not include_calibration:
        return jobs
    calibration = spec["distributed"]["calibration"]
    if not calibration["enabled"]:
        return jobs
    baseline = next(
        (trial for trial in plan["trials"] if trial["trial_id"] == "baseline"),
        None,
    )
    if baseline is None:
        raise MailboxError("trial plan does not contain an unchanged baseline")
    for worker_id in calibration["worker_ids"]:
        worker = _worker(spec, worker_id)
        job_id = (
            f"{spec['distributed']['campaign_id']}--calibration--"
            f"{worker_id}--seed-{calibration['seed']}"
        )
        jobs.append(
            {
                "schema_version": 1,
                "campaign_id": spec["distributed"]["campaign_id"],
                "job_id": job_id,
                "kind": "calibration",
                "worker_id": worker_id,
                "worker_branch": worker["branch"],
                "source_git_commit": spec["training"]["source_git_commit"],
                "session_sha256": session_sha,
                "plan_sha256": plan_sha,
                "run": {
                    "run_id": job_id,
                    "stage": "calibration",
                    "trial_id": "baseline",
                    "seed": calibration["seed"],
                    "overrides": {},
                },
            }
        )
    return jobs


def _inspect_source(worker: dict[str, Any], expected_commit: str) -> dict[str, Any]:
    source = Path(worker["source_repo"]).resolve()
    if not source.is_dir():
        raise MailboxError(f"worker source repository does not exist: {source}")
    head = _git(source, "rev-parse", "HEAD").stdout.decode().strip()
    if not GIT_SHA_RE.fullmatch(head) or head != expected_commit:
        raise MailboxError(
            f"worker source HEAD {head} does not match approved commit {expected_commit}"
        )
    status = _git(source, "status", "--porcelain=v1").stdout.decode()
    if status:
        raise MailboxError("worker source repository must be clean before claiming a job")
    return {"source_repo": str(source), "source_git_commit": head, "source_git_dirty": False}


def history_publish(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    index_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    _inspect_source(worker, spec["training"]["source_git_commit"])
    try:
        index = _validate_history_index(
            spec,
            _load_json(index_path),
        )
    except SpecError as exc:
        raise MailboxError("local history index fails validation") from exc
    _checkout_worker(repo, worker, spec["distributed"]["coordinator_branch"])
    path = _history_index_path(spec, worker_id)
    _write_immutable(repo, path, index)
    _commit_and_push(
        repo,
        worker["branch"],
        [path],
        f"mailbox: publish bounded history {worker_id}",
    )
    return {
        "state": "history_index_published",
        "worker_id": worker_id,
        "index_sha256": index["index_sha256"],
    }


def history_initialize(
    repo_path: str | Path,
    session_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    history = spec.get("history_prior")
    if not isinstance(history, dict) or not history.get("enabled"):
        raise MailboxError("session does not enable bounded history collection")
    _verify_remote(repo, spec)
    _fetch(repo)
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    manifest = {
        "schema_version": 1,
        "event": "history_collection_initialized",
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "coordinator_id": spec["distributed"]["coordinator_id"],
        "worker_ids": list(history["worker_roots"]),
        "max_selected_runs": history["max_selected_runs"],
        "max_points_per_run": history["max_points_per_run"],
        "history_index_schema_version": 2,
        "source_policy": history["compatibility"]["source_policy"],
        "expected_context_sha256": _sha256(
            history["compatibility"]["expected_context"]
        ),
        "quality_gates_sha256": _sha256(history["quality_gates"]),
        "artifact_policy": "metadata_only",
    }
    path = f"{_campaign_root(spec)}/history/collection.json"
    _write_immutable(repo, path, manifest)
    _commit_and_push(
        repo,
        branch,
        [path],
        f"mailbox: initialize bounded history {spec['distributed']['campaign_id']}",
    )
    return {
        "state": "history_collection_initialized",
        "session_sha256": manifest["session_sha256"],
    }


def history_collect(
    repo_path: str | Path,
    session_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    history = spec.get("history_prior")
    if not isinstance(history, dict) or not history.get("enabled"):
        raise MailboxError("session does not enable bounded history collection")
    _verify_remote(repo, spec)
    _fetch(repo)
    indexes: list[dict[str, Any]] = []
    for worker_id in history["worker_roots"]:
        worker = _worker(spec, worker_id)
        ref = f"origin/{worker['branch']}"
        index = _read_ref_json(
            repo,
            ref,
            _history_index_path(spec, worker_id),
        )
        if not isinstance(index, dict):
            raise MailboxError(
                f"worker history index is not published: {worker_id}"
            )
        indexes.append(index)
    try:
        prior = merge_history_indexes(spec, indexes)
    except SpecError as exc:
        raise MailboxError("worker history indexes cannot be merged") from exc
    output = Path(output_path)
    if (
        not output.is_absolute()
        or output.exists()
        or not output.parent.is_dir()
        or output.parent.is_symlink()
    ):
        raise MailboxError(
            "history prior output must be a new absolute file under an "
            "existing regular parent"
        )
    output.write_bytes(_canonical_bytes(prior) + b"\n")
    return {
        "state": "history_prior_collected",
        "selected_run_count": prior["selected_run_count"],
        "guidance_eligible_count": prior["guidance_eligible_count"],
        "prior_sha256": prior["prior_sha256"],
        "output": str(output),
    }


def publish(repo_path: str | Path, session_path: str | Path, plan_path: str | Path) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    if spec.get("version") != 7:
        raise MailboxError("Git mailbox publication requires a version-7 session")
    plan = _load_plan(plan_path, spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    jobs = _build_jobs(spec, plan)
    root = _campaign_root(spec)
    manifest = {
        "schema_version": 1,
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "plan_sha256": _sha256(plan),
        "source_git_commit": spec["training"]["source_git_commit"],
        "coordinator_id": spec["distributed"]["coordinator_id"],
        "coordinator_branch": branch,
        "artifact_policy": "metadata_only",
        "worker_branches": {
            worker["id"]: worker["branch"]
            for worker in spec["distributed"]["workers"]
        },
    }
    changed: list[str] = []
    plan_path = _plan_snapshot_path(spec, plan)
    if _write_immutable(repo, plan_path, plan):
        changed.append(plan_path)
    manifest_path = f"{root}/campaign.json"
    if _write_immutable(repo, manifest_path, manifest):
        changed.append(manifest_path)
    for job in jobs:
        path = _job_path(spec, job["worker_id"], job["job_id"])
        if _write_immutable(repo, path, job):
            changed.append(path)
    _commit_and_push(
        repo,
        branch,
        changed,
        f"mailbox: publish {spec['distributed']['campaign_id']}",
    )
    return {"campaign_id": spec["distributed"]["campaign_id"], "published_jobs": len(jobs)}


def publish_adaptive_round(
    repo_path: str | Path,
    session_path: str | Path,
    previous_plan_path: str | Path,
    expanded_plan_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    previous = _load_plan(previous_plan_path, spec)
    expanded = _load_plan(expanded_plan_path, spec)
    previous_decisions = previous.get("adaptive", {}).get("decisions")
    expanded_decisions = expanded.get("adaptive", {}).get("decisions")
    new_decision = (
        expanded_decisions[-1]
        if isinstance(expanded_decisions, list) and expanded_decisions
        else None
    )
    stopped = (
        isinstance(new_decision, dict)
        and new_decision.get("action") == "stop"
    )
    expected_round_delta = 0 if stopped else 1
    if (
        previous.get("version") != 5
        or expanded.get("version") != 5
        or not isinstance(previous_decisions, list)
        or not isinstance(expanded_decisions, list)
        or len(expanded_decisions) != len(previous_decisions) + 1
        or expanded_decisions[: len(previous_decisions)]
        != previous_decisions
        or len(expanded["adaptive"]["rounds"])
        != len(previous["adaptive"]["rounds"]) + expected_round_delta
        or expanded["trials"][: len(previous["trials"])]
        != previous["trials"]
        or expanded["runs"][: len(previous["runs"])] != previous["runs"]
        or (
            stopped
            and (
                expanded["trials"] != previous["trials"]
                or expanded["runs"] != previous["runs"]
            )
        )
    ):
        raise MailboxError(
            "adaptive publication requires one exact decision and optional round"
        )
    _verify_remote(repo, spec)
    _fetch(repo)
    collected = _collect_report(repo, spec)
    expected_runs = {
        run["run_id"]: run for run in previous["runs"]
    }
    observed_job_ids: set[str] = set()
    accepted_results: list[dict[str, Any]] = []
    for envelope in collected["accepted_results"]:
        job_id = envelope.get("job_id")
        if job_id not in expected_runs:
            continue
        job = _coordinator_job(
            repo,
            spec,
            envelope["worker_id"],
            job_id,
        )
        result_value = envelope.get("result")
        expected_run = expected_runs[job_id]
        if (
            job.get("kind") != "trial"
            or job.get("run") != expected_run
            or not isinstance(result_value, dict)
            or result_value.get("trial_id") != expected_run["trial_id"]
            or result_value.get("seed") != expected_run["seed"]
            or result_value.get("status") != "completed"
        ):
            raise MailboxError(
                "adaptive result identity or completion state is invalid"
            )
        observed_job_ids.add(job_id)
        accepted_results.append(result_value)
    accepted_results.sort(key=lambda item: item["trial_id"])
    if (
        collected["invalid_results"]
        or observed_job_ids != set(expected_runs)
        or accepted_results != new_decision.get("input_results")
    ):
        raise MailboxError(
            "adaptive round inputs do not exactly match valid mailbox results"
        )
    new_runs = expanded["runs"][len(previous["runs"]):]
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    jobs = _build_jobs(
        spec,
        expanded,
        runs=new_runs,
        include_calibration=False,
    )
    manifest = {
        "schema_version": 1,
        "event": (
            "adaptive_search_stopped"
            if stopped
            else "adaptive_round_published"
        ),
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "decision": new_decision["decision"],
        "evaluated_round": new_decision["evaluated_round"],
        "previous_plan_sha256": _sha256(previous),
        "expanded_plan_sha256": _sha256(expanded),
        "input_results_sha256": new_decision["input_results_sha256"],
        "action": new_decision["action"],
        "reason": new_decision["reason"],
        "job_ids": [job["job_id"] for job in jobs],
    }
    changed: list[str] = []
    plan_path = _plan_snapshot_path(spec, expanded)
    if _write_immutable(repo, plan_path, expanded):
        changed.append(plan_path)
    manifest_path = _adaptive_round_path(spec, new_decision["decision"])
    if _write_immutable(repo, manifest_path, manifest):
        changed.append(manifest_path)
    for job in jobs:
        path = _job_path(spec, job["worker_id"], job["job_id"])
        if _write_immutable(repo, path, job):
            changed.append(path)
    _commit_and_push(
        repo,
        branch,
        changed,
        (
            f"mailbox: stop adaptive search {new_decision['decision']}"
            if stopped
            else f"mailbox: publish adaptive round {new_decision['decision'] + 1}"
        ),
    )
    return {
        "state": (
            "adaptive_search_stopped"
            if stopped
            else "adaptive_round_published"
        ),
        "decision": new_decision["decision"],
        "reason": new_decision["reason"],
        "published_jobs": len(jobs),
        "expanded_plan_sha256": manifest["expanded_plan_sha256"],
    }


def publish_multifidelity_rung(
    repo_path: str | Path,
    session_path: str | Path,
    previous_plan_path: str | Path,
    expanded_plan_path: str | Path,
) -> dict[str, Any]:
    """Publish one hash-bound synchronized rung decision and its jobs."""
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    previous = _load_plan(previous_plan_path, spec)
    expanded = _load_plan(expanded_plan_path, spec)
    previous_fidelity = previous.get("multi_fidelity")
    expanded_fidelity = expanded.get("multi_fidelity")
    previous_decisions = (
        previous_fidelity.get("decisions")
        if isinstance(previous_fidelity, dict)
        else None
    )
    expanded_decisions = (
        expanded_fidelity.get("decisions")
        if isinstance(expanded_fidelity, dict)
        else None
    )
    new_decision = (
        expanded_decisions[-1]
        if isinstance(expanded_decisions, list) and expanded_decisions
        else None
    )
    action = (
        new_decision.get("action")
        if isinstance(new_decision, dict)
        else None
    )
    terminal = action in {"stop", "complete"}
    if (
        previous.get("version") != 6
        or expanded.get("version") != 6
        or not isinstance(previous_fidelity, dict)
        or not isinstance(expanded_fidelity, dict)
        or not isinstance(previous_decisions, list)
        or not isinstance(expanded_decisions, list)
        or len(expanded_decisions) != len(previous_decisions) + 1
        or expanded_decisions[: len(previous_decisions)]
        != previous_decisions
        or action not in {"continue", "stop", "complete"}
        or len(expanded_fidelity["rungs"])
        != len(previous_fidelity["rungs"]) + (0 if terminal else 1)
        or expanded["trials"] != previous["trials"]
        or expanded["runs"][: len(previous["runs"])] != previous["runs"]
        or (terminal and expanded["runs"] != previous["runs"])
        or (
            action == "continue"
            and len(expanded["runs"]) <= len(previous["runs"])
        )
    ):
        raise MailboxError(
            "multi-fidelity publication requires one exact decision and "
            "optional synchronized rung"
        )
    _verify_remote(repo, spec)
    _fetch(repo)
    collected = _collect_report(repo, spec)
    evaluated_rung = new_decision["evaluated_rung"]
    expected_runs = {
        run["run_id"]: run
        for run in previous["runs"]
        if run.get("rung") == evaluated_rung
    }
    observed_job_ids: set[str] = set()
    accepted_results: list[dict[str, Any]] = []
    parent_workers: dict[str, str] = {}
    for envelope in collected["accepted_results"]:
        job_id = envelope.get("job_id")
        if job_id not in expected_runs:
            continue
        job = _coordinator_job(
            repo,
            spec,
            envelope["worker_id"],
            job_id,
        )
        result_value = envelope.get("result")
        expected_run = expected_runs[job_id]
        artifacts = envelope.get("artifact_manifest", {}).get("artifacts")
        checkpoint = (
            result_value.get("checkpoint")
            if isinstance(result_value, dict)
            else None
        )
        checkpoint_artifacts = (
            [
                artifact
                for artifact in artifacts
                if isinstance(artifact, dict)
                and artifact.get("kind") == "checkpoint"
            ]
            if isinstance(artifacts, list)
            else []
        )
        if (
            job.get("kind") != "trial"
            or job.get("run") != expected_run
            or not isinstance(result_value, dict)
            or result_value.get("run_id") != job_id
            or result_value.get("trial_id") != expected_run["trial_id"]
            or result_value.get("seed") != expected_run["seed"]
            or result_value.get("status") != "completed"
            or result_value.get("rung") != evaluated_rung
            or not isinstance(checkpoint, dict)
            or len(checkpoint_artifacts) != 1
            or checkpoint_artifacts[0].get("path") != checkpoint.get("path")
            or checkpoint_artifacts[0].get("sha256")
            != checkpoint.get("sha256")
        ):
            raise MailboxError(
                "multi-fidelity result or checkpoint evidence is invalid"
            )
        observed_job_ids.add(job_id)
        accepted_results.append(result_value)
        parent_workers[job_id] = envelope["worker_id"]
    accepted_results.sort(key=lambda item: item["run_id"])
    if (
        collected["invalid_results"]
        or observed_job_ids != set(expected_runs)
        or accepted_results != new_decision.get("input_results")
    ):
        raise MailboxError(
            "multi-fidelity decision inputs do not exactly match valid mailbox "
            "results"
        )
    new_runs = expanded["runs"][len(previous["runs"]):]
    for run in new_runs:
        resume = run.get("resume_from")
        worker = _assigned_worker(spec, run)
        if (
            not isinstance(resume, dict)
            or parent_workers.get(resume.get("parent_run_id"))
            != worker["id"]
        ):
            raise MailboxError(
                "multi-fidelity promotion must remain on the checkpoint's "
                "original worker"
            )
    jobs = _build_jobs(
        spec,
        expanded,
        runs=new_runs,
        include_calibration=False,
    )
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    manifest = {
        "schema_version": 1,
        "event": {
            "continue": "multi_fidelity_rung_published",
            "complete": "multi_fidelity_completed",
            "stop": "multi_fidelity_stopped",
        }[action],
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "decision": new_decision["decision"],
        "evaluated_rung": evaluated_rung,
        "previous_plan_sha256": _sha256(previous),
        "expanded_plan_sha256": _sha256(expanded),
        "input_results_sha256": new_decision["input_results_sha256"],
        "action": action,
        "reason": new_decision["reason"],
        "promoted_trial_ids": new_decision["promoted_trial_ids"],
        "job_ids": [job["job_id"] for job in jobs],
    }
    changed: list[str] = []
    plan_path = _plan_snapshot_path(spec, expanded)
    if _write_immutable(repo, plan_path, expanded):
        changed.append(plan_path)
    manifest_path = _multifidelity_decision_path(
        spec,
        new_decision["decision"],
    )
    if _write_immutable(repo, manifest_path, manifest):
        changed.append(manifest_path)
    for job in jobs:
        path = _job_path(spec, job["worker_id"], job["job_id"])
        if _write_immutable(repo, path, job):
            changed.append(path)
    _commit_and_push(
        repo,
        branch,
        changed,
        (
            f"mailbox: publish multi-fidelity decision "
            f"{new_decision['decision']}"
        ),
    )
    return {
        "state": manifest["event"],
        "decision": new_decision["decision"],
        "reason": new_decision["reason"],
        "published_jobs": len(jobs),
        "expanded_plan_sha256": manifest["expanded_plan_sha256"],
    }


def _coordinator_job(
    repo: Path,
    spec: dict[str, Any],
    worker_id: str,
    job_id: str,
) -> dict[str, Any]:
    ref = f"origin/{spec['distributed']['coordinator_branch']}"
    job = _read_ref_json(repo, ref, _job_path(spec, worker_id, job_id))
    if not isinstance(job, dict):
        raise MailboxError(f"job is not published for worker {worker_id}: {job_id}")
    if (
        job.get("worker_id") != worker_id
        or job.get("session_sha256") != _sha256(spec)
        or job.get("source_git_commit") != spec["training"]["source_git_commit"]
    ):
        raise MailboxError("published job does not match the approved session")
    return job


def _cancel_path(spec: dict[str, Any], worker_id: str, job_id: str) -> str:
    return f"{_campaign_root(spec)}/control/cancel/{worker_id}/{job_id}.json"


def _archive_lease_config(spec: dict[str, Any]) -> dict[str, Any]:
    archive = spec.get("archive")
    lease = archive.get("distributed_lease") if isinstance(archive, dict) else None
    if (
        spec.get("version") != 7
        or not isinstance(lease, dict)
        or not lease.get("enabled")
    ):
        raise MailboxError(
            "session does not enable distributed policy archive leases"
        )
    return lease


def _archive_request_path(
    spec: dict[str, Any],
    worker_id: str,
    request_id: str,
) -> str:
    return (
        f"{_campaign_root(spec)}/policy-archive/requests/"
        f"{worker_id}/{request_id}.json"
    )


def _archive_grant_path(spec: dict[str, Any], lease_id: str) -> str:
    return f"{_campaign_root(spec)}/policy-archive/grants/{lease_id}.json"


def _archive_completion_path(
    spec: dict[str, Any],
    worker_id: str,
    lease_id: str,
) -> str:
    return (
        f"{_campaign_root(spec)}/policy-archive/completions/"
        f"{worker_id}/{lease_id}.json"
    )


def _archive_closure_path(spec: dict[str, Any], lease_id: str) -> str:
    return f"{_campaign_root(spec)}/policy-archive/closures/{lease_id}.json"


def _validate_archive_request(
    spec: dict[str, Any],
    worker_id: str,
    request: Any,
) -> dict[str, Any]:
    lease = _archive_lease_config(spec)
    if worker_id not in lease["authorized_worker_ids"]:
        raise MailboxError("worker is not authorized to request archive leases")
    if not isinstance(request, dict):
        raise MailboxError("archive request must be a JSON object")
    required = {
        "schema_version",
        "event",
        "campaign_id",
        "session_sha256",
        "worker_id",
        "candidate_id",
        "artifacts",
        "storage_remote_url",
        "storage_branch",
        "storage_base_commit",
        "request_id",
    }
    if set(request) != required:
        raise MailboxError(
            "archive request fields do not match the version-1 schema"
        )
    if (
        request["schema_version"] != 1
        or request["event"] != "policy_archive_requested"
        or request["campaign_id"] != spec["distributed"]["campaign_id"]
        or request["session_sha256"] != _sha256(spec)
        or request["worker_id"] != worker_id
        or request["storage_remote_url"] != lease["storage_remote_url"]
        or request["storage_branch"] != lease["storage_branch"]
        or not isinstance(request["candidate_id"], str)
        or not request["candidate_id"]
        or not isinstance(request["storage_base_commit"], str)
        or not GIT_SHA_RE.fullmatch(request["storage_base_commit"])
    ):
        raise MailboxError("archive request identity does not match the session")
    artifacts = request["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"jit", "onnx"}:
        raise MailboxError("archive request must bind exactly JIT and ONNX")
    for kind in ("jit", "onnx"):
        record = artifacts[kind]
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256"}
            or not isinstance(record["sha256"], str)
            or not SHA256_RE.fullmatch(record["sha256"])
        ):
            raise MailboxError(f"archive request {kind} hash is invalid")
    identity = dict(request)
    request_id = identity.pop("request_id")
    if not isinstance(request_id, str) or request_id != _sha256(identity):
        raise MailboxError("archive request ID does not match its identity hash")
    return request


def _validate_archive_grant(
    spec: dict[str, Any],
    grant: Any,
) -> dict[str, Any]:
    if not isinstance(grant, dict):
        raise MailboxError("archive grant must be a JSON object")
    required = {
        "schema_version",
        "event",
        "campaign_id",
        "session_sha256",
        "lease_id",
        "worker_id",
        "coordinator_id",
        "request_sha256",
        "request",
        "granted_at",
        "takeover_policy",
    }
    if set(grant) != required:
        raise MailboxError("archive grant fields do not match the version-1 schema")
    worker_id = grant.get("worker_id")
    if not isinstance(worker_id, str):
        raise MailboxError("archive grant worker is invalid")
    request = _validate_archive_request(spec, worker_id, grant.get("request"))
    if (
        grant.get("schema_version") != 1
        or grant.get("event") != "policy_archive_granted"
        or grant.get("campaign_id") != spec["distributed"]["campaign_id"]
        or grant.get("session_sha256") != _sha256(spec)
        or grant.get("lease_id") != request["request_id"]
        or grant.get("coordinator_id")
        != spec["distributed"]["coordinator_id"]
        or grant.get("request_sha256") != _sha256(request)
        or not isinstance(grant.get("granted_at"), str)
        or not grant["granted_at"]
        or grant.get("takeover_policy") != "explicit_revoke_only"
    ):
        raise MailboxError("archive grant identity does not match the session")
    return grant


def _validate_archive_closure(
    spec: dict[str, Any],
    grant: dict[str, Any],
    closure: Any,
) -> dict[str, Any]:
    if not isinstance(closure, dict):
        raise MailboxError("archive closure must be a JSON object")
    common = {
        "schema_version",
        "event",
        "campaign_id",
        "session_sha256",
        "lease_id",
        "worker_id",
        "closed_at",
    }
    event = closure.get("event")
    if event == "policy_archive_released":
        required = common | {"storage_commit", "completion_sha256"}
        details_valid = (
            isinstance(closure.get("storage_commit"), str)
            and GIT_SHA_RE.fullmatch(closure["storage_commit"]) is not None
            and isinstance(closure.get("completion_sha256"), str)
            and SHA256_RE.fullmatch(closure["completion_sha256"]) is not None
        )
    elif event == "policy_archive_revoked":
        required = common | {"reason", "automatic_takeover"}
        details_valid = (
            isinstance(closure.get("reason"), str)
            and 1 <= len(closure["reason"].strip()) <= 1000
            and closure.get("automatic_takeover") is False
        )
    else:
        raise MailboxError("archive closure event is invalid")
    if set(closure) != required:
        raise MailboxError("archive closure fields do not match the version-1 schema")
    if (
        closure.get("schema_version") != 1
        or closure.get("campaign_id") != spec["distributed"]["campaign_id"]
        or closure.get("session_sha256") != _sha256(spec)
        or closure.get("lease_id") != grant["lease_id"]
        or closure.get("worker_id") != grant["worker_id"]
        or not isinstance(closure.get("closed_at"), str)
        or not closure["closed_at"]
        or not details_valid
    ):
        raise MailboxError("archive closure identity does not match the grant")
    return closure


def _archive_lease_state(
    repo: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    coordinator_ref = f"origin/{spec['distributed']['coordinator_branch']}"
    grants: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    prefix = f"{_campaign_root(spec)}/policy-archive/grants"
    for path in _list_ref(repo, coordinator_ref, prefix):
        grant = _validate_archive_grant(
            spec,
            _read_ref_json(repo, coordinator_ref, path),
        )
        lease_id = grant["lease_id"]
        closure = _read_ref_json(
            repo,
            coordinator_ref,
            _archive_closure_path(spec, lease_id),
        )
        if closure is not None:
            closure = _validate_archive_closure(spec, grant, closure)
        entry = {"grant": grant, "closure": closure}
        grants.append(entry)
        if closure is None:
            active.append(grant)
    if len(active) > 1:
        raise MailboxError("multiple active policy archive leases detected")
    return {
        "active_grant": active[0] if active else None,
        "grants": grants,
    }


def status(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    inspect_source: bool = True,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    if inspect_source:
        _inspect_source(worker, spec["training"]["source_git_commit"])
    coordinator_ref = f"origin/{spec['distributed']['coordinator_branch']}"
    prefix = f"{_campaign_root(spec)}/jobs/{worker_id}"
    job_paths = _list_ref(repo, coordinator_ref, prefix)
    worker_ref = (
        f"origin/{worker['branch']}"
        if _branch_exists(repo, f"refs/remotes/origin/{worker['branch']}")
        else None
    )
    jobs: list[dict[str, Any]] = []
    for path in job_paths:
        job = _read_ref_json(repo, coordinator_ref, path)
        if not isinstance(job, dict):
            raise MailboxError(f"invalid job document at {path}")
        job_id = job["job_id"]
        cancelled = _read_ref_json(
            repo, coordinator_ref, _cancel_path(spec, worker_id, job_id)
        ) is not None
        receipts = (
            _list_ref(
                repo,
                worker_ref,
                f"{_campaign_root(spec)}/receipts/{worker_id}/{job_id}",
            )
            if worker_ref
            else []
        )
        results = (
            _list_ref(
                repo,
                worker_ref,
                f"{_campaign_root(spec)}/results/{worker_id}/{job_id}",
            )
            if worker_ref
            else []
        )
        progress_paths = (
            _list_ref(
                repo,
                worker_ref,
                f"{_campaign_root(spec)}/progress/{worker_id}/{job_id}",
            )
            if worker_ref
            else []
        )
        state = "completed" if results else "claimed" if receipts else "pending"
        last_observed_at = None
        if progress_paths:
            latest_progress = _read_ref_json(
                repo, worker_ref, sorted(progress_paths)[-1]
            )
            if isinstance(latest_progress, dict):
                last_observed_at = latest_progress.get("observed_at")
        elif receipts:
            latest_receipt = _read_ref_json(
                repo, worker_ref, sorted(receipts)[-1]
            )
            if isinstance(latest_receipt, dict):
                last_observed_at = latest_receipt.get("claimed_at")
        age = _age_seconds(last_observed_at)
        if (
            state == "claimed"
            and age is not None
            and age >= spec["distributed"]["remote_state_unknown_after_seconds"]
        ):
            state = "remote_state_unknown"
        if cancelled and state != "completed":
            state = "cancel_requested"
        jobs.append(
            {
                "job_id": job_id,
                "kind": job["kind"],
                "state": state,
                "last_observed_at": last_observed_at,
                "seconds_since_remote_observation": age,
            }
        )
    return {"worker_id": worker_id, "jobs": jobs}


def claim(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    job_id: str,
    attempt: int,
) -> dict[str, Any]:
    if not 1 <= attempt <= 4:
        raise MailboxError("attempt must be between 1 and 4")
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    job = _coordinator_job(repo, spec, worker_id, job_id)
    coordinator_ref = f"origin/{spec['distributed']['coordinator_branch']}"
    if _read_ref_json(repo, coordinator_ref, _cancel_path(spec, worker_id, job_id)):
        raise MailboxError("job has a published cancel request")
    source = _inspect_source(worker, job["source_git_commit"])
    _checkout_worker(repo, worker, spec["distributed"]["coordinator_branch"])
    path = _event_path(spec, "receipts", worker_id, job_id, attempt)
    existing = _read_ref_json(repo, "HEAD", path)
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or existing.get("job_sha256") != _sha256(job)
            or existing.get("worker_id") != worker_id
            or existing.get("attempt") != attempt
        ):
            raise MailboxError("existing claim receipt fails identity binding")
        _commit_and_push(
            repo,
            worker["branch"],
            [path],
            f"mailbox: claim {job_id} attempt {attempt}",
        )
        return {"state": "claim_published", "job": job, "receipt": existing}
    receipt = {
        "schema_version": 1,
        "event": "claimed",
        "campaign_id": spec["distributed"]["campaign_id"],
        "job_id": job_id,
        "job_sha256": _sha256(job),
        "worker_id": worker_id,
        "attempt": attempt,
        "claimed_at": _utc_now(),
        **source,
    }
    _write_immutable(repo, path, receipt)
    _commit_and_push(
        repo,
        worker["branch"],
        [path],
        f"mailbox: claim {job_id} attempt {attempt}",
    )
    return {"state": "claim_published", "job": job, "receipt": receipt}


def prepare_job(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    job_id: str,
    attempt: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Materialize one remotely claimed job for the bounded local executor."""
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    job = _coordinator_job(repo, spec, worker_id, job_id)
    worker_ref = f"origin/{worker['branch']}"
    if not _branch_exists(repo, f"refs/remotes/origin/{worker['branch']}"):
        raise MailboxError("worker branch has not published a claim")
    receipt_path = _event_path(spec, "receipts", worker_id, job_id, attempt)
    receipt = _read_ref_json(repo, worker_ref, receipt_path)
    if (
        not isinstance(receipt, dict)
        or receipt.get("event") != "claimed"
        or receipt.get("job_sha256") != _sha256(job)
        or receipt.get("worker_id") != worker_id
        or receipt.get("attempt") != attempt
    ):
        raise MailboxError("remote claim receipt is missing or fails identity binding")
    _inspect_source(worker, job["source_git_commit"])
    coordinator_ref = f"origin/{spec['distributed']['coordinator_branch']}"
    plan = _read_ref_json(
        repo,
        coordinator_ref,
        f"{_campaign_root(spec)}/plans/{job['plan_sha256']}.json",
    )
    if plan is not None:
        if not isinstance(plan, dict) or _sha256(plan) != job["plan_sha256"]:
            raise MailboxError(
                "published job plan snapshot fails hash binding"
            )
        try:
            validate_trial_plan(spec, plan)
        except SpecError as exc:
            raise MailboxError("published job plan snapshot is invalid") from exc
    prepared = {
        "schema_version": 1,
        "job_sha256": _sha256(job),
        "receipt_sha256": _sha256(receipt),
        "job": job,
        "receipt": receipt,
    }
    if plan is not None:
        prepared["plan"] = plan
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(prepared) + b"\n")
    return {
        "state": "job_prepared",
        "job_id": job_id,
        "job_sha256": prepared["job_sha256"],
        "output": str(output),
    }


def progress(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    job_id: str,
    attempt: int,
    sequence: int,
    progress_path: str | Path,
) -> dict[str, Any]:
    if not 1 <= sequence <= 100000:
        raise MailboxError("progress sequence must be between 1 and 100000")
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    job = _coordinator_job(repo, spec, worker_id, job_id)
    _checkout_worker(repo, worker, spec["distributed"]["coordinator_branch"])
    receipt_path = _event_path(spec, "receipts", worker_id, job_id, attempt)
    receipt = _read_ref_json(repo, "HEAD", receipt_path)
    if not isinstance(receipt, dict) or receipt.get("job_sha256") != _sha256(job):
        raise MailboxError("publish a matching claim receipt before progress")
    path = _event_path(
        spec, "progress", worker_id, job_id, attempt, f"-{sequence:06d}"
    )
    existing = _read_ref_json(repo, "HEAD", path)
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or existing.get("job_sha256") != _sha256(job)
            or existing.get("sequence") != sequence
        ):
            raise MailboxError("existing progress event fails identity binding")
        _commit_and_push(
            repo,
            worker["branch"],
            [path],
            f"mailbox: progress {job_id} {sequence}",
        )
        return {"state": "progress_published", "sequence": sequence}
    payload = _load_json(progress_path)
    _canonical_bytes(payload)
    event = {
        "schema_version": 1,
        "event": "progress",
        "campaign_id": spec["distributed"]["campaign_id"],
        "job_id": job_id,
        "job_sha256": _sha256(job),
        "worker_id": worker_id,
        "attempt": attempt,
        "sequence": sequence,
        "observed_at": _utc_now(),
        "progress": payload,
    }
    _write_immutable(repo, path, event)
    _commit_and_push(
        repo,
        worker["branch"],
        [path],
        f"mailbox: progress {job_id} {sequence}",
    )
    return {"state": "progress_published", "sequence": sequence}


def _validate_artifacts(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"artifacts"}:
        raise MailboxError("artifact manifest must contain only artifacts")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) > 32:
        raise MailboxError("artifact manifest artifacts must be an array of at most 32")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != {
            "kind",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise MailboxError(
                f"artifacts[{index}] must contain kind, path, sha256, and size_bytes"
            )
        if not isinstance(artifact["kind"], str) or not artifact["kind"]:
            raise MailboxError(f"artifacts[{index}].kind must be non-empty")
        if not isinstance(artifact["path"], str) or not Path(artifact["path"]).is_absolute():
            raise MailboxError(f"artifacts[{index}].path must be absolute")
        if not isinstance(artifact["sha256"], str) or not SHA256_RE.fullmatch(
            artifact["sha256"]
        ):
            raise MailboxError(f"artifacts[{index}].sha256 must be lowercase SHA-256")
        size = artifact["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise MailboxError(f"artifacts[{index}].size_bytes must be non-negative")
    _canonical_bytes(value)
    return value


def result(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    job_id: str,
    attempt: int,
    result_path: str | Path,
    artifacts_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    job = _coordinator_job(repo, spec, worker_id, job_id)
    source = _inspect_source(worker, job["source_git_commit"])
    _checkout_worker(repo, worker, spec["distributed"]["coordinator_branch"])
    receipt_path = _event_path(spec, "receipts", worker_id, job_id, attempt)
    receipt = _read_ref_json(repo, "HEAD", receipt_path)
    if not isinstance(receipt, dict) or receipt.get("job_sha256") != _sha256(job):
        raise MailboxError("publish a matching claim receipt before a result")
    path = _event_path(spec, "results", worker_id, job_id, attempt)
    existing = _read_ref_json(repo, "HEAD", path)
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or existing.get("job_sha256") != _sha256(job)
            or existing.get("worker_id") != worker_id
            or existing.get("attempt") != attempt
        ):
            raise MailboxError("existing result fails identity binding")
        _commit_and_push(
            repo,
            worker["branch"],
            [path],
            f"mailbox: result {job_id} attempt {attempt}",
        )
        return {
            "state": "result_published",
            "result_sha256": existing.get("result_sha256"),
        }
    payload = _load_json(result_path)
    _canonical_bytes(payload)
    artifacts = _validate_artifacts(_load_json(artifacts_path))
    envelope = {
        "schema_version": 1,
        "event": "completed",
        "campaign_id": spec["distributed"]["campaign_id"],
        "job_id": job_id,
        "job_sha256": _sha256(job),
        "worker_id": worker_id,
        "attempt": attempt,
        "completed_at": _utc_now(),
        "source": source,
        "result_sha256": _sha256(payload),
        "result": payload,
        "artifact_manifest_sha256": _sha256(artifacts),
        "artifact_manifest": artifacts,
    }
    _write_immutable(repo, path, envelope)
    _commit_and_push(
        repo,
        worker["branch"],
        [path],
        f"mailbox: result {job_id} attempt {attempt}",
    )
    return {"state": "result_published", "result_sha256": envelope["result_sha256"]}


def cancel(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    job_id: str,
    reason: str,
) -> dict[str, Any]:
    if not reason.strip() or len(reason) > 1000:
        raise MailboxError("cancel reason must contain between 1 and 1000 characters")
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    _coordinator_job(repo, spec, worker_id, job_id)
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    path = _cancel_path(spec, worker_id, job_id)
    existing = _read_ref_json(repo, "HEAD", path)
    if existing is not None:
        if (
            not isinstance(existing, dict)
            or existing.get("job_id") != job_id
            or existing.get("worker_id") != worker_id
            or existing.get("reason") != reason
        ):
            raise MailboxError("existing cancel request differs from this request")
        _commit_and_push(repo, branch, [path], f"mailbox: cancel {job_id}")
        return {"state": "cancel_published", "job_id": job_id}
    event = {
        "schema_version": 1,
        "event": "cancel_requested",
        "campaign_id": spec["distributed"]["campaign_id"],
        "job_id": job_id,
        "worker_id": worker_id,
        "requested_at": _utc_now(),
        "reason": reason,
    }
    _write_immutable(repo, path, event)
    _commit_and_push(repo, branch, [path], f"mailbox: cancel {job_id}")
    return {"state": "cancel_published", "job_id": job_id}


def archive_request(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    request_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    request = _validate_archive_request(
        spec,
        worker_id,
        _load_json(request_path),
    )
    coordinator_ref = f"refs/remotes/origin/{spec['distributed']['coordinator_branch']}"
    if not _branch_exists(repo, coordinator_ref):
        raise MailboxError("coordinator campaign must be published first")
    _checkout_worker(repo, worker, spec["distributed"]["coordinator_branch"])
    path = _archive_request_path(spec, worker_id, request["request_id"])
    _write_immutable(repo, path, request)
    _commit_and_push(
        repo,
        worker["branch"],
        [path],
        f"mailbox: request policy archive {request['request_id'][:12]}",
    )
    return {
        "state": "archive_request_published",
        "worker_id": worker_id,
        "request_id": request["request_id"],
    }


def archive_status(
    repo_path: str | Path,
    session_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    lease = _archive_lease_config(spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    requests: list[dict[str, Any]] = []
    for worker_id in lease["authorized_worker_ids"]:
        worker = _worker(spec, worker_id)
        ref_name = f"refs/remotes/origin/{worker['branch']}"
        if not _branch_exists(repo, ref_name):
            continue
        ref = f"origin/{worker['branch']}"
        prefix = (
            f"{_campaign_root(spec)}/policy-archive/requests/{worker_id}"
        )
        for path in _list_ref(repo, ref, prefix):
            request = _validate_archive_request(
                spec,
                worker_id,
                _read_ref_json(repo, ref, path),
            )
            requests.append(
                {
                    "worker_id": worker_id,
                    "request_id": request["request_id"],
                    "candidate_id": request["candidate_id"],
                    "storage_base_commit": request["storage_base_commit"],
                }
            )
    state = _archive_lease_state(repo, spec)
    active = state["active_grant"]
    active_summary = None
    if isinstance(active, dict):
        active_summary = {
            "lease_id": active["lease_id"],
            "worker_id": active["worker_id"],
            "candidate_id": active["request"]["candidate_id"],
            "granted_at": active["granted_at"],
        }
    return {
        "campaign_id": spec["distributed"]["campaign_id"],
        "takeover_policy": lease["takeover_policy"],
        "active_lease": active_summary,
        "requests": sorted(
            requests,
            key=lambda item: (item["worker_id"], item["request_id"]),
        ),
    }


def _remote_archive_request(
    repo: Path,
    spec: dict[str, Any],
    worker_id: str,
    request_id: str,
) -> dict[str, Any]:
    worker = _worker(spec, worker_id)
    ref = f"origin/{worker['branch']}"
    request = _read_ref_json(
        repo,
        ref,
        _archive_request_path(spec, worker_id, request_id),
    )
    return _validate_archive_request(spec, worker_id, request)


def archive_grant(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    request_id: str,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    _archive_lease_config(spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    request = _remote_archive_request(
        repo,
        spec,
        worker_id,
        request_id,
    )
    state = _archive_lease_state(repo, spec)
    active = state["active_grant"]
    if isinstance(active, dict):
        if (
            active.get("lease_id") == request_id
            and active.get("worker_id") == worker_id
        ):
            return {"state": "archive_lease_granted", "grant": active}
        raise MailboxError(
            "another policy archive lease is active; release or explicitly "
            "revoke it before granting another"
        )
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    grant = {
        "schema_version": 1,
        "event": "policy_archive_granted",
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "lease_id": request_id,
        "worker_id": worker_id,
        "coordinator_id": spec["distributed"]["coordinator_id"],
        "request_sha256": _sha256(request),
        "request": request,
        "granted_at": _utc_now(),
        "takeover_policy": "explicit_revoke_only",
    }
    path = _archive_grant_path(spec, request_id)
    _write_immutable(repo, path, grant)
    _commit_and_push(
        repo,
        branch,
        [path],
        f"mailbox: grant policy archive {request_id[:12]}",
    )
    return {"state": "archive_lease_granted", "grant": grant}


def archive_prepare(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    lease_id: str,
    output_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    _archive_lease_config(spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    state = _archive_lease_state(repo, spec)
    grant = state["active_grant"]
    if (
        not isinstance(grant, dict)
        or grant.get("lease_id") != lease_id
        or grant.get("worker_id") != worker_id
    ):
        raise MailboxError("requested active policy archive lease is unavailable")
    output = Path(output_path)
    if not output.is_absolute():
        raise MailboxError("archive grant output path must be absolute")
    if output.exists():
        raise MailboxError("archive grant output already exists")
    if not output.parent.is_dir():
        raise MailboxError("archive grant output parent does not exist")
    output.write_bytes(_canonical_bytes(grant) + b"\n")
    return {
        "state": "archive_lease_materialized",
        "lease_id": lease_id,
        "output": str(output),
    }


def _policy_storage_completion(
    spec: dict[str, Any],
    worker_id: str,
    grant: dict[str, Any],
    receipt: Any,
) -> dict[str, Any]:
    lease = _archive_lease_config(spec)
    if not isinstance(receipt, dict):
        raise MailboxError("archive receipt must be a JSON object")
    binding = receipt.get("distributed_archive_lease")
    if (
        receipt.get("status")
        != "archived_simulation_qualified_hardware_candidate"
        or not isinstance(binding, dict)
        or binding.get("lease_id") != grant["lease_id"]
        or binding.get("worker_id") != worker_id
        or binding.get("request_sha256") != grant["request_sha256"]
        or receipt.get("candidate_id") != grant["request"]["candidate_id"]
        or receipt.get("storage_base_commit")
        != grant["request"]["storage_base_commit"]
    ):
        raise MailboxError("archive receipt does not match the active lease")
    root = Path(lease["worker_storage_roots"][worker_id]).resolve()
    top = Path(
        _git(root, "rev-parse", "--show-toplevel").stdout.decode().strip()
    ).resolve()
    if top != root:
        raise MailboxError("policy storage path must be its Git worktree root")
    origin = _git(root, "remote", "get-url", "origin").stdout.decode().strip()
    if (
        origin != lease["storage_remote_url"]
        and os.environ.get("POLICY_ARCHIVE_ALLOW_TEST_REMOTE") != "1"
    ):
        raise MailboxError("policy storage origin does not match the lease")
    branch = _git(root, "branch", "--show-current").stdout.decode().strip()
    if branch != lease["storage_branch"]:
        raise MailboxError("policy storage branch does not match the lease")
    if _git(root, "status", "--porcelain=v1").stdout.decode():
        raise MailboxError(
            "policy storage must be clean after the separately approved commit"
        )
    head = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    if head == grant["request"]["storage_base_commit"]:
        raise MailboxError("policy storage archive has not been committed")
    remote_lines = _git(
        root,
        "ls-remote",
        "--heads",
        "origin",
        f"refs/heads/{lease['storage_branch']}",
    ).stdout.decode().splitlines()
    remote_commits = {
        line.split()[0]
        for line in remote_lines
        if len(line.split()) == 2
        and line.split()[1] == f"refs/heads/{lease['storage_branch']}"
    }
    if remote_commits != {head}:
        raise MailboxError(
            "policy storage archive commit is not the exact remote branch head"
        )
    archive_path = Path(receipt.get("archive_path", "")).resolve()
    try:
        relative_archive = archive_path.relative_to(root)
    except ValueError as exc:
        raise MailboxError("archive receipt path escapes policy storage") from exc
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise MailboxError("archive receipt files are missing")
    expected = {
        "jit": ("policy.pt", receipt.get("sha256", {}).get("policy.pt")),
        "onnx": ("policy.onnx", receipt.get("sha256", {}).get("policy.onnx")),
        "manifest": (
            "archive_manifest.json",
            receipt.get("manifest_sha256"),
        ),
        "description": ("策略说明.txt", None),
    }
    for kind, (name, expected_hash) in expected.items():
        path = Path(files.get(kind, "")).resolve()
        if path != archive_path / name or not path.is_file() or path.is_symlink():
            raise MailboxError(f"archive receipt {kind} file is invalid")
        relative = path.relative_to(root).as_posix()
        if _git(
            root,
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
            check=False,
        ).returncode != 0:
            raise MailboxError(f"archive receipt {kind} file is not tracked")
        if expected_hash is not None and _sha256_file(path) != expected_hash:
            raise MailboxError(f"archive receipt {kind} hash changed")
    return {
        "schema_version": 1,
        "event": "policy_archive_completed",
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "lease_id": grant["lease_id"],
        "worker_id": worker_id,
        "candidate_id": grant["request"]["candidate_id"],
        "request_sha256": grant["request_sha256"],
        "storage_commit": head,
        "storage_branch": lease["storage_branch"],
        "archive_relative_path": relative_archive.as_posix(),
        "artifact_sha256": receipt["sha256"],
        "manifest_sha256": receipt["manifest_sha256"],
        "receipt_sha256": _sha256(receipt),
        "completed_at": _utc_now(),
    }


def _validate_archive_completion(
    spec: dict[str, Any],
    grant: dict[str, Any],
    completion: Any,
) -> dict[str, Any]:
    if not isinstance(completion, dict):
        raise MailboxError("archive completion must be a JSON object")
    required = {
        "schema_version",
        "event",
        "campaign_id",
        "session_sha256",
        "lease_id",
        "worker_id",
        "candidate_id",
        "request_sha256",
        "storage_commit",
        "storage_branch",
        "archive_relative_path",
        "artifact_sha256",
        "manifest_sha256",
        "receipt_sha256",
        "completed_at",
    }
    if set(completion) != required:
        raise MailboxError(
            "archive completion fields do not match the version-1 schema"
        )
    artifact_sha256 = completion.get("artifact_sha256")
    artifact_hashes_valid = (
        isinstance(artifact_sha256, dict)
        and set(artifact_sha256) == {"policy.pt", "policy.onnx"}
        and all(
            isinstance(value, str) and SHA256_RE.fullmatch(value) is not None
            for value in artifact_sha256.values()
        )
    )
    lease = _archive_lease_config(spec)
    if (
        completion.get("schema_version") != 1
        or completion.get("event") != "policy_archive_completed"
        or completion.get("campaign_id") != spec["distributed"]["campaign_id"]
        or completion.get("session_sha256") != _sha256(spec)
        or completion.get("lease_id") != grant["lease_id"]
        or completion.get("worker_id") != grant["worker_id"]
        or completion.get("candidate_id") != grant["request"]["candidate_id"]
        or completion.get("request_sha256") != grant["request_sha256"]
        or not isinstance(completion.get("storage_commit"), str)
        or GIT_SHA_RE.fullmatch(completion["storage_commit"]) is None
        or completion.get("storage_branch") != lease["storage_branch"]
        or not isinstance(completion.get("archive_relative_path"), str)
        or not completion["archive_relative_path"]
        or PurePosixPath(completion["archive_relative_path"]).is_absolute()
        or ".." in PurePosixPath(completion["archive_relative_path"]).parts
        or not artifact_hashes_valid
        or not isinstance(completion.get("manifest_sha256"), str)
        or SHA256_RE.fullmatch(completion["manifest_sha256"]) is None
        or not isinstance(completion.get("receipt_sha256"), str)
        or SHA256_RE.fullmatch(completion["receipt_sha256"]) is None
        or not isinstance(completion.get("completed_at"), str)
        or not completion["completed_at"]
    ):
        raise MailboxError("archive completion identity does not match the grant")
    return completion


def archive_complete(
    repo_path: str | Path,
    session_path: str | Path,
    worker_id: str,
    lease_id: str,
    receipt_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    worker = _worker(spec, worker_id)
    _verify_remote(repo, spec)
    _fetch(repo)
    state = _archive_lease_state(repo, spec)
    grant = state["active_grant"]
    if (
        not isinstance(grant, dict)
        or grant.get("lease_id") != lease_id
        or grant.get("worker_id") != worker_id
    ):
        raise MailboxError("worker does not hold the active archive lease")
    completion = _policy_storage_completion(
        spec,
        worker_id,
        grant,
        _load_json(receipt_path),
    )
    _checkout_worker(repo, worker, spec["distributed"]["coordinator_branch"])
    path = _archive_completion_path(spec, worker_id, lease_id)
    _write_immutable(repo, path, completion)
    _commit_and_push(
        repo,
        worker["branch"],
        [path],
        f"mailbox: complete policy archive {lease_id[:12]}",
    )
    return {"state": "archive_completion_published", "completion": completion}


def _active_grant_for_closure(
    repo: Path,
    spec: dict[str, Any],
    lease_id: str,
) -> dict[str, Any]:
    state = _archive_lease_state(repo, spec)
    grant = state["active_grant"]
    if not isinstance(grant, dict) or grant.get("lease_id") != lease_id:
        raise MailboxError("policy archive lease is not active")
    return grant


def archive_release(
    repo_path: str | Path,
    session_path: str | Path,
    lease_id: str,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    lease = _archive_lease_config(spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    grant = _active_grant_for_closure(repo, spec, lease_id)
    worker = _worker(spec, grant["worker_id"])
    worker_ref = f"origin/{worker['branch']}"
    completion = _read_ref_json(
        repo,
        worker_ref,
        _archive_completion_path(
            spec,
            grant["worker_id"],
            lease_id,
        ),
    )
    if completion is None:
        raise MailboxError(
            "valid worker completion evidence is required before release"
        )
    completion = _validate_archive_completion(spec, grant, completion)
    if os.environ.get("POLICY_ARCHIVE_ALLOW_TEST_REMOTE") == "1":
        storage_root = Path(
            lease["worker_storage_roots"][grant["worker_id"]]
        ).resolve()
        remote_target = _git(
            storage_root,
            "remote",
            "get-url",
            "origin",
        ).stdout.decode().strip()
    else:
        remote_target = lease["storage_remote_url"]
    remote_lines = _git(
        repo,
        "ls-remote",
        "--heads",
        remote_target,
        f"refs/heads/{lease['storage_branch']}",
    ).stdout.decode().splitlines()
    remote_commits = {
        line.split()[0]
        for line in remote_lines
        if len(line.split()) == 2
        and line.split()[1] == f"refs/heads/{lease['storage_branch']}"
    }
    if remote_commits != {completion["storage_commit"]}:
        raise MailboxError(
            "policy storage remote branch no longer matches completion evidence"
        )
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    closure = {
        "schema_version": 1,
        "event": "policy_archive_released",
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "lease_id": lease_id,
        "worker_id": grant["worker_id"],
        "storage_commit": completion["storage_commit"],
        "completion_sha256": _sha256(completion),
        "closed_at": _utc_now(),
    }
    path = _archive_closure_path(spec, lease_id)
    _write_immutable(repo, path, closure)
    _commit_and_push(
        repo,
        branch,
        [path],
        f"mailbox: release policy archive {lease_id[:12]}",
    )
    return {"state": "archive_lease_released", "closure": closure}


def archive_revoke(
    repo_path: str | Path,
    session_path: str | Path,
    lease_id: str,
    reason: str,
) -> dict[str, Any]:
    if not reason.strip() or len(reason) > 1000:
        raise MailboxError("revoke reason must contain between 1 and 1000 characters")
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    _archive_lease_config(spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    grant = _active_grant_for_closure(repo, spec, lease_id)
    worker = _worker(spec, grant["worker_id"])
    worker_ref = f"origin/{worker['branch']}"
    completion = _read_ref_json(
        repo,
        worker_ref,
        _archive_completion_path(spec, grant["worker_id"], lease_id),
    )
    if completion is not None:
        raise MailboxError(
            "completed archive lease must be released, not revoked"
        )
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    closure = {
        "schema_version": 1,
        "event": "policy_archive_revoked",
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "lease_id": lease_id,
        "worker_id": grant["worker_id"],
        "reason": reason,
        "closed_at": _utc_now(),
        "automatic_takeover": False,
    }
    path = _archive_closure_path(spec, lease_id)
    _write_immutable(repo, path, closure)
    _commit_and_push(
        repo,
        branch,
        [path],
        f"mailbox: revoke policy archive {lease_id[:12]}",
    )
    return {"state": "archive_lease_revoked", "closure": closure}


def publish_confirmation(
    repo_path: str | Path,
    session_path: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    """Publish an immutable top-k selection and any remaining-seed jobs."""
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    if spec.get("version") != 7:
        raise MailboxError("confirmation publication requires a version-7 session")
    plan = _load_plan(plan_path, spec)
    _verify_remote(repo, spec)
    _fetch(repo)
    report = _collect_report(repo, spec)
    if report["invalid_results"]:
        raise MailboxError("invalid worker results block confirmation publication")

    expected = {
        (run["trial_id"], run["seed"])
        for run in plan["runs"]
        if run["stage"] == "screening"
    }
    observed: set[tuple[str, int]] = set()
    screening_results: list[dict[str, Any]] = []
    evidence_hashes: list[str] = []
    for envelope in report["accepted_results"]:
        job = _coordinator_job(
            repo,
            spec,
            envelope["worker_id"],
            envelope["job_id"],
        )
        run = job["run"]
        if job["kind"] != "trial" or run["stage"] != "screening":
            continue
        payload = envelope["result"]
        if (
            not isinstance(payload, dict)
            or payload.get("trial_id") != run["trial_id"]
            or payload.get("seed") != run["seed"]
            or payload.get("status") != "completed"
        ):
            raise MailboxError("screening result identity or completion state is invalid")
        observed.add((run["trial_id"], run["seed"]))
        screening_results.append(payload)
        evidence_hashes.append(envelope["result_sha256"])
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise MailboxError(
            f"screening result coverage mismatch; missing={missing}, extra={extra}"
        )
    try:
        selected = select_confirmation_candidates(spec, screening_results)
        confirmation_runs = build_confirmation_runs(spec, plan, selected)
    except SpecError as exc:
        raise MailboxError(str(exc)) from exc
    selection = {
        "schema_version": 1,
        "campaign_id": spec["distributed"]["campaign_id"],
        "session_sha256": _sha256(spec),
        "plan_sha256": _sha256(plan),
        "selected_trial_ids": selected,
        "screening_result_sha256s": sorted(evidence_hashes),
    }
    jobs = _build_jobs(
        spec,
        plan,
        runs=confirmation_runs,
        include_calibration=False,
        selection=selection,
    )
    branch = spec["distributed"]["coordinator_branch"]
    _checkout_coordinator(repo, branch)
    selection_path = f"{_campaign_root(spec)}/selections/confirmation.json"
    changed: list[str] = []
    if _write_immutable(repo, selection_path, selection):
        changed.append(selection_path)
    for job in jobs:
        path = _job_path(spec, job["worker_id"], job["job_id"])
        if _write_immutable(repo, path, job):
            changed.append(path)
    _commit_and_push(
        repo,
        branch,
        changed,
        f"mailbox: publish confirmation {spec['distributed']['campaign_id']}",
    )
    return {
        "state": (
            "single_seed_selection_published"
            if spec["tuning"]["seed_strategy"].get(
                "mode", "robust_multi_seed"
            )
            == "fixed_single_seed"
            else "confirmation_published"
        ),
        "selected_trial_ids": selected,
        "published_jobs": len(jobs),
        "selection_sha256": _sha256(selection),
    }


def _collect_report(repo: Path, spec: dict[str, Any]) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    invalid: list[dict[str, str]] = []
    accepted_job_ids: set[str] = set()
    for worker in spec["distributed"]["workers"]:
        worker_ref = f"origin/{worker['branch']}"
        if not _branch_exists(repo, f"refs/remotes/{worker_ref}"):
            continue
        prefix = f"{_campaign_root(spec)}/results/{worker['id']}"
        for path in _list_ref(repo, worker_ref, prefix):
            try:
                envelope = _read_ref_json(repo, worker_ref, path)
                if not isinstance(envelope, dict):
                    raise MailboxError("result envelope is not an object")
                job_id = envelope.get("job_id")
                if not isinstance(job_id, str):
                    raise MailboxError("result job_id is invalid")
                job = _coordinator_job(repo, spec, worker["id"], job_id)
                attempt = envelope.get("attempt")
                if isinstance(attempt, bool) or not isinstance(attempt, int):
                    raise MailboxError("result attempt is invalid")
                receipt = _read_ref_json(
                    repo,
                    worker_ref,
                    _event_path(
                        spec,
                        "receipts",
                        worker["id"],
                        job_id,
                        attempt,
                    ),
                )
                source = envelope.get("source")
                if not isinstance(source, dict):
                    raise MailboxError("result source binding is invalid")
                if (
                    envelope.get("worker_id") != worker["id"]
                    or envelope.get("job_sha256") != _sha256(job)
                    or not isinstance(receipt, dict)
                    or receipt.get("job_sha256") != _sha256(job)
                    or receipt.get("worker_id") != worker["id"]
                    or source.get("source_git_commit")
                    != job["source_git_commit"]
                    or source.get("source_git_dirty") is not False
                    or envelope.get("result_sha256") != _sha256(envelope.get("result"))
                    or envelope.get("artifact_manifest_sha256")
                    != _sha256(envelope.get("artifact_manifest"))
                ):
                    raise MailboxError("result hash or identity binding failed")
                if job_id in accepted_job_ids:
                    raise MailboxError("multiple completed attempts exist for one job")
                _validate_artifacts(envelope["artifact_manifest"])
                accepted.append(envelope)
                accepted_job_ids.add(job_id)
            except (MailboxError, KeyError) as exc:
                invalid.append({"path": path, "error": str(exc)})
    report = {
        "schema_version": 1,
        "campaign_id": spec["distributed"]["campaign_id"],
        "collected_at": _utc_now(),
        "accepted_results": accepted,
        "invalid_results": invalid,
    }
    return report


def collect(
    repo_path: str | Path,
    session_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    repo = _repo(repo_path)
    spec = load_and_validate(session_path)
    _verify_remote(repo, spec)
    _fetch(repo)
    report = _collect_report(repo, spec)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(report) + b"\n")
    return {
        "accepted_result_count": len(report["accepted_results"]),
        "invalid_result_count": len(report["invalid_results"]),
        "output": str(output),
    }


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--repo", required=True)
        command.add_argument("--session", required=True)

    publish_parser = subparsers.add_parser("publish")
    common(publish_parser)
    publish_parser.add_argument("--plan", required=True)

    history_publish_parser = subparsers.add_parser("history-publish")
    common(history_publish_parser)
    history_publish_parser.add_argument("--worker", required=True)
    history_publish_parser.add_argument("--index-json", required=True)

    history_initialize_parser = subparsers.add_parser("history-initialize")
    common(history_initialize_parser)

    history_collect_parser = subparsers.add_parser("history-collect")
    common(history_collect_parser)
    history_collect_parser.add_argument("--output", required=True)

    adaptive_parser = subparsers.add_parser("publish-adaptive-round")
    common(adaptive_parser)
    adaptive_parser.add_argument("--previous-plan", required=True)
    adaptive_parser.add_argument("--expanded-plan", required=True)

    multifidelity_parser = subparsers.add_parser(
        "publish-multifidelity-rung"
    )
    common(multifidelity_parser)
    multifidelity_parser.add_argument("--previous-plan", required=True)
    multifidelity_parser.add_argument("--expanded-plan", required=True)

    confirmation_parser = subparsers.add_parser("publish-confirmation")
    common(confirmation_parser)
    confirmation_parser.add_argument("--plan", required=True)

    status_parser = subparsers.add_parser("status")
    common(status_parser)
    status_parser.add_argument("--worker", required=True)
    status_parser.add_argument("--skip-source-check", action="store_true")

    claim_parser = subparsers.add_parser("claim")
    common(claim_parser)
    claim_parser.add_argument("--worker", required=True)
    claim_parser.add_argument("--job-id", required=True)
    claim_parser.add_argument("--attempt", type=int, default=1)

    prepare_parser = subparsers.add_parser("prepare-job")
    common(prepare_parser)
    prepare_parser.add_argument("--worker", required=True)
    prepare_parser.add_argument("--job-id", required=True)
    prepare_parser.add_argument("--attempt", type=int, default=1)
    prepare_parser.add_argument("--output", required=True)

    progress_parser = subparsers.add_parser("progress")
    common(progress_parser)
    progress_parser.add_argument("--worker", required=True)
    progress_parser.add_argument("--job-id", required=True)
    progress_parser.add_argument("--attempt", type=int, default=1)
    progress_parser.add_argument("--sequence", type=int, required=True)
    progress_parser.add_argument("--progress-json", required=True)

    result_parser = subparsers.add_parser("result")
    common(result_parser)
    result_parser.add_argument("--worker", required=True)
    result_parser.add_argument("--job-id", required=True)
    result_parser.add_argument("--attempt", type=int, default=1)
    result_parser.add_argument("--result-json", required=True)
    result_parser.add_argument("--artifact-manifest", required=True)

    cancel_parser = subparsers.add_parser("cancel")
    common(cancel_parser)
    cancel_parser.add_argument("--worker", required=True)
    cancel_parser.add_argument("--job-id", required=True)
    cancel_parser.add_argument("--reason", required=True)

    archive_request_parser = subparsers.add_parser("archive-request")
    common(archive_request_parser)
    archive_request_parser.add_argument("--worker", required=True)
    archive_request_parser.add_argument("--request-json", required=True)

    archive_status_parser = subparsers.add_parser("archive-status")
    common(archive_status_parser)

    archive_grant_parser = subparsers.add_parser("archive-grant")
    common(archive_grant_parser)
    archive_grant_parser.add_argument("--worker", required=True)
    archive_grant_parser.add_argument("--request-id", required=True)

    archive_prepare_parser = subparsers.add_parser("archive-prepare")
    common(archive_prepare_parser)
    archive_prepare_parser.add_argument("--worker", required=True)
    archive_prepare_parser.add_argument("--lease-id", required=True)
    archive_prepare_parser.add_argument("--output", required=True)

    archive_complete_parser = subparsers.add_parser("archive-complete")
    common(archive_complete_parser)
    archive_complete_parser.add_argument("--worker", required=True)
    archive_complete_parser.add_argument("--lease-id", required=True)
    archive_complete_parser.add_argument("--archive-receipt", required=True)

    archive_release_parser = subparsers.add_parser("archive-release")
    common(archive_release_parser)
    archive_release_parser.add_argument("--lease-id", required=True)

    archive_revoke_parser = subparsers.add_parser("archive-revoke")
    common(archive_revoke_parser)
    archive_revoke_parser.add_argument("--lease-id", required=True)
    archive_revoke_parser.add_argument("--reason", required=True)

    collect_parser = subparsers.add_parser("collect")
    common(collect_parser)
    collect_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    try:
        if args.command == "publish":
            value = publish(args.repo, args.session, args.plan)
        elif args.command == "history-initialize":
            value = history_initialize(args.repo, args.session)
        elif args.command == "history-publish":
            value = history_publish(
                args.repo,
                args.session,
                args.worker,
                args.index_json,
            )
        elif args.command == "history-collect":
            value = history_collect(
                args.repo,
                args.session,
                args.output,
            )
        elif args.command == "publish-adaptive-round":
            value = publish_adaptive_round(
                args.repo,
                args.session,
                args.previous_plan,
                args.expanded_plan,
            )
        elif args.command == "publish-multifidelity-rung":
            value = publish_multifidelity_rung(
                args.repo,
                args.session,
                args.previous_plan,
                args.expanded_plan,
            )
        elif args.command == "publish-confirmation":
            value = publish_confirmation(args.repo, args.session, args.plan)
        elif args.command == "status":
            value = status(
                args.repo,
                args.session,
                args.worker,
                inspect_source=not args.skip_source_check,
            )
        elif args.command == "claim":
            value = claim(
                args.repo, args.session, args.worker, args.job_id, args.attempt
            )
        elif args.command == "prepare-job":
            value = prepare_job(
                args.repo,
                args.session,
                args.worker,
                args.job_id,
                args.attempt,
                args.output,
            )
        elif args.command == "progress":
            value = progress(
                args.repo,
                args.session,
                args.worker,
                args.job_id,
                args.attempt,
                args.sequence,
                args.progress_json,
            )
        elif args.command == "result":
            value = result(
                args.repo,
                args.session,
                args.worker,
                args.job_id,
                args.attempt,
                args.result_json,
                args.artifact_manifest,
            )
        elif args.command == "cancel":
            value = cancel(
                args.repo,
                args.session,
                args.worker,
                args.job_id,
                args.reason,
            )
        elif args.command == "archive-request":
            value = archive_request(
                args.repo,
                args.session,
                args.worker,
                args.request_json,
            )
        elif args.command == "archive-status":
            value = archive_status(args.repo, args.session)
        elif args.command == "archive-grant":
            value = archive_grant(
                args.repo,
                args.session,
                args.worker,
                args.request_id,
            )
        elif args.command == "archive-prepare":
            value = archive_prepare(
                args.repo,
                args.session,
                args.worker,
                args.lease_id,
                args.output,
            )
        elif args.command == "archive-complete":
            value = archive_complete(
                args.repo,
                args.session,
                args.worker,
                args.lease_id,
                args.archive_receipt,
            )
        elif args.command == "archive-release":
            value = archive_release(
                args.repo,
                args.session,
                args.lease_id,
            )
        elif args.command == "archive-revoke":
            value = archive_revoke(
                args.repo,
                args.session,
                args.lease_id,
                args.reason,
            )
        else:
            value = collect(args.repo, args.session, args.output)
    except (MailboxError, SpecError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    _print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

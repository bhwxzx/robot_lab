#!/usr/bin/env python3
"""Plan, apply, and verify first-run configuration for this training skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
GIT_BRANCH_RE = re.compile(
    r"^(?!/)(?!.*(?:\.\.|//|@\{|\\|\s))(?!.*[/.]$)[A-Za-z0-9._/-]+$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CONFIG_ENV_VAR = "ROBOT_LAB_TUNER_CONFIG"


class SetupError(ValueError):
    """Raised when a setup input, plan, or local prerequisite is unsafe."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SetupError(f"value is not finite canonical JSON: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def locate_configuration() -> dict[str, Any]:
    """Resolve the stable first-run configuration path without writing it."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override is not None:
        configuration_path = Path(override)
        source = "environment"
        if not configuration_path.is_absolute():
            raise SetupError(f"{CONFIG_ENV_VAR} must be an absolute path")
        if configuration_path.name != "configuration.json":
            raise SetupError(
                f"{CONFIG_ENV_VAR} must end with configuration.json"
            )
        configuration_path = configuration_path.resolve(strict=False)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        if xdg is not None:
            base = Path(xdg)
            if not base.is_absolute():
                raise SetupError("XDG_CONFIG_HOME must be an absolute path")
            base = base.resolve(strict=False)
        else:
            base = (Path.home() / ".config").resolve(strict=False)
        configuration_path = (
            base
            / "robot-lab"
            / "monitor-tune-isaaclab-training"
            / "configuration.json"
        )
        source = "default"
    receipt_path = configuration_path.parent / "setup_receipt.json"
    return {
        "configuration_path": str(configuration_path),
        "receipt_path": str(receipt_path),
        "source": source,
        "configuration_exists": configuration_path.is_file(),
        "receipt_exists": receipt_path.is_file(),
    }


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SetupError(f"{label} does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise SetupError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SetupError(f"{label} must be a JSON object")
    _canonical_bytes(value)
    return value


def _check_exact_keys(value: dict[str, Any], expected: set[str], path: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise SetupError(f"{path} is missing field(s): {', '.join(missing)}")
    if unknown:
        raise SetupError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SetupError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SetupError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise SetupError(f"{path} must be between {minimum} and {maximum}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SetupError(f"{path} must be a boolean")
    return value


def _absolute_path(value: Any, path: str) -> Path:
    result = Path(_string(value, path))
    if not result.is_absolute():
        raise SetupError(f"{path} must be an absolute path")
    return result.resolve(strict=False)


def _optional_absolute_path(value: Any, path: str) -> Path | None:
    if value is None:
        return None
    return _absolute_path(value, path)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_https_url(value: Any, path: str) -> str:
    url = _string(value, path)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SetupError(
            f"{path} must be HTTPS without credentials, query, or fragment"
        )
    return url


def _safe_branch(value: Any, path: str) -> str:
    branch = _string(value, path)
    if (
        not GIT_BRANCH_RE.fullmatch(branch)
        or any(
            part.startswith(".") or part.endswith(".lock")
            for part in branch.split("/")
        )
    ):
        raise SetupError(f"{path} is not a safe Git branch")
    return branch


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        return subprocess.run(
            argv,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SetupError(f"command could not complete: {argv[0]}: {exc}") from exc


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    completed = _run(["git", *args], cwd=cwd)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SetupError(f"git {' '.join(args)} failed: {message}")
    return completed


def _git_root(path: Path, label: str) -> Path:
    if not path.is_dir() or path.is_symlink():
        raise SetupError(f"{label} must be an existing regular directory: {path}")
    root = Path(
        _git(path, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve(strict=True)
    if root != path.resolve(strict=True):
        raise SetupError(f"{label} must be the Git worktree root: {path}")
    return root


def _normalize_machine(
    value: Any,
    index: int,
    setup_mode: str,
) -> dict[str, Any]:
    path = f"machines[{index}]"
    if not isinstance(value, dict):
        raise SetupError(f"{path} must be an object")
    _check_exact_keys(
        value,
        {
            "id",
            "source_repo",
            "mailbox_repo",
            "state_dir",
            "effective_config_baseline_path",
            "evaluation_output_dir",
            "hardware_feedback_output_dir",
            "policy_storage_root",
            "gpu_index",
            "worker_branch",
        },
        path,
    )
    machine_id = _string(value["id"], f"{path}.id")
    if not IDENTIFIER_RE.fullmatch(machine_id):
        raise SetupError(f"{path}.id contains unsupported characters")
    source = _absolute_path(value["source_repo"], f"{path}.source_repo")
    state = _absolute_path(value["state_dir"], f"{path}.state_dir")
    baseline = _absolute_path(
        value["effective_config_baseline_path"],
        f"{path}.effective_config_baseline_path",
    )
    if baseline.suffix.lower() != ".json":
        raise SetupError(f"{path}.effective_config_baseline_path must end in .json")
    evaluation = _absolute_path(
        value["evaluation_output_dir"],
        f"{path}.evaluation_output_dir",
    )
    feedback = _absolute_path(
        value["hardware_feedback_output_dir"],
        f"{path}.hardware_feedback_output_dir",
    )
    policy_storage = _optional_absolute_path(
        value["policy_storage_root"],
        f"{path}.policy_storage_root",
    )
    mailbox = _optional_absolute_path(
        value["mailbox_repo"],
        f"{path}.mailbox_repo",
    )
    worker_branch = value["worker_branch"]
    if setup_mode == "git_mailbox":
        if mailbox is None:
            raise SetupError(f"{path}.mailbox_repo is required in git_mailbox mode")
        worker_branch = _safe_branch(worker_branch, f"{path}.worker_branch")
    elif mailbox is not None or worker_branch is not None:
        raise SetupError(
            f"{path}.mailbox_repo and worker_branch must be null in single_host mode"
        )
    for output_name, output_path in (
        ("state_dir", state),
        ("effective_config_baseline_path", baseline),
        ("evaluation_output_dir", evaluation),
        ("hardware_feedback_output_dir", feedback),
        ("mailbox_repo", mailbox),
        ("policy_storage_root", policy_storage),
    ):
        if output_path is not None and _is_within(output_path, source):
            raise SetupError(
                f"{path}.{output_name} must be outside the source worktree"
            )
    return {
        "id": machine_id,
        "source_repo": str(source),
        "mailbox_repo": None if mailbox is None else str(mailbox),
        "state_dir": str(state),
        "effective_config_baseline_path": str(baseline),
        "evaluation_output_dir": str(evaluation),
        "hardware_feedback_output_dir": str(feedback),
        "policy_storage_root": (
            None if policy_storage is None else str(policy_storage)
        ),
        "gpu_index": _integer(value["gpu_index"], f"{path}.gpu_index", 0, 1024),
        "worker_branch": worker_branch,
    }


def normalize_answers(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one user-approved first-run answer document."""
    _check_exact_keys(
        value,
        {
            "version",
            "setup_id",
            "setup_mode",
            "configuration_dir",
            "conda_env",
            "default_seed",
            "source_remote_url",
            "local_machine_id",
            "machines",
            "distributed",
        },
        "answers",
    )
    if value["version"] != 1:
        raise SetupError("answers.version must be 1")
    setup_id = _string(value["setup_id"], "answers.setup_id")
    if not IDENTIFIER_RE.fullmatch(setup_id):
        raise SetupError("answers.setup_id contains unsupported characters")
    setup_mode = value["setup_mode"]
    if setup_mode not in {"single_host", "git_mailbox"}:
        raise SetupError("answers.setup_mode must be single_host or git_mailbox")
    configuration_dir = _absolute_path(
        value["configuration_dir"],
        "answers.configuration_dir",
    )
    conda_env = _string(value["conda_env"], "answers.conda_env")
    if conda_env != "isaacsim-5.1":
        raise SetupError("answers.conda_env must be isaacsim-5.1")
    default_seed = _integer(
        value["default_seed"],
        "answers.default_seed",
        0,
        2**31 - 1,
    )
    source_remote_url = _safe_https_url(
        value["source_remote_url"],
        "answers.source_remote_url",
    )
    machines_value = value["machines"]
    if not isinstance(machines_value, list):
        raise SetupError("answers.machines must be an array")
    minimum = 2 if setup_mode == "git_mailbox" else 1
    maximum = 16 if setup_mode == "git_mailbox" else 1
    if not minimum <= len(machines_value) <= maximum:
        raise SetupError(
            f"answers.machines must contain between {minimum} and {maximum} machine(s)"
        )
    machines = [
        _normalize_machine(item, index, setup_mode)
        for index, item in enumerate(machines_value)
    ]
    machine_ids = [machine["id"] for machine in machines]
    if len(machine_ids) != len(set(machine_ids)):
        raise SetupError("answers.machines contains duplicate IDs")
    local_machine_id = _string(
        value["local_machine_id"],
        "answers.local_machine_id",
    )
    if local_machine_id not in machine_ids:
        raise SetupError("answers.local_machine_id must name one configured machine")
    for index, machine in enumerate(machines):
        outputs = [
            Path(machine["state_dir"]),
            Path(machine["effective_config_baseline_path"]),
            Path(machine["evaluation_output_dir"]),
            Path(machine["hardware_feedback_output_dir"]),
        ]
        mailbox_path = (
            Path(machine["mailbox_repo"])
            if machine["mailbox_repo"] is not None
            else None
        )
        policy_path = (
            Path(machine["policy_storage_root"])
            if machine["policy_storage_root"] is not None
            else None
        )
        for output in outputs:
            if mailbox_path is not None and _is_within(output, mailbox_path):
                raise SetupError(
                    f"machines[{index}] runtime outputs must be outside mailbox_repo"
                )
            if policy_path is not None and _is_within(output, policy_path):
                raise SetupError(
                    f"machines[{index}] runtime outputs must be outside "
                    "policy_storage_root"
                )
        if (
            mailbox_path is not None
            and policy_path is not None
            and (
                _is_within(mailbox_path, policy_path)
                or _is_within(policy_path, mailbox_path)
            )
        ):
            raise SetupError(
                f"machines[{index}] mailbox_repo and policy_storage_root "
                "must not overlap"
            )
        if machine["id"] == local_machine_id:
            for protected_name, protected_path in (
                ("mailbox_repo", mailbox_path),
                ("policy_storage_root", policy_path),
            ):
                if protected_path is not None and (
                    _is_within(configuration_dir, protected_path)
                    or _is_within(protected_path, configuration_dir)
                ):
                    raise SetupError(
                        "answers.configuration_dir must not overlap local "
                        f"{protected_name}"
                    )
    for machine in machines:
        source = Path(machine["source_repo"])
        if _is_within(configuration_dir, source):
            raise SetupError(
                "answers.configuration_dir must be outside every source worktree"
            )

    distributed_value = value["distributed"]
    distributed: dict[str, Any] | None
    if setup_mode == "single_host":
        if distributed_value is not None:
            raise SetupError("answers.distributed must be null in single_host mode")
        distributed = None
    else:
        if not isinstance(distributed_value, dict):
            raise SetupError("answers.distributed must be an object")
        _check_exact_keys(
            distributed_value,
            {
                "transport",
                "remote_url",
                "coordinator_id",
                "coordinator_branch",
                "poll_interval_seconds",
                "remote_state_unknown_after_seconds",
                "artifact_policy",
                "assignment_mode_default",
                "host_effect_calibration_default_enabled",
            },
            "answers.distributed",
        )
        if distributed_value["transport"] != "git_mailbox":
            raise SetupError("answers.distributed.transport must be git_mailbox")
        if distributed_value["artifact_policy"] != "metadata_only":
            raise SetupError(
                "answers.distributed.artifact_policy must be metadata_only"
            )
        coordinator_id = _string(
            distributed_value["coordinator_id"],
            "answers.distributed.coordinator_id",
        )
        if coordinator_id not in machine_ids:
            raise SetupError(
                "answers.distributed.coordinator_id must name one machine"
            )
        coordinator_branch = _safe_branch(
            distributed_value["coordinator_branch"],
            "answers.distributed.coordinator_branch",
        )
        worker_branches = [machine["worker_branch"] for machine in machines]
        if (
            len(worker_branches) != len(set(worker_branches))
            or coordinator_branch in worker_branches
        ):
            raise SetupError(
                "coordinator and worker branches must all be unique"
            )
        poll = _integer(
            distributed_value["poll_interval_seconds"],
            "answers.distributed.poll_interval_seconds",
            60,
            3600,
        )
        unknown_after = _integer(
            distributed_value["remote_state_unknown_after_seconds"],
            "answers.distributed.remote_state_unknown_after_seconds",
            120,
            86400,
        )
        if unknown_after < 2 * poll:
            raise SetupError(
                "remote_state_unknown_after_seconds must be at least twice "
                "poll_interval_seconds"
            )
        assignment = distributed_value["assignment_mode_default"]
        if assignment not in {"by_trial", "by_seed"}:
            raise SetupError(
                "answers.distributed.assignment_mode_default must be by_trial "
                "or by_seed"
            )
        distributed = {
            "transport": "git_mailbox",
            "remote_url": _safe_https_url(
                distributed_value["remote_url"],
                "answers.distributed.remote_url",
            ),
            "coordinator_id": coordinator_id,
            "coordinator_branch": coordinator_branch,
            "poll_interval_seconds": poll,
            "remote_state_unknown_after_seconds": unknown_after,
            "artifact_policy": "metadata_only",
            "assignment_mode_default": assignment,
            "host_effect_calibration_default_enabled": _boolean(
                distributed_value["host_effect_calibration_default_enabled"],
                "answers.distributed.host_effect_calibration_default_enabled",
            ),
        }
    return {
        "version": 1,
        "setup_id": setup_id,
        "setup_mode": setup_mode,
        "configuration_dir": str(configuration_dir),
        "conda_env": conda_env,
        "default_seed": default_seed,
        "source_remote_url": source_remote_url,
        "local_machine_id": local_machine_id,
        "machines": machines,
        "distributed": distributed,
    }


def _configuration_from_answers(
    answers: dict[str, Any],
    answers_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "setup_id": answers["setup_id"],
        "setup_mode": answers["setup_mode"],
        "configuration_path": str(
            Path(answers["configuration_dir"]) / "configuration.json"
        ),
        "receipt_path": str(
            Path(answers["configuration_dir"]) / "setup_receipt.json"
        ),
        "configured_from_answers_sha256": answers_sha256,
        "conda_env": answers["conda_env"],
        "default_seed": answers["default_seed"],
        "source_remote_url": answers["source_remote_url"],
        "local_machine_id": answers["local_machine_id"],
        "machines": answers["machines"],
        "distributed": answers["distributed"],
    }


def _local_machine(configuration: dict[str, Any]) -> dict[str, Any]:
    matches = [
        machine
        for machine in configuration["machines"]
        if machine["id"] == configuration["local_machine_id"]
    ]
    if len(matches) != 1:
        raise SetupError("configuration local machine identity is invalid")
    return matches[0]


def _expected_operations(configuration: dict[str, Any], config_dir: Path) -> list[dict[str, Any]]:
    local = _local_machine(configuration)
    directories = {
        config_dir,
        Path(local["state_dir"]),
        Path(local["effective_config_baseline_path"]).parent,
        Path(local["evaluation_output_dir"]),
        Path(local["hardware_feedback_output_dir"]),
    }
    operations = [
        {"kind": "ensure_directory", "path": str(path)}
        for path in sorted(directories, key=str)
    ]
    if local["policy_storage_root"] is not None:
        operations.append(
            {
                "kind": "verify_git_worktree",
                "path": local["policy_storage_root"],
                "label": "policy_storage_root",
            }
        )
    if configuration["setup_mode"] == "git_mailbox":
        operations.append(
            {
                "kind": "ensure_git_clone",
                "path": local["mailbox_repo"],
                "remote_url": configuration["distributed"]["remote_url"],
            }
        )
    operations.extend(
        [
            {
                "kind": "write_new_json",
                "path": str(config_dir / "setup_receipt.json"),
                "document": "setup_receipt",
            },
            {
                "kind": "write_new_json",
                "path": str(config_dir / "configuration.json"),
                "document": "configuration",
            },
        ]
    )
    return operations


def build_plan(answers_value: dict[str, Any]) -> dict[str, Any]:
    """Build a hash-bound, non-executing setup plan."""
    answers = normalize_answers(answers_value)
    local = next(
        machine
        for machine in answers["machines"]
        if machine["id"] == answers["local_machine_id"]
    )
    _git_root(Path(local["source_repo"]), "local source_repo")
    source_origin = _git(
        Path(local["source_repo"]),
        "remote",
        "get-url",
        "origin",
    ).stdout.strip()
    if source_origin != answers["source_remote_url"]:
        raise SetupError(
            "local source_repo origin does not match answers.source_remote_url"
        )
    answers_sha256 = _sha256(answers)
    configuration = _configuration_from_answers(answers, answers_sha256)
    config_dir = Path(answers["configuration_dir"])
    unsigned = {
        "schema_version": 1,
        "plan_type": "monitor_tune_first_run_setup",
        "generated_at": _utc_now(),
        "setup_id": answers["setup_id"],
        "local_machine_id": answers["local_machine_id"],
        "configuration_path": str(config_dir / "configuration.json"),
        "receipt_path": str(config_dir / "setup_receipt.json"),
        "answers_sha256": answers_sha256,
        "configuration": configuration,
        "operations": _expected_operations(configuration, config_dir),
        "discovery": {
            "environment_variable": CONFIG_ENV_VAR,
            "required_value": str(config_dir / "configuration.json"),
            "matches_current_discovery": (
                locate_configuration()["configuration_path"]
                == str(config_dir / "configuration.json")
            ),
        },
        "remote_push_permitted": False,
        "credential_storage_permitted": False,
        "overwrite_permitted": False,
    }
    return {**unsigned, "plan_sha256": _sha256(unsigned)}


def _validate_plan(plan: dict[str, Any]) -> dict[str, Any]:
    _check_exact_keys(
        plan,
        {
            "schema_version",
            "plan_type",
            "generated_at",
            "setup_id",
            "local_machine_id",
            "configuration_path",
            "receipt_path",
            "answers_sha256",
            "configuration",
            "operations",
            "discovery",
            "remote_push_permitted",
            "credential_storage_permitted",
            "overwrite_permitted",
            "plan_sha256",
        },
        "setup_plan",
    )
    if plan["schema_version"] != 1:
        raise SetupError("setup plan schema_version must be 1")
    if plan["plan_type"] != "monitor_tune_first_run_setup":
        raise SetupError("setup plan type is unsupported")
    digest = plan["plan_sha256"]
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise SetupError("setup plan SHA-256 is invalid")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if _sha256(unsigned) != digest:
        raise SetupError("setup plan SHA-256 does not match its contents")
    for field in (
        "remote_push_permitted",
        "credential_storage_permitted",
        "overwrite_permitted",
    ):
        if plan[field] is not False:
            raise SetupError(f"setup plan {field} must be false")
    configuration = plan["configuration"]
    if not isinstance(configuration, dict):
        raise SetupError("setup plan configuration must be an object")
    if (
        configuration.get("setup_id") != plan["setup_id"]
        or configuration.get("local_machine_id") != plan["local_machine_id"]
    ):
        raise SetupError("setup plan identity does not match its configuration")
    configured_from = configuration.get("configured_from_answers_sha256")
    if (
        not isinstance(configured_from, str)
        or not SHA256_RE.fullmatch(configured_from)
        or configured_from != plan["answers_sha256"]
    ):
        raise SetupError("setup plan answers SHA-256 binding is invalid")
    answers_like = {
        "version": 1,
        "setup_id": configuration.get("setup_id"),
        "setup_mode": configuration.get("setup_mode"),
        "configuration_dir": str(Path(plan["configuration_path"]).parent),
        "conda_env": configuration.get("conda_env"),
        "default_seed": configuration.get("default_seed"),
        "source_remote_url": configuration.get("source_remote_url"),
        "local_machine_id": configuration.get("local_machine_id"),
        "machines": configuration.get("machines"),
        "distributed": configuration.get("distributed"),
    }
    normalized = normalize_answers(answers_like)
    if _sha256(normalized) != configured_from:
        raise SetupError("setup plan normalized answers hash is invalid")
    expected_configuration = _configuration_from_answers(
        normalized,
        configured_from,
    )
    if expected_configuration != configuration:
        raise SetupError("setup plan configuration is not normalized")
    config_dir = Path(normalized["configuration_dir"])
    if Path(plan["configuration_path"]) != config_dir / "configuration.json":
        raise SetupError("setup plan configuration path is invalid")
    if Path(plan["receipt_path"]) != config_dir / "setup_receipt.json":
        raise SetupError("setup plan receipt path is invalid")
    if plan["operations"] != _expected_operations(configuration, config_dir):
        raise SetupError("setup plan operations do not match its configuration")
    discovery = plan["discovery"]
    if (
        not isinstance(discovery, dict)
        or set(discovery)
        != {
            "environment_variable",
            "required_value",
            "matches_current_discovery",
        }
        or discovery["environment_variable"] != CONFIG_ENV_VAR
        or discovery["required_value"] != str(config_dir / "configuration.json")
        or not isinstance(discovery["matches_current_discovery"], bool)
    ):
        raise SetupError("setup plan discovery binding is invalid")
    return plan


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_object(path, f"existing {path.name}")
        if existing != value:
            raise SetupError(f"refusing to overwrite changed file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise SetupError(f"refusing to overwrite existing file: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_clone(path: Path, remote_url: str) -> bool:
    if path.exists():
        _git_root(path, "mailbox_repo")
        origin = _git(path, "remote", "get-url", "origin").stdout.strip()
        if origin != remote_url:
            raise SetupError(
                "existing mailbox_repo origin does not match approved remote_url"
            )
        return False
    remote = _run(["git", "ls-remote", "--heads", remote_url], timeout=30)
    if remote.returncode != 0:
        raise SetupError(
            "coordination remote is unreachable without an interactive prompt"
        )
    if not remote.stdout.strip():
        raise SetupError(
            "coordination remote has no branch; create one initial commit first"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(["git", "clone", "--origin", "origin", "--", remote_url, str(path)])
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise SetupError(f"git clone failed: {message}")
    _git_root(path, "mailbox_repo")
    return True


def apply_plan(plan_value: dict[str, Any], approval_sha256: str) -> dict[str, Any]:
    """Apply only the exact user-approved setup plan without remote writes."""
    plan = _validate_plan(plan_value)
    if approval_sha256 != plan["plan_sha256"]:
        raise SetupError("approval SHA-256 does not match the setup plan")
    config_path = Path(plan["configuration_path"])
    receipt_path = Path(plan["receipt_path"])
    configuration = plan["configuration"]
    existing_configuration = (
        _load_object(config_path, "existing configuration")
        if config_path.exists()
        else None
    )
    existing_receipt = (
        _load_object(receipt_path, "existing setup receipt")
        if receipt_path.exists()
        else None
    )
    if existing_configuration is not None and existing_configuration != configuration:
        raise SetupError("existing configuration differs; create and approve a new plan")
    if (
        existing_receipt is not None
        and existing_receipt.get("plan_sha256") != plan["plan_sha256"]
    ):
        raise SetupError("existing setup receipt belongs to another plan")
    if existing_configuration is not None and existing_receipt is not None:
        if existing_receipt.get("configuration_sha256") != _sha256(configuration):
            raise SetupError("existing setup receipt configuration hash is invalid")
        discoverable = (
            locate_configuration()["configuration_path"] == str(config_path)
        )
        return {
            "state": "already_configured",
            "plan_sha256": plan["plan_sha256"],
            "configuration_path": str(config_path),
            "receipt_path": str(receipt_path),
            "configuration_discoverable": discoverable,
            "activation_required": (
                None
                if discoverable
                else {
                    "environment_variable": CONFIG_ENV_VAR,
                    "required_value": str(config_path),
                }
            ),
            "remote_push_performed": False,
        }

    clone_performed = False
    local = _local_machine(configuration)
    source = Path(local["source_repo"])
    _git_root(source, "local source_repo")
    source_origin = _git(source, "remote", "get-url", "origin").stdout.strip()
    if source_origin != configuration["source_remote_url"]:
        raise SetupError(
            "local source_repo origin no longer matches the approved plan"
        )
    for operation in plan["operations"]:
        kind = operation["kind"]
        if kind == "ensure_directory":
            Path(operation["path"]).mkdir(parents=True, exist_ok=True)
        elif kind == "verify_git_worktree":
            _git_root(Path(operation["path"]), operation["label"])
        elif kind == "ensure_git_clone":
            clone_performed = _ensure_clone(
                Path(operation["path"]),
                operation["remote_url"],
            )
        elif kind == "write_new_json":
            continue
        else:
            raise SetupError(f"unsupported setup operation: {kind}")

    receipt = existing_receipt
    if receipt is None:
        receipt = {
            "schema_version": 1,
            "setup_id": plan["setup_id"],
            "local_machine_id": plan["local_machine_id"],
            "plan_sha256": plan["plan_sha256"],
            "configuration_sha256": _sha256(configuration),
            "applied_at": _utc_now(),
            "git_clone_performed": clone_performed,
            "remote_push_performed": False,
            "credentials_stored": False,
        }
        _write_new_json(receipt_path, receipt)
    elif receipt.get("configuration_sha256") != _sha256(configuration):
        raise SetupError("existing setup receipt configuration hash is invalid")
    _write_new_json(config_path, configuration)
    discoverable = (
        locate_configuration()["configuration_path"] == str(config_path)
    )
    return {
        "state": "configured",
        "plan_sha256": plan["plan_sha256"],
        "configuration_path": str(config_path),
        "receipt_path": str(receipt_path),
        "configuration_discoverable": discoverable,
        "activation_required": (
            None
            if discoverable
            else {
                "environment_variable": CONFIG_ENV_VAR,
                "required_value": str(config_path),
            }
        ),
        "git_clone_performed": clone_performed,
        "remote_push_performed": False,
    }


def _runtime_checks(
    configuration: dict[str, Any],
    local: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    conda = shutil.which("conda")
    if conda is None:
        blockers.append("conda executable is unavailable")
    else:
        completed = _run([conda, "env", "list", "--json"])
        if completed.returncode != 0:
            blockers.append("conda environment list could not be read")
        else:
            try:
                roots = json.loads(completed.stdout).get("envs", [])
            except json.JSONDecodeError:
                roots = []
            names = {Path(item).name for item in roots if isinstance(item, str)}
            if configuration["conda_env"] not in names:
                blockers.append(
                    f"conda environment {configuration['conda_env']} is unavailable"
                )
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        blockers.append("nvidia-smi is unavailable")
    else:
        completed = _run(
            [
                nvidia_smi,
                "--query-gpu=index",
                "--format=csv,noheader,nounits",
            ]
        )
        indexes: set[int] = set()
        if completed.returncode == 0:
            for line in completed.stdout.splitlines():
                try:
                    indexes.add(int(line.strip()))
                except ValueError:
                    continue
        if local["gpu_index"] not in indexes:
            blockers.append(
                f"GPU index {local['gpu_index']} is unavailable"
            )
    return blockers


def verify_configuration(
    configuration_value: dict[str, Any],
    *,
    check_runtime: bool = True,
    check_remote: bool = True,
    loaded_from: str | Path | None = None,
) -> dict[str, Any]:
    """Verify one machine's local prerequisites without changing local or remote state."""
    configuration = configuration_value
    expected_keys = {
        "schema_version",
        "setup_id",
        "setup_mode",
        "configuration_path",
        "receipt_path",
        "configured_from_answers_sha256",
        "conda_env",
        "default_seed",
        "source_remote_url",
        "local_machine_id",
        "machines",
        "distributed",
    }
    _check_exact_keys(configuration, expected_keys, "configuration")
    if configuration["schema_version"] != 1:
        raise SetupError("configuration schema_version must be 1")
    configuration_path = _absolute_path(
        configuration["configuration_path"],
        "configuration.configuration_path",
    )
    receipt_path = _absolute_path(
        configuration["receipt_path"],
        "configuration.receipt_path",
    )
    if (
        configuration_path.name != "configuration.json"
        or receipt_path != configuration_path.parent / "setup_receipt.json"
    ):
        raise SetupError("configuration and receipt paths are invalid")
    answers_like = {
        "version": 1,
        "setup_id": configuration["setup_id"],
        "setup_mode": configuration["setup_mode"],
        "configuration_dir": str(configuration_path.parent),
        "conda_env": configuration["conda_env"],
        "default_seed": configuration["default_seed"],
        "source_remote_url": configuration["source_remote_url"],
        "local_machine_id": configuration["local_machine_id"],
        "machines": configuration["machines"],
        "distributed": configuration["distributed"],
    }
    normalized_answers = normalize_answers(answers_like)
    configured_from = configuration["configured_from_answers_sha256"]
    if (
        not isinstance(configured_from, str)
        or not SHA256_RE.fullmatch(configured_from)
        or _sha256(normalized_answers) != configured_from
    ):
        raise SetupError("configuration answers hash binding is invalid")
    local = _local_machine(configuration)
    blockers: list[str] = []
    warnings: list[str] = []
    if locate_configuration()["configuration_path"] != str(configuration_path):
        warnings.append(
            f"{CONFIG_ENV_VAR} does not currently discover this configuration"
        )
    if loaded_from is not None:
        actual_path = Path(loaded_from).resolve(strict=False)
        if actual_path != configuration_path:
            raise SetupError(
                "loaded configuration path differs from its embedded path"
            )
        if not actual_path.is_file() or actual_path.is_symlink():
            blockers.append("configuration file is missing or symlinked")
        if not receipt_path.is_file() or receipt_path.is_symlink():
            blockers.append("setup receipt is missing or symlinked")
        else:
            try:
                receipt = _load_object(receipt_path, "setup receipt")
                if (
                    receipt.get("schema_version") != 1
                    or receipt.get("setup_id") != configuration["setup_id"]
                    or receipt.get("local_machine_id")
                    != configuration["local_machine_id"]
                    or receipt.get("configuration_sha256")
                    != _sha256(configuration)
                    or receipt.get("remote_push_performed") is not False
                    or receipt.get("credentials_stored") is not False
                ):
                    blockers.append(
                        "setup receipt identity or safety binding is invalid"
                    )
            except SetupError as exc:
                blockers.append(str(exc))
    source = Path(local["source_repo"])
    try:
        _git_root(source, "local source_repo")
        source_origin = _git(
            source,
            "remote",
            "get-url",
            "origin",
        ).stdout.strip()
        if source_origin != configuration["source_remote_url"]:
            blockers.append("local source_repo origin differs from configuration")
        if _git(source, "status", "--porcelain=v1").stdout:
            blockers.append("local source repository is dirty")
    except SetupError as exc:
        blockers.append(str(exc))
    for field in (
        "state_dir",
        "evaluation_output_dir",
        "hardware_feedback_output_dir",
    ):
        path = Path(local[field])
        if not path.is_dir() or path.is_symlink():
            blockers.append(f"{field} is not an existing regular directory")
    baseline_parent = Path(local["effective_config_baseline_path"]).parent
    if not baseline_parent.is_dir() or baseline_parent.is_symlink():
        blockers.append(
            "effective_config_baseline_path parent is not an existing regular directory"
        )
    if local["policy_storage_root"] is not None:
        try:
            storage = Path(local["policy_storage_root"])
            _git_root(storage, "policy_storage_root")
            if _git(storage, "status", "--porcelain=v1").stdout:
                warnings.append("policy_storage_root is currently dirty")
        except SetupError as exc:
            blockers.append(str(exc))
    if configuration["setup_mode"] == "git_mailbox":
        mailbox = Path(local["mailbox_repo"])
        try:
            _git_root(mailbox, "mailbox_repo")
            origin = _git(mailbox, "remote", "get-url", "origin").stdout.strip()
            if origin != configuration["distributed"]["remote_url"]:
                blockers.append("mailbox_repo origin differs from configuration")
            if _git(mailbox, "status", "--porcelain=v1").stdout:
                blockers.append("mailbox_repo is dirty")
        except SetupError as exc:
            blockers.append(str(exc))
        if check_remote:
            completed = _run(
                [
                    "git",
                    "ls-remote",
                    "--heads",
                    configuration["distributed"]["remote_url"],
                ],
                timeout=30,
            )
            if completed.returncode != 0:
                blockers.append(
                    "coordination remote is unreachable without an interactive prompt"
                )
            elif not completed.stdout.strip():
                blockers.append(
                    "coordination remote has no branch; create one initial commit first"
                )
        else:
            warnings.append("coordination remote connectivity was not checked")
    if check_remote:
        completed = _run(
            [
                "git",
                "ls-remote",
                "--heads",
                configuration["source_remote_url"],
            ],
            timeout=30,
        )
        if completed.returncode != 0:
            blockers.append(
                "source remote is unreachable without an interactive prompt"
            )
        elif not completed.stdout.strip():
            blockers.append("source remote has no branch")
    else:
        warnings.append("source remote connectivity was not checked")
    if check_runtime:
        blockers.extend(_runtime_checks(configuration, local))
    else:
        warnings.append("Conda and GPU runtime checks were skipped")
    return {
        "schema_version": 1,
        "configuration_sha256": _sha256(configuration),
        "setup_id": configuration["setup_id"],
        "local_machine_id": configuration["local_machine_id"],
        "configuration_status": "valid",
        "ready_for_training": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "remote_write_tested": False,
        "credentials_present_in_configuration": False,
    }


def _write_plan(path: Path, plan: dict[str, Any]) -> None:
    _write_new_json(path, plan)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    subparsers.add_parser("locate")

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--answers", required=True)
    plan_parser.add_argument("--output", required=True)

    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--plan", required=True)
    apply_parser.add_argument("--approval-sha256", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip only remote connectivity; local Git, Conda, and GPU checks remain",
    )

    args = parser.parse_args()
    try:
        if args.action == "locate":
            result = locate_configuration()
        elif args.action == "plan":
            plan = build_plan(_load_object(args.answers, "setup answers"))
            _write_plan(Path(args.output), plan)
            result = {
                "state": "approval_required",
                "plan_path": str(Path(args.output).resolve(strict=False)),
                "plan_sha256": plan["plan_sha256"],
                "operations": plan["operations"],
                "discovery": plan["discovery"],
                "remote_push_permitted": False,
            }
        elif args.action == "apply":
            result = apply_plan(
                _load_object(args.plan, "setup plan"),
                args.approval_sha256,
            )
        else:
            result = verify_configuration(
                _load_object(args.config, "skill configuration"),
                check_remote=not args.offline,
                loaded_from=args.config,
            )
    except SetupError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0 if result.get("ready_for_training", True) else 3


if __name__ == "__main__":
    raise SystemExit(main())

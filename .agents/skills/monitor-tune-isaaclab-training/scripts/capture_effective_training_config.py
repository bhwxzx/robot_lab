#!/usr/bin/env python3
"""Capture immutable effective RSL-RL configuration evidence for one run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from capture_run_identity import RunIdentityError, validate_run_identity


class EffectiveConfigError(ValueError):
    """Raised when effective training configuration evidence is unsafe."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_symlink_components(path: Path, *, label: str) -> None:
    if not path.is_absolute():
        raise EffectiveConfigError(f"{label} must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise EffectiveConfigError(
                f"{label} contains a symlinked path component: {current}"
            )


def _signature(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_stable_utf8(path: Path, *, label: str) -> tuple[str, dict[str, Any]]:
    _reject_symlink_components(path, label=label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as exc:
        raise EffectiveConfigError(f"{label} does not exist: {path}") from exc
    except OSError as exc:
        raise EffectiveConfigError(f"cannot open {label}: {exc}") from exc
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise EffectiveConfigError(f"{label} must be a regular file")
        content = stream.read()
        after = os.fstat(stream.fileno())
    if _signature(before) != _signature(after) or len(content) != after.st_size:
        raise EffectiveConfigError(f"{label} changed while it was read")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EffectiveConfigError(f"{label} must be UTF-8") from exc
    return text, {
        "path": str(path),
        "sha256": _sha256_bytes(content),
        "size_bytes": len(content),
        "signature": _signature(after),
    }


def _verify_unchanged(path: Path, metadata: dict[str, Any], *, label: str) -> None:
    _reject_symlink_components(path, label=label)
    try:
        current = os.stat(path, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise EffectiveConfigError(f"{label} disappeared after capture") from exc
    if not stat.S_ISREG(current.st_mode):
        raise EffectiveConfigError(f"{label} must remain a regular file")
    if _signature(current) != tuple(metadata["signature"]):
        raise EffectiveConfigError(f"{label} changed during paired capture")


def _scalar_value(node: ScalarNode, *, label: str) -> Any:
    tag = node.tag
    if tag == "tag:yaml.org,2002:null":
        return None
    if tag == "tag:yaml.org,2002:bool":
        value = node.value.casefold()
        if value in {"true", "yes", "on"}:
            return True
        if value in {"false", "no", "off"}:
            return False
        raise EffectiveConfigError(f"{label} contains an invalid YAML boolean")
    if tag in {"tag:yaml.org,2002:int", "tag:yaml.org,2002:float"}:
        value = yaml.safe_load(node.value)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EffectiveConfigError(f"{label} contains an invalid YAML number")
        if isinstance(value, float) and not math.isfinite(value):
            raise EffectiveConfigError(f"{label} contains a non-finite value")
        return value
    return node.value


def _mapping_items(node: Node, *, label: str) -> dict[str, Node]:
    if not isinstance(node, MappingNode):
        raise EffectiveConfigError(f"{label} must be a YAML mapping")
    result: dict[str, Node] = {}
    for key_node, value_node in node.value:
        if not isinstance(key_node, ScalarNode):
            raise EffectiveConfigError(f"{label} contains a non-scalar key")
        key = _scalar_value(key_node, label=f"{label} key")
        if not isinstance(key, str) or not key:
            raise EffectiveConfigError(f"{label} contains an invalid key")
        if key in result:
            raise EffectiveConfigError(f"{label} contains duplicate key: {key}")
        result[key] = value_node
    return result


def _canonical_node(node: Node, *, label: str) -> dict[str, Any]:
    if isinstance(node, ScalarNode):
        value = _scalar_value(node, label=label)
        return {"kind": "scalar", "tag": node.tag, "value": value}
    if isinstance(node, SequenceNode):
        return {
            "kind": "sequence",
            "tag": node.tag,
            "items": [
                _canonical_node(item, label=f"{label}[{index}]")
                for index, item in enumerate(node.value)
            ],
        }
    if isinstance(node, MappingNode):
        items = _mapping_items(node, label=label)
        return {
            "kind": "mapping",
            "tag": node.tag,
            "items": {
                key: _canonical_node(items[key], label=f"{label}.{key}")
                for key in sorted(items)
            },
        }
    raise EffectiveConfigError(f"{label} contains an unsupported YAML node")


def _parse_yaml(text: str, *, label: str) -> tuple[MappingNode, dict[str, Any]]:
    try:
        root = yaml.compose(text)
    except yaml.YAMLError as exc:
        raise EffectiveConfigError(f"invalid {label} YAML: {exc}") from exc
    if not isinstance(root, MappingNode):
        raise EffectiveConfigError(f"{label} must contain a YAML mapping")
    return root, _canonical_node(root, label=label)


def _path_node(root: Node, path: tuple[str, ...], *, label: str) -> Node:
    current = root
    for component in path:
        items = _mapping_items(current, label=label)
        if component not in items:
            raise EffectiveConfigError(
                f"{label} is missing required path: {'.'.join(path)}"
            )
        current = items[component]
    return current


def _required_scalar(root: Node, path: tuple[str, ...], *, label: str) -> Any:
    node = _path_node(root, path, label=label)
    if not isinstance(node, ScalarNode):
        raise EffectiveConfigError(
            f"{label}.{'.'.join(path)} must be a scalar"
        )
    return _scalar_value(node, label=f"{label}.{'.'.join(path)}")


def _optional_scalar(root: Node, path: tuple[str, ...], *, label: str) -> Any:
    current = root
    for component in path:
        if not isinstance(current, MappingNode):
            return None
        items = _mapping_items(current, label=label)
        if component not in items:
            return None
        current = items[component]
    if not isinstance(current, ScalarNode):
        return None
    return _scalar_value(current, label=f"{label}.{'.'.join(path)}")


def _reward_inventory(rewards: Node) -> tuple[dict[str, Any], dict[str, Any]]:
    terms = _mapping_items(rewards, label="env.rewards")
    if not terms:
        raise EffectiveConfigError("env.rewards must contain at least one term")
    inventory: dict[str, Any] = {}
    weights: dict[str, Any] = {}
    for name in sorted(terms):
        term = terms[name]
        if isinstance(term, ScalarNode) and _scalar_value(
            term,
            label=f"env.rewards.{name}",
        ) is None:
            inventory[name] = {"enabled": False, "weight": None}
            weights[name] = None
            continue
        term_items = _mapping_items(term, label=f"env.rewards.{name}")
        weight_node = term_items.get("weight")
        if not isinstance(weight_node, ScalarNode):
            raise EffectiveConfigError(
                f"env.rewards.{name}.weight must be a scalar"
            )
        weight = _scalar_value(
            weight_node,
            label=f"env.rewards.{name}.weight",
        )
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise EffectiveConfigError(
                f"env.rewards.{name}.weight must be a finite number"
            )
        inventory[name] = {
            "enabled": True,
            "weight": weight,
            "config_fingerprint": _sha256_bytes(
                _canonical_json(
                    _canonical_node(term, label=f"env.rewards.{name}")
                ).encode("utf-8")
            ),
        }
        weights[name] = weight
    return inventory, weights


def _load_identity(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EffectiveConfigError(f"run identity does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EffectiveConfigError(
            f"invalid run identity JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    try:
        validate_run_identity(value)
    except RunIdentityError as exc:
        raise EffectiveConfigError(str(exc)) from exc
    assert isinstance(value, dict)
    return value


def _command_contains_task(command: list[str], task: str) -> bool:
    for index, token in enumerate(command):
        if token == f"--task={task}":
            return True
        if token == "--task" and index + 1 < len(command):
            return command[index + 1] == task
    return False


def capture_effective_config(
    run_identity: dict[str, Any],
    log_directory: Path,
) -> dict[str, Any]:
    """Return deterministic evidence from one run's effective YAML dumps."""
    try:
        validate_run_identity(run_identity)
    except RunIdentityError as exc:
        raise EffectiveConfigError(str(exc)) from exc
    repository_root = Path(run_identity["source"]["repository_root"])
    if not _command_contains_task(
        run_identity["training"]["command"],
        run_identity["task"],
    ):
        raise EffectiveConfigError(
            "run_identity.training.command must contain the exact task"
        )
    _reject_symlink_components(repository_root, label="repository root")
    _reject_symlink_components(log_directory, label="log directory")
    if repository_root != repository_root.resolve():
        raise EffectiveConfigError("repository root must be a canonical path")
    if log_directory != log_directory.resolve():
        raise EffectiveConfigError("log directory must be a canonical path")
    if not log_directory.is_dir():
        raise EffectiveConfigError("log directory must be an existing directory")
    expected_log_root = repository_root / "logs" / "rsl_rl"
    try:
        relative_log_directory = log_directory.relative_to(expected_log_root)
    except ValueError as exc:
        raise EffectiveConfigError(
            "log directory must be below the run identity repository logs/rsl_rl"
        ) from exc
    if not relative_log_directory.parts:
        raise EffectiveConfigError("log directory must identify one training run")
    if run_identity["run_id"] != log_directory.name:
        raise EffectiveConfigError(
            "run_identity.run_id does not match the run log directory"
        )

    env_path = log_directory / "params" / "env.yaml"
    agent_path = log_directory / "params" / "agent.yaml"
    env_text, env_source = _read_stable_utf8(env_path, label="env config")
    agent_text, agent_source = _read_stable_utf8(agent_path, label="agent config")
    _verify_unchanged(env_path, env_source, label="env config")
    _verify_unchanged(agent_path, agent_source, label="agent config")

    env_root, env_canonical = _parse_yaml(env_text, label="env config")
    agent_root, agent_canonical = _parse_yaml(agent_text, label="agent config")
    rewards = _path_node(env_root, ("rewards",), label="env config")
    reward_inventory, reward_weights = _reward_inventory(rewards)

    env_seed = _required_scalar(env_root, ("seed",), label="env config")
    agent_seed = _required_scalar(agent_root, ("seed",), label="agent config")
    runner_class = _required_scalar(
        agent_root,
        ("class_name",),
        label="agent config",
    )
    algorithm_class = _required_scalar(
        agent_root,
        ("algorithm", "class_name"),
        label="agent config",
    )
    experiment_name = _required_scalar(
        agent_root,
        ("experiment_name",),
        label="agent config",
    )
    if env_seed != run_identity["seed"] or agent_seed != run_identity["seed"]:
        raise EffectiveConfigError(
            "effective env/agent seed does not match run_identity.seed"
        )
    if runner_class != run_identity["runner"]:
        raise EffectiveConfigError(
            "effective runner class does not match run_identity.runner"
        )
    if not isinstance(algorithm_class, str) or not algorithm_class:
        raise EffectiveConfigError("effective algorithm class must be a string")
    if experiment_name != log_directory.parent.name:
        raise EffectiveConfigError(
            "effective experiment_name does not match the run log directory"
        )

    env_fingerprint = _sha256_bytes(
        _canonical_json(env_canonical).encode("utf-8")
    )
    agent_fingerprint = _sha256_bytes(
        _canonical_json(agent_canonical).encode("utf-8")
    )
    reward_fingerprint = _sha256_bytes(
        _canonical_json(
            _canonical_node(rewards, label="env.rewards")
        ).encode("utf-8")
    )
    effective_fingerprint = _sha256_bytes(
        _canonical_json(
            {"agent": agent_fingerprint, "environment": env_fingerprint}
        ).encode("utf-8")
    )

    for source in (env_source, agent_source):
        source.pop("signature")
    return {
        "version": 1,
        "task": run_identity["task"],
        "run_id": run_identity["run_id"],
        "host_id": run_identity["host_id"],
        "run_identity_sha256": run_identity["identity_sha256"],
        "log_directory": str(log_directory),
        "source_files": {
            "environment": {**env_source, "content_utf8": env_text},
            "agent": {**agent_source, "content_utf8": agent_text},
        },
        "resolved_identity": {
            "declared_algorithm": run_identity["algorithm"],
            "algorithm_class": algorithm_class,
            "runner_class": runner_class,
            "seed": agent_seed,
        },
        "training_parameters": {
            "device": _optional_scalar(agent_root, ("device",), label="agent config"),
            "experiment_name": _optional_scalar(
                agent_root,
                ("experiment_name",),
                label="agent config",
            ),
            "logger": _optional_scalar(agent_root, ("logger",), label="agent config"),
            "max_iterations": _optional_scalar(
                agent_root,
                ("max_iterations",),
                label="agent config",
            ),
            "num_envs": _optional_scalar(
                env_root,
                ("scene", "num_envs"),
                label="env config",
            ),
            "num_steps_per_env": _optional_scalar(
                agent_root,
                ("num_steps_per_env",),
                label="agent config",
            ),
            "resume": _optional_scalar(agent_root, ("resume",), label="agent config"),
            "run_name": _optional_scalar(agent_root, ("run_name",), label="agent config"),
        },
        "reward_terms": reward_inventory,
        "reward_weights": reward_weights,
        "fingerprints": {
            "agent": agent_fingerprint,
            "effective_config": effective_fingerprint,
            "environment": env_fingerprint,
            "reward": reward_fingerprint,
        },
    }


def write_new_evidence(path: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    """Atomically publish one new effective-config evidence JSON."""
    if not path.is_absolute():
        raise EffectiveConfigError("--output must be a new absolute path")
    _reject_symlink_components(path.parent, label="--output")
    if not path.parent.is_dir():
        raise EffectiveConfigError("--output parent directory does not exist")
    if path.exists() or path.is_symlink():
        raise EffectiveConfigError("--output must be a new absolute path")
    encoded = json.dumps(
        evidence,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{os.urandom(8).hex()}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise EffectiveConfigError("--output must be a new absolute path") from exc
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "version": 1,
        "effective_config_fingerprint": evidence["fingerprints"]["effective_config"],
        "evidence_path": str(path),
        "sha256": _sha256_bytes(encoded.encode("utf-8")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_identity", help="Validated host-local run identity JSON")
    parser.add_argument("--log-dir", required=True, help="Absolute RSL-RL run directory")
    parser.add_argument("--output", required=True, help="New absolute evidence JSON path")
    args = parser.parse_args()
    try:
        evidence = capture_effective_config(
            _load_identity(Path(args.run_identity)),
            Path(args.log_dir),
        )
        receipt = write_new_evidence(Path(args.output), evidence)
    except (EffectiveConfigError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

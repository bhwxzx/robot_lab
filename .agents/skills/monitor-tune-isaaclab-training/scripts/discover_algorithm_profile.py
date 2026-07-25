#!/usr/bin/env python3
"""Discover an algorithm identity and propose a profile for an unknown trainer."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from algorithm_profiles import (
    DEFAULT_REGISTRY_PATH,
    ProfileError,
    load_registry,
    match_profile,
    normalize_metric_name,
    profile_fingerprint,
)


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMERIC_LABEL_RE = re.compile(
    r"^\s*([^:]+):\s*[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)s?\s*$",
    re.IGNORECASE,
)
CLASS_NAME_RE = re.compile(r"^\s*class_name\s*:\s*['\"]?([A-Za-z0-9_]+)", re.MULTILINE)


def _load_draft(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"draft session does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid draft JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("training"), dict):
        raise ProfileError("draft session must contain a training object")
    return data


def _infer_backend(command: list[Any]) -> str:
    joined = " ".join(str(item) for item in command).lower()
    if "rsl_rl" in joined:
        return "rsl_rl"
    if "skrl" in joined:
        return "skrl"
    if "cusrl" in joined:
        return "cusrl"
    return "custom"


def _config_class_names(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise ProfileError(f"config file does not exist: {path}") from exc
    return CLASS_NAME_RE.findall(text)


def _infer_identity(draft: dict[str, Any], config_path: Path | None) -> dict[str, str]:
    training = draft["training"]
    command = training.get("command")
    if not isinstance(command, list) or not command:
        raise ProfileError("draft training.command must be a non-empty argv array")
    algorithm_obj = draft.get("algorithm")
    if not isinstance(algorithm_obj, dict):
        algorithm_obj = {}
    backend = algorithm_obj.get("backend")
    if not isinstance(backend, str) or backend in {"", "auto"}:
        backend = _infer_backend(command)
    algorithm = algorithm_obj.get("name")
    runner = algorithm_obj.get("runner_class")

    class_names = _config_class_names(config_path)
    if not isinstance(runner, str) or runner in {"", "auto"}:
        runner_candidates = [name for name in class_names if "Runner" in name]
        runner = runner_candidates[0] if runner_candidates else "unknown"
    if not isinstance(algorithm, str) or algorithm in {"", "auto"}:
        algorithm_candidates = [
            name
            for name in class_names
            if name != runner and ("PPO" in name or name == "Distillation")
        ]
        algorithm = algorithm_candidates[0] if algorithm_candidates else "unknown"
    return {"backend": backend, "name": algorithm, "runner_class": runner}


def _discover_metric_aliases(log_path: Path | None) -> dict[str, str]:
    if log_path is None:
        return {}
    try:
        stream = log_path.open("r", encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise ProfileError(f"log file does not exist: {log_path}") from exc
    aliases: dict[str, str] = {}
    with stream:
        for raw_line in stream:
            line = ANSI_RE.sub("", raw_line.rstrip("\n"))
            match = NUMERIC_LABEL_RE.match(line)
            if match:
                label = match.group(1).strip()
                aliases[label] = normalize_metric_name(label)
            if len(aliases) >= 512:
                break
    return dict(sorted(aliases.items()))


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"


def discover(
    draft: dict[str, Any],
    registry: dict[str, Any],
    log_path: Path | None,
    config_path: Path | None,
) -> dict[str, Any]:
    """Return a matched profile or a candidate specific profile."""
    identity = _infer_identity(draft, config_path)
    matched = match_profile(
        registry,
        identity["backend"],
        identity["name"],
        identity["runner_class"],
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "identity": identity,
        "matched_profile": {
            "id": matched["id"],
            "profile_version": matched["profile_version"],
            "is_generic": matched["is_generic"],
            "fingerprint": profile_fingerprint(matched),
        },
    }
    if not matched["is_generic"]:
        result["status"] = "matched"
        return result

    parent_id = matched["id"]
    candidate_id = (
        f"{_safe_id(identity['backend'])}-"
        f"{_safe_id(identity['name'])}-"
        f"{_safe_id(identity['runner_class'])}"
    )
    result["status"] = "candidate"
    result["candidate_profile"] = {
        "id": candidate_id,
        "profile_version": 2,
        "is_generic": False,
        "extends": parent_id,
        "match": {
            "backends": [identity["backend"]],
            "algorithm_names": [identity["name"]],
            "runner_classes": [identity["runner_class"]],
        },
        "metric_aliases": _discover_metric_aliases(log_path),
        "protected_parameter_patterns": [],
        "evaluation_capabilities": {
            "play_entrypoint": matched["evaluation_capabilities"].get(
                "play_entrypoint"
            ),
            "supported_artifacts": ["native"],
            "history_contract": "review_required",
        },
    }
    result["required_next_action"] = "review_and_approve_persistent_profile"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("draft_session", help="Draft session JSON; auto identities are allowed")
    parser.add_argument("--config", help="Optional dumped trainer configuration")
    parser.add_argument("--log", help="Optional console log for metric discovery")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--output")
    args = parser.parse_args()
    try:
        result = discover(
            _load_draft(Path(args.draft_session)),
            load_registry(args.registry),
            Path(args.log) if args.log else None,
            Path(args.config) if args.config else None,
        )
    except ProfileError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

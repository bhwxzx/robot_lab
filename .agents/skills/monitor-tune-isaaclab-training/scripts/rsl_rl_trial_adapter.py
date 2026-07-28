#!/usr/bin/env python3
"""Run one exact RSL-RL trial and emit atomic, machine-verifiable artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from algorithm_profiles import load_registry, resolve_profile
from summarize_training_log import StreamingLogSummary
from validate_session_spec import SpecError


LOG_ROOT_RE = re.compile(r"Logging experiment in directory:\s*(.+?)\s*$")
RUN_STAMP_RE = re.compile(
    r"Exact experiment name requested from command line:\s*"
    r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\s*$"
)
MODEL_RE = re.compile(r"model_(\d+)\.pt$")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise SpecError(f"{label} must be an existing absolute regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid {label} JSON at line {exc.lineno}: {exc.msg}"
        ) from exc
    if not isinstance(value, dict):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or path.parent.is_symlink():
        raise SpecError(
            f"refusing to overwrite or write outside a regular parent: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _atomic_replace(path: Path, value: dict[str, Any]) -> None:
    if (
        not path.is_absolute()
        or path.parent.is_symlink()
        or (path.exists() and (not path.is_file() or path.is_symlink()))
    ):
        raise SpecError(f"refusing unsafe atomic replacement: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _rolling_summary(
    parser: StreamingLogSummary,
    rsl_run_dir: Path | None,
    include_current: bool = False,
) -> dict[str, Any]:
    summary = parser.snapshot(include_current=include_current)
    summary["live_evidence"] = {
        "updated_at": time.time(),
        "lines_seen": parser.lines_seen,
        "rsl_rl_run_dir": str(rsl_run_dir) if rsl_run_dir is not None else None,
    }
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hydra_value(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SpecError("override values must be finite JSON data") from exc
    return encoded


def build_child_argv(
    contract: dict[str, Any],
    overrides: dict[str, Any],
) -> list[str]:
    """Build exact argv from the approved base command and parameter map."""
    base = contract.get("training_argv")
    mapping = contract.get("parameter_cli_map")
    if (
        not isinstance(base, list)
        or not base
        or not all(isinstance(token, str) and token for token in base)
        or not isinstance(mapping, dict)
    ):
        raise SpecError("adapter contract command or parameter map is invalid")
    unknown = sorted(set(overrides) - set(mapping))
    if unknown:
        raise SpecError(f"adapter received unauthorized overrides: {unknown}")
    reserved = ("--seed", "--run_name", "--run-name")
    if any(
        token == prefix or token.startswith(prefix + "=")
        for token in base
        for prefix in reserved
    ):
        raise SpecError("base command already sets adapter-managed seed or run name")
    argv = [
        *base,
        "--seed",
        str(contract["seed"]),
        "--run_name",
        contract["run_id"],
    ]
    for path in sorted(overrides):
        cli_path = mapping[path]
        if not isinstance(cli_path, str) or not cli_path:
            raise SpecError(f"adapter CLI mapping is invalid for {path}")
        argv.append(f"{cli_path}={_hydra_value(overrides[path])}")
    if contract["version"] == 2:
        fidelity = contract["multi_fidelity"]
        budget_path = fidelity["budget_cli_path"]
        argv.append(
            f"{budget_path}={_hydra_value(contract['target_budget'])}"
        )
        resume = contract["resume_from"]
        if resume is not None:
            checkpoint_path = Path(resume["checkpoint_path"])
            rsl_run_dir = Path(resume["rsl_rl_run_dir"])
            if (
                not checkpoint_path.is_absolute()
                or not checkpoint_path.is_file()
                or checkpoint_path.is_symlink()
                or not rsl_run_dir.is_absolute()
                or not rsl_run_dir.is_dir()
                or rsl_run_dir.is_symlink()
                or checkpoint_path.parent.resolve() != rsl_run_dir.resolve()
                or _sha256(checkpoint_path) != resume["checkpoint_sha256"]
            ):
                raise SpecError(
                    "multi-fidelity parent checkpoint is missing, linked, "
                    "moved, or hash-mismatched"
                )
            resume_paths = fidelity["resume_cli_paths"]
            run_reference = (
                rsl_run_dir.name
                if fidelity["load_run_reference"] == "basename"
                else str(rsl_run_dir)
            )
            argv.extend(
                [
                    f"{resume_paths['enabled']}=true",
                    f"{resume_paths['load_run']}={_hydra_value(run_reference)}",
                    (
                        f"{resume_paths['load_checkpoint']}="
                        f"{_hydra_value(checkpoint_path.name)}"
                    ),
                ]
            )
    return argv


def _load_yaml(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SpecError(f"RSL-RL {label} config is missing or linked")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"invalid RSL-RL {label} YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise SpecError(f"RSL-RL {label} config must be an object")
    return value


def _checkpoint(run_dir: Path, required: bool) -> dict[str, Any] | None:
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("model_*.pt"):
        match = MODEL_RE.fullmatch(path.name)
        if match and path.is_file() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        if required:
            raise SpecError("completed RSL-RL trial produced no model_N.pt checkpoint")
        return None
    step, path = max(candidates)
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "step": step,
    }


def _validate_contract(contract: dict[str, Any]) -> None:
    required = {
        "version",
        "adapter_id",
        "profile_id",
        "training_argv",
        "training_cwd",
        "parameter_cli_map",
        "summary_last",
        "required_metrics",
        "require_checkpoint",
        "run_id",
        "trial_id",
        "stage",
        "seed",
    }
    if contract.get("version") == 2:
        required.update(
            {
                "rung",
                "target_budget",
                "resume_from",
                "multi_fidelity",
            }
        )
    if set(contract) != required:
        raise SpecError(
            f"adapter contract fields do not match schema: {sorted(set(contract) ^ required)}"
        )
    if contract["version"] not in {1, 2} or contract["adapter_id"] != "rsl-rl":
        raise SpecError("unsupported adapter contract")
    if not Path(contract["training_cwd"]).is_absolute():
        raise SpecError("adapter training_cwd must be absolute")
    if (
        isinstance(contract["seed"], bool)
        or not isinstance(contract["seed"], int)
        or not isinstance(contract["summary_last"], int)
        or not 1 <= contract["summary_last"] <= 10000
        or not isinstance(contract["require_checkpoint"], bool)
        or not isinstance(contract["required_metrics"], list)
        or not all(
            isinstance(metric, str) and metric
            for metric in contract["required_metrics"]
        )
    ):
        raise SpecError("adapter contract scalar fields are invalid")
    if contract["version"] == 2:
        fidelity = contract["multi_fidelity"]
        resume = contract["resume_from"]
        if (
            isinstance(contract["rung"], bool)
            or not isinstance(contract["rung"], int)
            or contract["rung"] < 1
            or isinstance(contract["target_budget"], bool)
            or not isinstance(contract["target_budget"], int)
            or contract["target_budget"] < 1
            or not isinstance(fidelity, dict)
            or set(fidelity)
            != {
                "budget_cli_path",
                "resume_cli_paths",
                "load_run_reference",
            }
            or not isinstance(fidelity.get("budget_cli_path"), str)
            or not isinstance(fidelity.get("resume_cli_paths"), dict)
            or set(fidelity["resume_cli_paths"])
            != {"enabled", "load_run", "load_checkpoint"}
            or not all(
                isinstance(path, str) and path
                for path in fidelity["resume_cli_paths"].values()
            )
            or fidelity.get("load_run_reference")
            not in {"basename", "absolute"}
        ):
            raise SpecError("multi-fidelity adapter contract is invalid")
        if resume is not None and (
            not isinstance(resume, dict)
            or set(resume)
            != {
                "parent_run_id",
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_step",
                "rsl_rl_run_dir",
            }
            or not isinstance(resume.get("parent_run_id"), str)
            or not isinstance(resume.get("checkpoint_path"), str)
            or not re.fullmatch(
                r"[0-9a-f]{64}",
                str(resume.get("checkpoint_sha256")),
            )
            or isinstance(resume.get("checkpoint_step"), bool)
            or not isinstance(resume.get("checkpoint_step"), int)
            or not isinstance(resume.get("rsl_rl_run_dir"), str)
        ):
            raise SpecError(
                "multi-fidelity resume checkpoint contract is invalid"
            )


def run_trial(
    contract_path: Path,
    executor_run_id: str,
    overrides: dict[str, Any],
    effective_config_path: Path,
    result_path: Path,
    summary_path: Path,
    terminal_path: Path,
    log_path: Path,
) -> int:
    """Execute one RSL-RL child and produce artifacts on complete success."""
    started_at = time.time()
    contract = _load_object(contract_path, "adapter contract")
    _validate_contract(contract)
    if executor_run_id != contract["run_id"]:
        raise SpecError("executor run ID does not match adapter contract")
    child_argv: list[str] = []
    terminal: dict[str, Any] = {
        "version": contract["version"],
        "adapter_id": "rsl-rl",
        "run_id": contract["run_id"],
        "trial_id": contract["trial_id"],
        "stage": contract["stage"],
        "seed": contract["seed"],
        "status": "adapter_failed",
        "exit_code": None,
        "started_at": started_at,
        "finished_at": None,
        "child_argv": None,
        "rsl_rl_run_dir": None,
        "checkpoint": None,
        "summary_updates": 0,
        "failure_reason": None,
    }
    if contract["version"] == 2:
        terminal["rung"] = contract["rung"]
        terminal["target_budget"] = contract["target_budget"]
        terminal["resume_from"] = contract["resume_from"]
    try:
        child_argv = build_child_argv(contract, overrides)
        terminal["child_argv"] = child_argv
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        profile = resolve_profile(
            load_registry(),
            contract["profile_id"],
        )
        parser = StreamingLogSummary(
            log_path,
            contract["summary_last"],
            profile,
        )
        process = subprocess.Popen(
            child_argv,
            cwd=contract["training_cwd"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        log_root: Path | None = None
        run_stamp: str | None = None
        rsl_run_dir: Path | None = None
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            root_match = LOG_ROOT_RE.search(line)
            if root_match:
                log_root = Path(root_match.group(1)).resolve()
            stamp_match = RUN_STAMP_RE.search(line)
            if stamp_match:
                run_stamp = stamp_match.group(1)
            if log_root is not None and run_stamp is not None:
                rsl_run_dir = (
                    log_root / f"{run_stamp}_{contract['run_id']}"
                ).resolve()
            event = parser.feed_line(line)
            if event["completed"] or event["non_finite"]:
                _atomic_replace(
                    summary_path,
                    _rolling_summary(
                        parser,
                        rsl_run_dir,
                        include_current=event["non_finite"],
                    ),
                )
                terminal["summary_updates"] += 1
        exit_code = process.wait()
        parser.finish()
        _atomic_replace(
            summary_path,
            _rolling_summary(parser, rsl_run_dir),
        )
        terminal["summary_updates"] += 1
        terminal["exit_code"] = exit_code
        if exit_code != 0:
            terminal["status"] = "child_failed"
            terminal["failure_reason"] = f"RSL-RL child exited with code {exit_code}"
            return exit_code if 1 <= exit_code <= 125 else 2
        if log_root is None or run_stamp is None:
            raise SpecError("could not resolve RSL-RL log directory from stdout")
        rsl_run_dir = (log_root / f"{run_stamp}_{contract['run_id']}").resolve()
        if (
            not rsl_run_dir.is_absolute()
            or rsl_run_dir.parent != log_root
            or not rsl_run_dir.is_dir()
            or rsl_run_dir.is_symlink()
        ):
            raise SpecError(f"resolved RSL-RL run directory is invalid: {rsl_run_dir}")
        terminal["rsl_rl_run_dir"] = str(rsl_run_dir)
        effective = {
            "env": _load_yaml(rsl_run_dir / "params" / "env.yaml", "environment"),
            "agent": _load_yaml(rsl_run_dir / "params" / "agent.yaml", "agent"),
        }
        _atomic_write(effective_config_path, effective)
        summary = _rolling_summary(parser, rsl_run_dir)
        _atomic_replace(summary_path, summary)
        terminal["summary_updates"] += 1
        if summary["non_finite_metrics"]:
            raise SpecError("RSL-RL summary contains non-finite metrics")
        metrics: dict[str, float] = {}
        for metric in contract["required_metrics"]:
            value = summary["mean"].get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise SpecError(
                    f"RSL-RL summary is missing finite required metric {metric}"
                )
            metrics[metric] = float(value)
        terminal["checkpoint"] = _checkpoint(
            rsl_run_dir,
            contract["require_checkpoint"],
        )
        result = {
            "trial_id": contract["trial_id"],
            "seed": contract["seed"],
            "status": "completed",
            "metrics": metrics,
        }
        if contract["version"] == 2:
            checkpoint = terminal["checkpoint"]
            if not isinstance(checkpoint, dict):
                raise SpecError(
                    "multi-fidelity trial requires a completed checkpoint"
                )
            result.update(
                {
                    "run_id": contract["run_id"],
                    "checkpoint": {
                        **checkpoint,
                        "rsl_rl_run_dir": str(rsl_run_dir),
                    },
                    "rung": contract["rung"],
                    "target_budget": contract["target_budget"],
                }
            )
        _atomic_write(result_path, result)
        terminal["status"] = "completed"
        return 0
    except (OSError, SpecError, subprocess.SubprocessError) as exc:
        terminal["failure_reason"] = str(exc)
        print(f"[RSL-RL-ADAPTER-ERROR] {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        terminal["finished_at"] = time.time()
        try:
            _atomic_write(terminal_path, terminal)
        except SpecError as exc:
            print(f"[RSL-RL-ADAPTER-ERROR] {exc}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--executor-run-id", required=True)
    parser.add_argument("--overrides-json", required=True)
    parser.add_argument("--effective-config", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--terminal", required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()
    try:
        overrides = json.loads(args.overrides_json)
        if not isinstance(overrides, dict):
            raise SpecError("--overrides-json must decode to an object")
    except (json.JSONDecodeError, SpecError) as exc:
        parser.error(str(exc))
    return run_trial(
        Path(args.contract).resolve(),
        args.executor_run_id,
        overrides,
        Path(args.effective_config).resolve(),
        Path(args.result).resolve(),
        Path(args.summary).resolve(),
        Path(args.terminal).resolve(),
        Path(args.log_path).resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())

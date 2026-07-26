#!/usr/bin/env python3
"""Validate a per-session authorization contract for training supervision and tuning."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path, PurePosixPath
from string import Formatter
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from algorithm_profiles import (
    DEFAULT_REGISTRY_PATH,
    ProfileError,
    load_registry,
    profile_fingerprint,
    profile_matches,
    resolve_profile,
)

PARAMETER_PATH_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
COMMAND_PLACEHOLDER_RE = re.compile(r"\{([a-z0-9_]+)\}")
ALLOWED_COMMAND_PLACEHOLDERS = {
    "artifact_path",
    "artifact_kind",
    "candidate_id",
    "artifact_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
    "command_schedule_json",
    "duration_steps",
    "executor_run_id",
    "gpu_index",
    "result_path",
    "require_idle_gpu_flag",
    "run_id",
    "scenario_id",
    "scenario_overrides_json",
    "seed",
    "video_path",
}
ARTIFACT_KINDS = {"native", "jit", "onnx"}
SCENARIO_CATEGORIES = {
    "nominal",
    "command",
    "terrain",
    "dynamics",
    "disturbance",
    "latency",
}
AGGREGATIONS = {"max", "min", "mean"}
HARDWARE_FEEDBACK_OUTPUT_MODES = {
    "proposal_only",
    "prepare_authorized_draft",
}
EXECUTION_PLACEHOLDERS = {
    "adapter_contract_path",
    "effective_config_path",
    "gpu_index",
    "log_path",
    "overrides_json",
    "result_path",
    "run_dir",
    "run_id",
    "seed",
    "stage",
    "summary_path",
    "terminal_path",
    "trial_id",
}
REQUIRED_EXECUTION_PLACEHOLDERS = {
    "effective_config_path",
    "overrides_json",
    "result_path",
    "run_dir",
    "run_id",
    "seed",
    "stage",
    "summary_path",
    "trial_id",
}


class SpecError(ValueError):
    """Raised when a session specification violates the authorization schema."""


def _expect_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{path} must be an object")
    return value


def _check_keys(value: dict[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise SpecError(f"{path} contains unknown field(s): {', '.join(unknown)}")


def _expect_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SpecError(f"{path} must be a boolean")
    return value


def _expect_int(value: Any, path: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{path} must be an integer")
    if not minimum <= value <= maximum:
        raise SpecError(f"{path} must be between {minimum} and {maximum}")
    return value


def _expect_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SpecError(f"{path} must be a finite number")
    return result


def _expect_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"{path} must be a non-empty string")
    return value


def _validate_argv(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise SpecError(f"{path} must be a non-empty argv array")
    for index, item in enumerate(value):
        _expect_nonempty_string(item, f"{path}[{index}]")
        if "\x00" in item:
            raise SpecError(f"{path}[{index}] contains a NUL byte")
    return value


def _validate_scalar(value: Any, path: str) -> None:
    if value is None or isinstance(value, (dict, list)):
        raise SpecError(f"{path} must be a JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise SpecError(f"{path} must be finite")


def _validate_json_value(value: Any, path: str) -> None:
    try:
        encoded = json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SpecError(f"{path} must be finite JSON data") from exc
    if len(encoded) > 100_000:
        raise SpecError(f"{path} exceeds 100,000 encoded characters")


def _validate_parameter(parameter: Any, index: int) -> dict[str, Any]:
    path = f"tuning.allowed_parameters[{index}]"
    obj = _expect_object(parameter, path)
    _check_keys(obj, {"path", "values", "range", "baseline"}, path)
    parameter_path = _expect_nonempty_string(obj.get("path"), f"{path}.path")
    if not PARAMETER_PATH_RE.fullmatch(parameter_path):
        raise SpecError(f"{path}.path contains unsupported characters")

    has_values = "values" in obj
    has_range = "range" in obj
    if has_values == has_range:
        raise SpecError(f"{path} must contain exactly one of values or range")

    if has_values:
        values = obj["values"]
        if not isinstance(values, list) or not values:
            raise SpecError(f"{path}.values must be a non-empty array")
        if len(values) > 256:
            raise SpecError(f"{path}.values may contain at most 256 values")
        for value_index, value in enumerate(values):
            _validate_scalar(value, f"{path}.values[{value_index}]")
    else:
        range_obj = _expect_object(obj["range"], f"{path}.range")
        _check_keys(range_obj, {"min", "max", "step"}, f"{path}.range")
        minimum = _expect_number(range_obj.get("min"), f"{path}.range.min")
        maximum = _expect_number(range_obj.get("max"), f"{path}.range.max")
        step = _expect_number(range_obj.get("step"), f"{path}.range.step")
        if maximum < minimum:
            raise SpecError(f"{path}.range.max must be greater than or equal to min")
        if step <= 0:
            raise SpecError(f"{path}.range.step must be positive")
        estimated_values = math.floor((maximum - minimum) / step + 1e-12) + 1
        if estimated_values > 256:
            raise SpecError(f"{path}.range expands to more than 256 values")

    if "baseline" in obj:
        _validate_scalar(obj["baseline"], f"{path}.baseline")
    return obj


def _validate_objectives(
    value: Any,
    version: int = 5,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise SpecError("tuning.objectives must be a non-empty array")
    seen: set[str] = set()
    for index, item in enumerate(value):
        path = f"tuning.objectives[{index}]"
        obj = _expect_object(item, path)
        allowed = {"metric", "goal", "weight"}
        if version >= 6:
            allowed.add("minimum_improvement")
        _check_keys(obj, allowed, path)
        metric = _expect_nonempty_string(obj.get("metric"), f"{path}.metric")
        if metric in seen:
            raise SpecError(f"{path}.metric duplicates {metric}")
        seen.add(metric)
        if obj.get("goal") not in {"maximize", "minimize"}:
            raise SpecError(f"{path}.goal must be maximize or minimize")
        weight = _expect_number(obj.get("weight"), f"{path}.weight")
        if weight <= 0:
            raise SpecError(f"{path}.weight must be positive")
        if "minimum_improvement" in obj:
            improvement = _expect_number(
                obj["minimum_improvement"],
                f"{path}.minimum_improvement",
            )
            if improvement < 0:
                raise SpecError(
                    f"{path}.minimum_improvement must be non-negative"
                )
    return value


def _validate_constraints(
    value: Any,
    version: int = 5,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SpecError("tuning.constraints must be an array")
    for index, item in enumerate(value):
        path = f"tuning.constraints[{index}]"
        obj = _expect_object(item, path)
        allowed = {"metric", "op", "value"}
        if version >= 6:
            allowed.add("scope")
        _check_keys(obj, allowed, path)
        _expect_nonempty_string(obj.get("metric"), f"{path}.metric")
        if obj.get("op") not in {"<=", ">=", "<", ">"}:
            raise SpecError(f"{path}.op must be one of <=, >=, <, >")
        _expect_number(obj.get("value"), f"{path}.value")
        if "scope" in obj and obj["scope"] not in {"mean", "each_seed"}:
            raise SpecError(f"{path}.scope must be mean or each_seed")
    return value


def _validate_hardware_feedback_contract(
    value: Any,
    version: int,
    mode: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if version not in {5, 6}:
        raise SpecError(
            "hardware feedback authorization requires session version 5 or 6"
        )
    contract = _expect_object(value, "hardware_feedback")
    _check_keys(
        contract,
        {
            "enabled",
            "output_mode",
            "output_dir",
            "require_policy_manifest",
            "verify_artifact_hashes",
            "stop_on_safety_event",
            "require_new_session_approval",
        },
        "hardware_feedback",
    )
    enabled = _expect_bool(contract.get("enabled"), "hardware_feedback.enabled")
    output_mode = contract.get("output_mode")
    if output_mode not in HARDWARE_FEEDBACK_OUTPUT_MODES:
        raise SpecError(
            "hardware_feedback.output_mode must be proposal_only or "
            "prepare_authorized_draft"
        )
    output_dir = Path(
        _expect_nonempty_string(
            contract.get("output_dir"),
            "hardware_feedback.output_dir",
        )
    )
    if not output_dir.is_absolute():
        raise SpecError("hardware_feedback.output_dir must be an absolute path")
    required_true = {
        "require_policy_manifest":
            "hardware feedback must bind to an archived policy manifest",
        "verify_artifact_hashes":
            "hardware feedback must verify JIT and ONNX hashes",
        "stop_on_safety_event":
            "hardware feedback must stop on a safety event",
        "require_new_session_approval":
            "feedback-driven retuning requires a newly approved session",
    }
    for field, message in required_true.items():
        if not _expect_bool(
            contract.get(field),
            f"hardware_feedback.{field}",
        ):
            raise SpecError(message)
    if (
        enabled
        and output_mode == "prepare_authorized_draft"
        and mode != "tune"
    ):
        raise SpecError(
            "prepare_authorized_draft requires tune mode and existing "
            "authorized parameter domains"
        )
    return contract


def _validate_seed_array(
    value: Any,
    path: str,
    minimum_length: int = 1,
) -> list[int]:
    if not isinstance(value, list) or len(value) < minimum_length:
        raise SpecError(
            f"{path} must contain at least {minimum_length} seed(s)"
        )
    if len(value) > 16:
        raise SpecError(f"{path} may contain at most 16 seeds")
    for index, seed in enumerate(value):
        _expect_int(seed, f"{path}[{index}]", 0, 2**31 - 1)
    if len(value) != len(set(value)):
        raise SpecError(f"{path} must contain unique seeds")
    return value


def _validate_execution_template(
    value: Any,
    required_placeholders: set[str] | None = None,
) -> list[str]:
    argv = _validate_argv(value, "execution.run_command")
    placeholders: set[str] = set()
    for index, token in enumerate(argv):
        try:
            parsed = list(Formatter().parse(token))
        except ValueError as exc:
            raise SpecError(
                f"execution.run_command[{index}] has invalid braces: {exc}"
            ) from exc
        found = {
            field_name
            for _, field_name, _, _ in parsed
            if field_name is not None
        }
        unknown = sorted(found - EXECUTION_PLACEHOLDERS)
        if unknown:
            raise SpecError(
                "execution.run_command"
                f"[{index}] contains unknown placeholder(s): {', '.join(unknown)}"
            )
        if any(
            field_name is not None and (format_spec or conversion)
            for _, field_name, format_spec, conversion in parsed
        ):
            raise SpecError(
                f"execution.run_command[{index}] cannot format or convert placeholders"
            )
        placeholders.update(found)
    required = (
        REQUIRED_EXECUTION_PLACEHOLDERS
        if required_placeholders is None
        else required_placeholders
    )
    missing = sorted(required - placeholders)
    if missing:
        raise SpecError(
            "execution.run_command is missing required placeholder(s): "
            + ", ".join(missing)
        )
    return argv


def _validate_execution_contract(
    value: Any,
    version: int,
    mode: str,
    monitoring_gpu_index: int | None,
    algorithm: dict[str, Any] | None = None,
    parameter_paths: list[str] | None = None,
    required_metrics: set[str] | None = None,
    training_command: list[str] | None = None,
) -> dict[str, Any] | None:
    if value is None:
        if version == 6 and mode == "tune":
            raise SpecError("version-6 tune sessions require execution")
        return None
    if version != 6:
        raise SpecError("execution authorization requires session version 6")
    if mode != "tune":
        raise SpecError("execution authorization requires tune mode")
    execution = _expect_object(value, "execution")
    _check_keys(
        execution,
        {
            "enabled",
            "state_dir",
            "run_command",
            "gpu_index",
            "require_idle_gpu",
            "max_retries_per_run",
            "effective_config",
            "quality_rules",
            "nonfinite_action",
            "adapter",
            "resource_limits",
            "reproducibility",
        },
        "execution",
    )
    if not _expect_bool(execution.get("enabled"), "execution.enabled"):
        raise SpecError("version-6 tune execution must be enabled")
    state_dir = Path(
        _expect_nonempty_string(execution.get("state_dir"), "execution.state_dir")
    )
    if not state_dir.is_absolute():
        raise SpecError("execution.state_dir must be an absolute path")
    adapter_template_fields = {
        "adapter_contract_path",
        "effective_config_path",
        "log_path",
        "overrides_json",
        "result_path",
        "run_id",
        "summary_path",
        "terminal_path",
    }
    _validate_execution_template(
        execution.get("run_command"),
        (
            adapter_template_fields
            if execution.get("adapter") is not None
            else REQUIRED_EXECUTION_PLACEHOLDERS
        ),
    )
    gpu_index = _expect_int(
        execution.get("gpu_index"),
        "execution.gpu_index",
        0,
        1024,
    )
    if monitoring_gpu_index is not None and gpu_index != monitoring_gpu_index:
        raise SpecError(
            "execution.gpu_index must match monitoring.gpu_index"
        )
    if not _expect_bool(
        execution.get("require_idle_gpu"),
        "execution.require_idle_gpu",
    ):
        raise SpecError("version-6 execution must require an idle GPU")
    _expect_int(
        execution.get("max_retries_per_run"),
        "execution.max_retries_per_run",
        0,
        3,
    )
    effective = _expect_object(
        execution.get("effective_config"),
        "execution.effective_config",
    )
    _check_keys(
        effective,
        {
            "enabled",
            "baseline_path",
            "require_exact_override_match",
            "allow_baseline_bootstrap",
        },
        "execution.effective_config",
    )
    if not _expect_bool(
        effective.get("enabled"),
        "execution.effective_config.enabled",
    ):
        raise SpecError("version-6 execution requires effective config checks")
    baseline_path = Path(
        _expect_nonempty_string(
            effective.get("baseline_path"),
            "execution.effective_config.baseline_path",
        )
    )
    if not baseline_path.is_absolute():
        raise SpecError(
            "execution.effective_config.baseline_path must be absolute"
        )
    if not _expect_bool(
        effective.get("require_exact_override_match"),
        "execution.effective_config.require_exact_override_match",
    ):
        raise SpecError("effective config must require exact override matching")
    if "allow_baseline_bootstrap" in effective:
        _expect_bool(
            effective["allow_baseline_bootstrap"],
            "execution.effective_config.allow_baseline_bootstrap",
        )

    adapter_value = execution.get("adapter")
    limits_value = execution.get("resource_limits")
    if (adapter_value is None) != (limits_value is None):
        raise SpecError(
            "execution.adapter and execution.resource_limits must be provided together"
        )
    if adapter_value is not None:
        if (
            algorithm is None
            or parameter_paths is None
            or required_metrics is None
            or training_command is None
        ):
            raise SpecError("execution adapter validation requires tune fields")
        adapter = _expect_object(adapter_value, "execution.adapter")
        _check_keys(
            adapter,
            {
                "id",
                "parameter_cli_map",
                "runtime_config_paths",
                "summary_last",
                "require_checkpoint",
            },
            "execution.adapter",
        )
        if adapter.get("id") != "rsl-rl":
            raise SpecError("execution.adapter.id must be rsl-rl")
        if algorithm.get("backend") != "rsl_rl":
            raise SpecError("rsl-rl adapter requires algorithm.backend=rsl_rl")
        parameter_map = _expect_object(
            adapter.get("parameter_cli_map"),
            "execution.adapter.parameter_cli_map",
        )
        if set(parameter_map) != set(parameter_paths):
            raise SpecError(
                "execution.adapter.parameter_cli_map keys must exactly match "
                "tuning.allowed_parameters paths"
            )
        for path, cli_path_value in parameter_map.items():
            cli_path = _expect_nonempty_string(
                cli_path_value,
                f"execution.adapter.parameter_cli_map.{path}",
            )
            if not PARAMETER_PATH_RE.fullmatch(cli_path):
                raise SpecError(
                    f"execution adapter CLI path contains unsupported characters: {cli_path}"
                )
        runtime_paths = _expect_object(
            adapter.get("runtime_config_paths"),
            "execution.adapter.runtime_config_paths",
        )
        if not runtime_paths:
            raise SpecError(
                "execution.adapter.runtime_config_paths must not be empty"
            )
        for path, identity_value in runtime_paths.items():
            if not PARAMETER_PATH_RE.fullmatch(path):
                raise SpecError(
                    f"execution adapter runtime config path is invalid: {path}"
                )
            if identity_value not in {"seed", "run_id"}:
                raise SpecError(
                    "execution adapter runtime config values must be seed or run_id"
                )
            if path in parameter_map:
                raise SpecError(
                    "runtime config paths cannot also be tunable parameter paths"
                )
        _expect_int(
            adapter.get("summary_last"),
            "execution.adapter.summary_last",
            1,
            10000,
        )
        _expect_bool(
            adapter.get("require_checkpoint"),
            "execution.adapter.require_checkpoint",
        )
        placeholders = {
            field
            for token in execution["run_command"]
            for _, field, _, _ in Formatter().parse(token)
            if field is not None
        }
        adapter_required = {
            "adapter_contract_path",
            "log_path",
            "run_id",
            "terminal_path",
        }
        missing_adapter = sorted(adapter_required - placeholders)
        if missing_adapter:
            raise SpecError(
                "rsl-rl adapter command is missing placeholder(s): "
                + ", ".join(missing_adapter)
            )
        reserved_prefixes = (
            "--seed",
            "--run_name",
            "--run-name",
        )
        if any(
            token == prefix or token.startswith(prefix + "=")
            for token in training_command
            for prefix in reserved_prefixes
        ):
            raise SpecError(
                "adapter-managed training.command cannot set seed or run name"
            )

        limits = _expect_object(limits_value, "execution.resource_limits")
        _check_keys(
            limits,
            {
                "campaign_timeout_minutes",
                "min_free_disk_gb",
                "max_gpu_temperature_c",
                "stop_grace_seconds",
            },
            "execution.resource_limits",
        )
        _expect_int(
            limits.get("campaign_timeout_minutes"),
            "execution.resource_limits.campaign_timeout_minutes",
            1,
            10080,
        )
        min_disk = _expect_number(
            limits.get("min_free_disk_gb"),
            "execution.resource_limits.min_free_disk_gb",
        )
        if not 0 <= min_disk <= 100000:
            raise SpecError(
                "execution.resource_limits.min_free_disk_gb must be between 0 and 100000"
            )
        _expect_int(
            limits.get("max_gpu_temperature_c"),
            "execution.resource_limits.max_gpu_temperature_c",
            40,
            100,
        )
        _expect_int(
            limits.get("stop_grace_seconds"),
            "execution.resource_limits.stop_grace_seconds",
            1,
            600,
        )

    reproducibility_value = execution.get("reproducibility")
    if reproducibility_value is not None:
        reproducibility = _expect_object(
            reproducibility_value,
            "execution.reproducibility",
        )
        _check_keys(
            reproducibility,
            {
                "enabled",
                "capture_git_diff",
                "capture_gpu",
                "package_names",
                "input_paths",
            },
            "execution.reproducibility",
        )
        _expect_bool(
            reproducibility.get("enabled"),
            "execution.reproducibility.enabled",
        )
        _expect_bool(
            reproducibility.get("capture_git_diff"),
            "execution.reproducibility.capture_git_diff",
        )
        _expect_bool(
            reproducibility.get("capture_gpu"),
            "execution.reproducibility.capture_gpu",
        )
        package_names = reproducibility.get("package_names")
        if not isinstance(package_names, list) or len(package_names) > 32:
            raise SpecError(
                "execution.reproducibility.package_names must be an array "
                "with at most 32 entries"
            )
        for index, package_name in enumerate(package_names):
            package_name = _expect_nonempty_string(
                package_name,
                f"execution.reproducibility.package_names[{index}]",
            )
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", package_name):
                raise SpecError(
                    "execution reproducibility package name contains "
                    "unsupported characters"
                )
        if len(package_names) != len(set(package_names)):
            raise SpecError(
                "execution.reproducibility.package_names must be unique"
            )
        input_paths = reproducibility.get("input_paths")
        if not isinstance(input_paths, list) or len(input_paths) > 64:
            raise SpecError(
                "execution.reproducibility.input_paths must be an array "
                "with at most 64 entries"
            )
        for index, input_path_value in enumerate(input_paths):
            input_path = Path(
                _expect_nonempty_string(
                    input_path_value,
                    f"execution.reproducibility.input_paths[{index}]",
                )
            )
            if not input_path.is_absolute():
                raise SpecError(
                    "execution reproducibility input paths must be absolute"
                )
        if len(input_paths) != len(set(input_paths)):
            raise SpecError(
                "execution.reproducibility.input_paths must be unique"
            )

    rules = execution.get("quality_rules")
    if not isinstance(rules, list):
        raise SpecError("execution.quality_rules must be an array")
    if len(rules) > 64:
        raise SpecError("execution.quality_rules may contain at most 64 rules")
    seen_rule_ids: set[str] = set()
    for index, rule_value in enumerate(rules):
        path = f"execution.quality_rules[{index}]"
        rule = _expect_object(rule_value, path)
        _check_keys(
            rule,
            {
                "id",
                "metric",
                "op",
                "value",
                "consecutive_windows",
                "minimum_progress",
                "action",
            },
            path,
        )
        rule_id = _expect_nonempty_string(rule.get("id"), f"{path}.id")
        if not IDENTIFIER_RE.fullmatch(rule_id):
            raise SpecError(f"{path}.id contains unsupported characters")
        if rule_id in seen_rule_ids:
            raise SpecError(f"{path}.id duplicates {rule_id}")
        seen_rule_ids.add(rule_id)
        _expect_nonempty_string(rule.get("metric"), f"{path}.metric")
        if rule.get("op") not in {"<=", ">=", "<", ">"}:
            raise SpecError(f"{path}.op must be one of <=, >=, <, >")
        _expect_number(rule.get("value"), f"{path}.value")
        _expect_int(
            rule.get("consecutive_windows"),
            f"{path}.consecutive_windows",
            1,
            100,
        )
        if "minimum_progress" in rule:
            _expect_int(
                rule["minimum_progress"],
                f"{path}.minimum_progress",
                0,
                10**9,
            )
        if rule.get("action") not in {"mark_suspect", "stop_trial"}:
            raise SpecError(
                f"{path}.action must be mark_suspect or stop_trial"
            )
    if execution.get("nonfinite_action") != "stop_trial":
        raise SpecError("execution.nonfinite_action must be stop_trial")
    return execution


def _validate_evaluation(
    value: Any,
    profile: dict[str, Any],
    mode: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    evaluation = _expect_object(value, "evaluation")
    _check_keys(
        evaluation,
        {
            "enabled",
            "require_for_final_selection",
            "artifacts",
            "scenarios",
            "gates",
            "parity",
            "visual_review",
            "output_dir",
            "gpu_index",
            "require_idle_gpu",
            "max_concurrent_runs",
            "run_timeout_minutes",
            "allow_reject_candidate",
            "allow_retune_on_failure",
            "execution",
        },
        "evaluation",
    )
    enabled = _expect_bool(evaluation.get("enabled"), "evaluation.enabled")
    require_final = _expect_bool(
        evaluation.get("require_for_final_selection"),
        "evaluation.require_for_final_selection",
    )
    if not enabled:
        if require_final:
            raise SpecError(
                "evaluation.require_for_final_selection cannot be true when evaluation is disabled"
            )
        return evaluation
    if not require_final:
        raise SpecError(
            "enabled evaluation must require policy evaluation before final selection"
        )
    if profile["is_generic"]:
        raise SpecError(
            "final policy evaluation requires a reviewed non-generic algorithm profile"
        )

    capabilities = profile["evaluation_capabilities"]
    supported = set(capabilities["supported_artifacts"])
    artifacts = evaluation.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SpecError("evaluation.artifacts must be a non-empty array")
    artifact_kinds: list[str] = []
    required_artifacts: set[str] = set()
    required_placeholders = {
        "artifact_path",
        "artifact_kind",
        "candidate_id",
        "artifact_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "command_schedule_json",
        "duration_steps",
        "gpu_index",
        "scenario_id",
        "scenario_overrides_json",
        "seed",
        "result_path",
        "require_idle_gpu_flag",
        "run_id",
        "video_path",
    }
    execution_enabled = evaluation.get("execution") is not None
    if execution_enabled:
        required_placeholders.add("executor_run_id")
    for index, item in enumerate(artifacts):
        path = f"evaluation.artifacts[{index}]"
        artifact = _expect_object(item, path)
        _check_keys(artifact, {"kind", "required", "command"}, path)
        kind = artifact.get("kind")
        if kind not in ARTIFACT_KINDS:
            raise SpecError(
                f"{path}.kind must be one of {', '.join(sorted(ARTIFACT_KINDS))}"
            )
        if kind not in supported:
            raise SpecError(
                f"{path}.kind {kind} is not supported by profile {profile['id']}"
            )
        artifact_kinds.append(kind)
        if _expect_bool(artifact.get("required"), f"{path}.required"):
            required_artifacts.add(kind)
        command = _validate_argv(artifact.get("command"), f"{path}.command")
        placeholders: set[str] = set()
        for command_index, token in enumerate(command):
            token_placeholders = set(COMMAND_PLACEHOLDER_RE.findall(token))
            unsupported = sorted(
                token_placeholders - ALLOWED_COMMAND_PLACEHOLDERS
            )
            if unsupported:
                raise SpecError(
                    f"{path}.command[{command_index}] contains unsupported "
                    f"placeholder(s): {', '.join(unsupported)}"
                )
            placeholders.update(token_placeholders)
        missing_placeholders = sorted(required_placeholders - placeholders)
        if missing_placeholders:
            raise SpecError(
                f"{path}.command is missing required placeholder(s): "
                f"{', '.join(missing_placeholders)}"
            )
        if execution_enabled and "{executor_run_id}" not in command:
            raise SpecError(
                f"{path}.command must pass {{executor_run_id}} as a "
                "standalone argv token"
            )
    if len(artifact_kinds) != len(set(artifact_kinds)):
        raise SpecError("evaluation.artifacts contains duplicate kinds")
    if "native" not in required_artifacts:
        raise SpecError("evaluation must require the native policy artifact")
    if {"jit", "onnx"} & supported and not ({"jit", "onnx"} & required_artifacts):
        raise SpecError(
            "a profile with export support must require at least one of jit or onnx"
        )

    scenarios = evaluation.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise SpecError("evaluation.scenarios must be a non-empty array")
    scenario_ids: list[str] = []
    required_categories: set[str] = set()
    required_video_count = 0
    for index, item in enumerate(scenarios):
        path = f"evaluation.scenarios[{index}]"
        scenario = _expect_object(item, path)
        _check_keys(
            scenario,
            {
                "id",
                "category",
                "required",
                "seeds",
                "duration_steps",
                "overrides",
                "command_schedule",
                "video",
            },
            path,
        )
        scenario_id = _expect_nonempty_string(scenario.get("id"), f"{path}.id")
        if not IDENTIFIER_RE.fullmatch(scenario_id):
            raise SpecError(f"{path}.id contains unsupported characters")
        scenario_ids.append(scenario_id)
        category = scenario.get("category")
        if category not in SCENARIO_CATEGORIES:
            raise SpecError(
                f"{path}.category must be one of "
                f"{', '.join(sorted(SCENARIO_CATEGORIES))}"
            )
        required = _expect_bool(scenario.get("required"), f"{path}.required")
        if required:
            required_categories.add(category)
        seeds = scenario.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise SpecError(f"{path}.seeds must be a non-empty array")
        if len(seeds) > 16:
            raise SpecError(f"{path}.seeds may contain at most 16 seeds")
        for seed_index, seed in enumerate(seeds):
            _expect_int(seed, f"{path}.seeds[{seed_index}]", 0, 2**31 - 1)
        if len(seeds) != len(set(seeds)):
            raise SpecError(f"{path}.seeds must be unique")
        duration_steps = _expect_int(
            scenario.get("duration_steps"),
            f"{path}.duration_steps",
            1,
            10_000_000,
        )
        overrides = _expect_object(scenario.get("overrides"), f"{path}.overrides")
        for override_path, override_value in overrides.items():
            if (
                not isinstance(override_path, str)
                or not PARAMETER_PATH_RE.fullmatch(override_path)
            ):
                raise SpecError(
                    f"{path}.overrides contains an unsupported parameter path"
                )
            _validate_json_value(
                override_value,
                f"{path}.overrides.{override_path}",
            )
        command_schedule = scenario.get("command_schedule")
        if not isinstance(command_schedule, list):
            raise SpecError(f"{path}.command_schedule must be an array")
        previous_end = -1
        for segment_index, segment_value in enumerate(command_schedule):
            segment_path = f"{path}.command_schedule[{segment_index}]"
            segment = _expect_object(segment_value, segment_path)
            _check_keys(
                segment,
                {"start_step", "end_step", "command"},
                segment_path,
            )
            start_step = _expect_int(
                segment.get("start_step"),
                f"{segment_path}.start_step",
                0,
                duration_steps - 1,
            )
            end_step = _expect_int(
                segment.get("end_step"),
                f"{segment_path}.end_step",
                0,
                duration_steps - 1,
            )
            if end_step < start_step:
                raise SpecError(
                    f"{segment_path}.end_step must not precede start_step"
                )
            if start_step != previous_end + 1:
                raise SpecError(
                    f"{path}.command_schedule must be ordered and contiguous"
                )
            command = segment.get("command")
            if not isinstance(command, list) or len(command) != 3:
                raise SpecError(
                    f"{segment_path}.command must contain [vx, vy, yaw_rate]"
                )
            for command_index, component in enumerate(command):
                _expect_number(
                    component,
                    f"{segment_path}.command[{command_index}]",
                )
            previous_end = end_step
        if command_schedule and previous_end != duration_steps - 1:
            raise SpecError(
                f"{path}.command_schedule must cover every evaluation step"
            )
        if category == "command" and not command_schedule:
            raise SpecError(
                f"{path}.command_schedule is required for command scenarios"
            )
        video = _expect_bool(scenario.get("video"), f"{path}.video")
        if required and video:
            required_video_count += len(seeds) * len(required_artifacts)
    if len(scenario_ids) != len(set(scenario_ids)):
        raise SpecError("evaluation.scenarios contains duplicate IDs")
    if "nominal" not in required_categories:
        raise SpecError("evaluation must contain a required nominal scenario")
    if not (required_categories - {"nominal"}):
        raise SpecError(
            "evaluation must contain at least one required non-nominal stress scenario"
        )

    gates = evaluation.get("gates")
    if not isinstance(gates, list) or not gates:
        raise SpecError("evaluation.gates must be a non-empty array")
    for index, item in enumerate(gates):
        path = f"evaluation.gates[{index}]"
        gate = _expect_object(item, path)
        _check_keys(
            gate,
            {"metric", "op", "value", "aggregation", "artifacts", "scenarios"},
            path,
        )
        _expect_nonempty_string(gate.get("metric"), f"{path}.metric")
        if gate.get("op") not in {"<=", ">=", "<", ">"}:
            raise SpecError(f"{path}.op must be one of <=, >=, <, >")
        _expect_number(gate.get("value"), f"{path}.value")
        if gate.get("aggregation") not in AGGREGATIONS:
            raise SpecError(
                f"{path}.aggregation must be one of "
                f"{', '.join(sorted(AGGREGATIONS))}"
            )
        for selector_name, known_values in (
            ("artifacts", set(artifact_kinds)),
            ("scenarios", set(scenario_ids)),
        ):
            selectors = gate.get(selector_name)
            if not isinstance(selectors, list) or not selectors:
                raise SpecError(f"{path}.{selector_name} must be a non-empty array")
            for selector_index, selector in enumerate(selectors):
                _expect_nonempty_string(
                    selector,
                    f"{path}.{selector_name}[{selector_index}]",
                )
                if selector != "*" and selector not in known_values:
                    raise SpecError(
                        f"{path}.{selector_name} references unknown value {selector}"
                    )

    parity = _expect_object(evaluation.get("parity"), "evaluation.parity")
    _check_keys(
        parity,
        {
            "required",
            "reference_artifact",
            "max_abs_action_error",
            "closed_loop_metrics",
        },
        "evaluation.parity",
    )
    parity_required = _expect_bool(
        parity.get("required"), "evaluation.parity.required"
    )
    reference = parity.get("reference_artifact")
    if reference not in artifact_kinds:
        raise SpecError(
            "evaluation.parity.reference_artifact must name a selected artifact"
        )
    parity_limit = _expect_number(
        parity.get("max_abs_action_error"),
        "evaluation.parity.max_abs_action_error",
    )
    if parity_limit < 0:
        raise SpecError(
            "evaluation.parity.max_abs_action_error must be non-negative"
        )
    if required_artifacts - {reference} and not parity_required:
        raise SpecError(
            "evaluation.parity.required must be true when multiple artifacts are required"
        )
    closed_loop_metrics = parity.get("closed_loop_metrics", [])
    if not isinstance(closed_loop_metrics, list):
        raise SpecError(
            "evaluation.parity.closed_loop_metrics must be an array"
        )
    closed_loop_names: list[str] = []
    for index, metric_value in enumerate(closed_loop_metrics):
        path = f"evaluation.parity.closed_loop_metrics[{index}]"
        metric = _expect_object(metric_value, path)
        _check_keys(
            metric,
            {"metric", "max_abs_delta", "aggregation"},
            path,
        )
        metric_name = _expect_nonempty_string(
            metric.get("metric"),
            f"{path}.metric",
        )
        closed_loop_names.append(metric_name)
        maximum_delta = _expect_number(
            metric.get("max_abs_delta"),
            f"{path}.max_abs_delta",
        )
        if maximum_delta < 0:
            raise SpecError(f"{path}.max_abs_delta must be non-negative")
        if metric.get("aggregation") not in {"max", "mean"}:
            raise SpecError(f"{path}.aggregation must be max or mean")
    if len(closed_loop_names) != len(set(closed_loop_names)):
        raise SpecError(
            "evaluation.parity.closed_loop_metrics contains duplicate metrics"
        )
    if (
        execution_enabled
        and required_artifacts - {reference}
        and not closed_loop_metrics
    ):
        raise SpecError(
            "automated deployment-artifact evaluation requires at least one "
            "closed-loop parity metric"
        )

    visual = _expect_object(
        evaluation.get("visual_review"), "evaluation.visual_review"
    )
    _check_keys(
        visual,
        {"required", "minimum_reviewed_videos", "require_notes"},
        "evaluation.visual_review",
    )
    visual_required = _expect_bool(
        visual.get("required"), "evaluation.visual_review.required"
    )
    minimum_videos = _expect_int(
        visual.get("minimum_reviewed_videos"),
        "evaluation.visual_review.minimum_reviewed_videos",
        1,
        100_000,
    )
    _expect_bool(
        visual.get("require_notes"), "evaluation.visual_review.require_notes"
    )
    if not visual_required:
        raise SpecError("final policy evaluation requires visual review")
    if required_video_count == 0:
        raise SpecError(
            "at least one required evaluation scenario must record video"
        )
    if minimum_videos > required_video_count:
        raise SpecError(
            "evaluation.visual_review.minimum_reviewed_videos exceeds the "
            "number of required videos"
        )

    output_dir = Path(
        _expect_nonempty_string(
            evaluation.get("output_dir"), "evaluation.output_dir"
        )
    )
    if not output_dir.is_absolute():
        raise SpecError("evaluation.output_dir must be an absolute path")
    _expect_int(
        evaluation.get("gpu_index"),
        "evaluation.gpu_index",
        0,
        1024,
    )
    if not _expect_bool(
        evaluation.get("require_idle_gpu"),
        "evaluation.require_idle_gpu",
    ):
        raise SpecError("final policy evaluation requires an idle GPU")
    _expect_int(
        evaluation.get("max_concurrent_runs"),
        "evaluation.max_concurrent_runs",
        1,
        8,
    )
    _expect_int(
        evaluation.get("run_timeout_minutes"),
        "evaluation.run_timeout_minutes",
        1,
        10080,
    )
    allow_reject = _expect_bool(
        evaluation.get("allow_reject_candidate"),
        "evaluation.allow_reject_candidate",
    )
    if require_final and not allow_reject:
        raise SpecError(
            "final selection requires evaluation.allow_reject_candidate=true"
        )
    allow_retune = _expect_bool(
        evaluation.get("allow_retune_on_failure"),
        "evaluation.allow_retune_on_failure",
    )
    if allow_retune and mode != "tune":
        raise SpecError(
            "evaluation.allow_retune_on_failure requires tune mode"
        )
    execution_value = evaluation.get("execution")
    if execution_value is not None:
        execution = _expect_object(execution_value, "evaluation.execution")
        _check_keys(
            execution,
            {
                "state_dir",
                "max_retries_per_run",
                "stop_grace_seconds",
                "min_free_disk_gb",
                "max_gpu_temperature_c",
                "minimum_video_bytes",
            },
            "evaluation.execution",
        )
        state_dir = Path(
            _expect_nonempty_string(
                execution.get("state_dir"),
                "evaluation.execution.state_dir",
            )
        )
        if not state_dir.is_absolute():
            raise SpecError(
                "evaluation.execution.state_dir must be an absolute path"
            )
        try:
            state_dir.relative_to(output_dir)
        except ValueError as exc:
            raise SpecError(
                "evaluation.execution.state_dir must be inside evaluation.output_dir"
            ) from exc
        _expect_int(
            execution.get("max_retries_per_run"),
            "evaluation.execution.max_retries_per_run",
            0,
            10,
        )
        _expect_int(
            execution.get("stop_grace_seconds"),
            "evaluation.execution.stop_grace_seconds",
            1,
            300,
        )
        minimum_free_disk = _expect_number(
            execution.get("min_free_disk_gb"),
            "evaluation.execution.min_free_disk_gb",
        )
        if minimum_free_disk <= 0:
            raise SpecError(
                "evaluation.execution.min_free_disk_gb must be positive"
            )
        _expect_int(
            execution.get("max_gpu_temperature_c"),
            "evaluation.execution.max_gpu_temperature_c",
            1,
            120,
        )
        _expect_int(
            execution.get("minimum_video_bytes"),
            "evaluation.execution.minimum_video_bytes",
            1,
            10_000_000_000,
        )
        if evaluation["max_concurrent_runs"] != 1:
            raise SpecError(
                "automated evaluation execution currently requires "
                "evaluation.max_concurrent_runs=1"
            )
    return evaluation


def _validate_archive(
    value: Any,
    version: int,
    mode: str,
    evaluation: dict[str, Any] | None,
    remove_created_temp_files: bool,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if version < 4:
        raise SpecError(
            "policy archive authorization requires session version 4, 5, or 6"
        )
    archive = _expect_object(value, "archive")
    _check_keys(
        archive,
        {
            "enabled",
            "copy_after_qualification",
            "storage_root",
            "collection",
            "directory_naming",
            "timezone",
            "required_artifacts",
            "require_clean_git_worktree",
            "write_manifest",
            "description_notes",
            "git_action",
        },
        "archive",
    )
    enabled = _expect_bool(archive.get("enabled"), "archive.enabled")
    copy_after = _expect_bool(
        archive.get("copy_after_qualification"),
        "archive.copy_after_qualification",
    )
    if not enabled:
        if copy_after:
            raise SpecError(
                "archive.copy_after_qualification cannot be true when archive is disabled"
            )
        return archive
    if mode != "tune":
        raise SpecError("policy archive requires tune mode")
    if not copy_after:
        raise SpecError(
            "enabled archive must set copy_after_qualification=true"
        )
    if not isinstance(evaluation, dict) or not evaluation.get("enabled"):
        raise SpecError("policy archive requires enabled final policy evaluation")

    storage_root = Path(
        _expect_nonempty_string(
            archive.get("storage_root"),
            "archive.storage_root",
        )
    )
    if not storage_root.is_absolute():
        raise SpecError("archive.storage_root must be an absolute path")

    collection = _expect_nonempty_string(
        archive.get("collection"),
        "archive.collection",
    )
    collection_path = PurePosixPath(collection)
    if (
        collection_path.is_absolute()
        or not collection_path.parts
        or any(
            part in {"", ".", ".."} or not IDENTIFIER_RE.fullmatch(part)
            for part in collection_path.parts
        )
    ):
        raise SpecError(
            "archive.collection must be a safe relative POSIX path"
        )
    if archive.get("directory_naming") != "local_timestamp_seconds":
        raise SpecError(
            "archive.directory_naming must be local_timestamp_seconds"
        )
    timezone_name = _expect_nonempty_string(
        archive.get("timezone"),
        "archive.timezone",
    )
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise SpecError("archive.timezone is not available") from exc

    required_artifacts = archive.get("required_artifacts")
    if (
        not isinstance(required_artifacts, list)
        or len(required_artifacts) != 2
        or set(required_artifacts) != {"jit", "onnx"}
    ):
        raise SpecError(
            "archive.required_artifacts must contain exactly jit and onnx"
        )
    evaluation_required = {
        artifact["kind"]
        for artifact in evaluation["artifacts"]
        if artifact["required"]
    }
    if not {"jit", "onnx"} <= evaluation_required:
        raise SpecError(
            "archive requires jit and onnx to be required evaluation artifacts"
        )
    if not _expect_bool(
        archive.get("require_clean_git_worktree"),
        "archive.require_clean_git_worktree",
    ):
        raise SpecError("policy archive requires a clean storage Git worktree")
    if not _expect_bool(
        archive.get("write_manifest"),
        "archive.write_manifest",
    ):
        raise SpecError("policy archive requires archive_manifest.json")
    notes = archive.get("description_notes")
    if not isinstance(notes, str):
        raise SpecError("archive.description_notes must be a string")
    if len(notes) > 4000:
        raise SpecError(
            "archive.description_notes may contain at most 4000 characters"
        )
    if archive.get("git_action") != "none":
        raise SpecError("archive.git_action must be none")
    if not remove_created_temp_files:
        raise SpecError(
            "policy archive requires cleanup.remove_created_temp_files=true"
        )
    return archive


def validate_spec(
    spec: Any,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Validate and return a session specification."""
    root = _expect_object(spec, "session")
    _check_keys(
        root,
        {
            "version",
            "mode",
            "algorithm",
            "training",
            "monitoring",
            "recovery",
            "tuning",
            "evaluation",
            "archive",
            "hardware_feedback",
            "execution",
            "cleanup",
        },
        "session",
    )

    version = root.get("version")
    if version not in {3, 4, 5, 6}:
        raise SpecError("version must be 3, 4, 5, or 6")
    mode = root.get("mode")
    if mode not in {"monitor", "tune"}:
        raise SpecError("mode must be monitor or tune")

    algorithm = _expect_object(root.get("algorithm"), "algorithm")
    _check_keys(
        algorithm,
        {
            "backend",
            "name",
            "runner_class",
            "profile_id",
            "profile_version",
            "profile_fingerprint",
            "unknown_algorithm_policy",
        },
        "algorithm",
    )
    backend = _expect_nonempty_string(algorithm.get("backend"), "algorithm.backend")
    algorithm_name = _expect_nonempty_string(algorithm.get("name"), "algorithm.name")
    runner_class = _expect_nonempty_string(
        algorithm.get("runner_class"), "algorithm.runner_class"
    )
    for field, value in (
        ("backend", backend),
        ("name", algorithm_name),
        ("runner_class", runner_class),
    ):
        if value == "auto":
            raise SpecError(f"algorithm.{field} must be resolved before approval")
    profile_id = _expect_nonempty_string(
        algorithm.get("profile_id"), "algorithm.profile_id"
    )
    profile_version = _expect_int(
        algorithm.get("profile_version"), "algorithm.profile_version", 1, 2**31 - 1
    )
    fingerprint = _expect_nonempty_string(
        algorithm.get("profile_fingerprint"), "algorithm.profile_fingerprint"
    )
    unknown_policy = algorithm.get("unknown_algorithm_policy")
    if unknown_policy not in {"reject", "runtime_generic", "propose_persistent"}:
        raise SpecError(
            "algorithm.unknown_algorithm_policy must be reject, runtime_generic, or propose_persistent"
        )
    try:
        registry = load_registry(registry_path)
        profile = resolve_profile(registry, profile_id)
    except ProfileError as exc:
        raise SpecError(str(exc)) from exc
    if profile_version != profile["profile_version"]:
        raise SpecError(
            f"algorithm.profile_version does not match registry version {profile['profile_version']}"
        )
    expected_fingerprint = profile_fingerprint(profile)
    if fingerprint != expected_fingerprint:
        raise SpecError(
            f"algorithm.profile_fingerprint must be {expected_fingerprint}"
        )
    if not profile_matches(profile, backend, algorithm_name, runner_class):
        raise SpecError("selected algorithm profile does not match the exact identity")
    if profile["is_generic"] and unknown_policy == "reject":
        raise SpecError("generic profiles require runtime_generic or propose_persistent")
    if mode == "tune" and profile["is_generic"]:
        raise SpecError(
            "tune mode requires a reviewed non-generic algorithm profile"
        )

    training = _expect_object(root.get("training"), "training")
    _check_keys(
        training,
        {
            "command",
            "resume_command",
            "cwd",
            "log_path",
            "run_id",
            "checkpoint_path",
            "source_git_commit",
            "source_git_dirty",
        },
        "training",
    )
    _validate_argv(training.get("command"), "training.command")
    _expect_nonempty_string(training.get("cwd"), "training.cwd")
    _expect_nonempty_string(training.get("log_path"), "training.log_path")
    training_run_id = _expect_nonempty_string(
        training.get("run_id"),
        "training.run_id",
    )
    if version == 6 and not IDENTIFIER_RE.fullmatch(training_run_id):
        raise SpecError(
            "version-6 training.run_id must contain only letters, digits, "
            "dot, underscore, or hyphen"
        )
    if "checkpoint_path" in training:
        _expect_nonempty_string(training["checkpoint_path"], "training.checkpoint_path")
    has_source_commit = "source_git_commit" in training
    has_source_dirty = "source_git_dirty" in training
    if has_source_commit != has_source_dirty:
        raise SpecError(
            "training.source_git_commit and source_git_dirty must appear together"
        )
    if has_source_commit:
        source_commit = _expect_nonempty_string(
            training["source_git_commit"],
            "training.source_git_commit",
        )
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise SpecError(
                "training.source_git_commit must be a full lowercase Git SHA"
            )
        _expect_bool(
            training["source_git_dirty"],
            "training.source_git_dirty",
        )

    monitoring = _expect_object(root.get("monitoring"), "monitoring")
    _check_keys(
        monitoring,
        {
            "check_interval_seconds",
            "stale_after_seconds",
            "pid",
            "gpu_index",
            "tensorboard_path",
            "expected_process_pattern",
            "low_gpu_utilization_percent",
        },
        "monitoring",
    )
    interval = _expect_int(monitoring.get("check_interval_seconds"), "monitoring.check_interval_seconds", 60, 3600)
    stale_after = _expect_int(monitoring.get("stale_after_seconds"), "monitoring.stale_after_seconds", 120, 86400)
    if stale_after < interval:
        raise SpecError("monitoring.stale_after_seconds must be at least the check interval")
    if monitoring.get("pid") is not None:
        _expect_int(monitoring["pid"], "monitoring.pid", 1, 2**31 - 1)
    monitoring_gpu_index = monitoring.get("gpu_index")
    if monitoring_gpu_index is not None:
        _expect_int(monitoring_gpu_index, "monitoring.gpu_index", 0, 1024)
    if monitoring.get("tensorboard_path") is not None:
        _expect_nonempty_string(
            monitoring["tensorboard_path"], "monitoring.tensorboard_path"
        )
    _expect_nonempty_string(monitoring.get("expected_process_pattern"), "monitoring.expected_process_pattern")
    low_gpu = _expect_number(
        monitoring.get("low_gpu_utilization_percent"), "monitoring.low_gpu_utilization_percent"
    )
    if not 0 <= low_gpu <= 100:
        raise SpecError("monitoring.low_gpu_utilization_percent must be between 0 and 100")

    recovery = _expect_object(root.get("recovery"), "recovery")
    _check_keys(recovery, {"enabled", "max_restarts", "cooldown_seconds"}, "recovery")
    recovery_enabled = _expect_bool(recovery.get("enabled"), "recovery.enabled")
    max_restarts = _expect_int(recovery.get("max_restarts"), "recovery.max_restarts", 0, 10)
    _expect_int(recovery.get("cooldown_seconds"), "recovery.cooldown_seconds", 0, 86400)
    if recovery_enabled:
        resume_command = _validate_argv(training.get("resume_command"), "training.resume_command")
        for required_arg in profile["resume_required_args"]:
            if required_arg not in resume_command:
                raise SpecError(
                    f"training.resume_command must contain profile-required argument {required_arg}"
                )
        if max_restarts == 0:
            raise SpecError("recovery.max_restarts must be positive when recovery is enabled")
    elif "resume_command" in training:
        _validate_argv(training["resume_command"], "training.resume_command")

    cleanup = _expect_object(root.get("cleanup"), "cleanup")
    _check_keys(cleanup, {"remove_created_temp_files"}, "cleanup")
    remove_created_temp_files = _expect_bool(
        cleanup.get("remove_created_temp_files"),
        "cleanup.remove_created_temp_files",
    )

    evaluation = _validate_evaluation(root.get("evaluation"), profile, mode)
    _validate_hardware_feedback_contract(
        root.get("hardware_feedback"),
        version,
        mode,
    )
    archive_contract = _validate_archive(
        root.get("archive"),
        version,
        mode,
        evaluation,
        remove_created_temp_files,
    )
    if (
        isinstance(archive_contract, dict)
        and archive_contract.get("enabled")
        and not has_source_commit
    ):
        raise SpecError(
            "policy archive requires the recorded training source Git state"
        )

    if mode == "monitor":
        _validate_execution_contract(
            root.get("execution"),
            version,
            mode,
            monitoring_gpu_index,
        )
        if root.get("tuning") is not None:
            raise SpecError("tuning must be null or omitted in monitor mode")
        return root

    tuning = _expect_object(root.get("tuning"), "tuning")
    _check_keys(
        tuning,
        {
            "allowed_parameters",
            "protected_parameters_unlocked",
            "max_trials",
            "seeds",
            "trial_timeout_minutes",
            "max_concurrent_trials",
            "mutation_scope",
            "objectives",
            "constraints",
            "seed_strategy",
            "ranking",
        },
        "tuning",
    )
    parameters = tuning.get("allowed_parameters")
    if not isinstance(parameters, list) or not parameters:
        raise SpecError("tuning.allowed_parameters must be a non-empty array")
    validated_parameters = [_validate_parameter(parameter, index) for index, parameter in enumerate(parameters)]
    parameter_paths = [parameter["path"] for parameter in validated_parameters]
    if len(set(parameter_paths)) != len(parameter_paths):
        raise SpecError("tuning.allowed_parameters contains duplicate paths")

    unlocked = tuning.get("protected_parameters_unlocked", [])
    if not isinstance(unlocked, list):
        raise SpecError("tuning.protected_parameters_unlocked must be an array")
    for index, path in enumerate(unlocked):
        _expect_nonempty_string(path, f"tuning.protected_parameters_unlocked[{index}]")
        if path not in parameter_paths:
            raise SpecError(f"protected unlock path is not an allowed parameter: {path}")
    for path in parameter_paths:
        if (
            any(
                re.search(pattern, path, re.IGNORECASE)
                for pattern in profile["protected_parameter_patterns"]
            )
            and path not in unlocked
        ):
            raise SpecError(f"protected parameter requires an exact unlock entry: {path}")

    _expect_int(tuning.get("max_trials"), "tuning.max_trials", 2, 64)
    seeds = _validate_seed_array(tuning.get("seeds"), "tuning.seeds")
    _expect_int(tuning.get("trial_timeout_minutes"), "tuning.trial_timeout_minutes", 1, 10080)
    max_concurrent_trials = _expect_int(
        tuning.get("max_concurrent_trials"),
        "tuning.max_concurrent_trials",
        1,
        8,
    )
    if version >= 6 and max_concurrent_trials != 1:
        raise SpecError(
            "version-6 execution requires tuning.max_concurrent_trials=1"
        )
    if tuning.get("mutation_scope") != "overrides_only":
        raise SpecError("tuning.mutation_scope must be overrides_only")
    _validate_objectives(tuning.get("objectives"), version)
    _validate_constraints(tuning.get("constraints", []), version)
    if version == 6:
        seed_strategy = _expect_object(
            tuning.get("seed_strategy"),
            "tuning.seed_strategy",
        )
        _check_keys(
            seed_strategy,
            {
                "screening_seeds",
                "confirmation_seeds",
                "confirmation_top_k",
            },
            "tuning.seed_strategy",
        )
        screening_seeds = _validate_seed_array(
            seed_strategy.get("screening_seeds"),
            "tuning.seed_strategy.screening_seeds",
        )
        confirmation_seeds = _validate_seed_array(
            seed_strategy.get("confirmation_seeds"),
            "tuning.seed_strategy.confirmation_seeds",
            minimum_length=2,
        )
        if not set(screening_seeds) <= set(confirmation_seeds):
            raise SpecError(
                "screening seeds must be a subset of confirmation seeds"
            )
        if set(screening_seeds) == set(confirmation_seeds):
            raise SpecError(
                "screening seeds must be a proper subset of confirmation seeds"
            )
        if confirmation_seeds != seeds:
            raise SpecError(
                "tuning.seeds must exactly equal confirmation_seeds"
            )
        top_k = _expect_int(
            seed_strategy.get("confirmation_top_k"),
            "tuning.seed_strategy.confirmation_top_k",
            1,
            63,
        )
        if top_k >= tuning["max_trials"]:
            raise SpecError(
                "confirmation_top_k must be less than tuning.max_trials"
            )
        ranking = _expect_object(tuning.get("ranking"), "tuning.ranking")
        _check_keys(
            ranking,
            {
                "require_paired_baseline",
                "constraint_scope",
                "minimum_final_training_seeds",
                "pareto_front_required",
            },
            "tuning.ranking",
        )
        for field in ("require_paired_baseline", "pareto_front_required"):
            if not _expect_bool(ranking.get(field), f"tuning.ranking.{field}"):
                raise SpecError(f"tuning.ranking.{field} must be true")
        if ranking.get("constraint_scope") != "each_seed":
            raise SpecError(
                "tuning.ranking.constraint_scope must be each_seed"
            )
        minimum_final = _expect_int(
            ranking.get("minimum_final_training_seeds"),
            "tuning.ranking.minimum_final_training_seeds",
            2,
            16,
        )
        if minimum_final > len(confirmation_seeds):
            raise SpecError(
                "minimum_final_training_seeds exceeds confirmation seed count"
            )
        for constraint in tuning.get("constraints", []):
            if constraint.get("scope", "each_seed") != "each_seed":
                raise SpecError(
                    "version-6 constraints must use scope=each_seed"
                )
    elif "seed_strategy" in tuning or "ranking" in tuning:
        raise SpecError(
            "seed_strategy and ranking require session version 6"
        )
    required_metrics = {
        objective["metric"] for objective in tuning["objectives"]
    } | {
        constraint["metric"] for constraint in tuning.get("constraints", [])
    }
    _validate_execution_contract(
        root.get("execution"),
        version,
        mode,
        monitoring_gpu_index,
        algorithm=algorithm,
        parameter_paths=parameter_paths,
        required_metrics=required_metrics,
        training_command=training["command"],
    )
    return root


def load_and_validate(
    path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate a JSON session specification."""
    spec_path = Path(path)
    try:
        data = json.loads(spec_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"session file does not exist: {spec_path}") from exc
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    return validate_spec(data, registry_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="Path to the session JSON document")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--print-normalized", action="store_true", help="Print the validated JSON")
    args = parser.parse_args()
    try:
        spec = load_and_validate(args.session, args.registry)
    except SpecError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 2
    print("VALID")
    if args.print_normalized:
        print(json.dumps(spec, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Validate and exclusively publish source-bound policy-evaluation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SCENARIO_FIELDS = {
    "scenario_id",
    "scenario_overrides",
    "command_schedule",
    "duration_steps",
    "num_envs",
    "seed",
}


class EvaluationEvidenceError(ValueError):
    """Raised when evaluation evidence is unsafe, conflicting, or incomplete."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise EvaluationEvidenceError(
            f"{name} must be a safe ASCII identifier (letters, digits, '.', '_', '-')"
        )
    return value


def _validate_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise EvaluationEvidenceError(f"{name} must be a lowercase SHA-256")
    return value


def _reject_unsafe_path(path: Path, *, label: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise EvaluationEvidenceError(
            f"{label} must be an absolute path without traversal"
        )
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise EvaluationEvidenceError(
                f"{label} contains a symlinked path component: {current}"
            )


def _require_regular_file(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, str]:
    _reject_unsafe_path(path, label=label)
    if not path.is_file():
        raise EvaluationEvidenceError(f"{label} is not an existing regular file")
    actual_sha256 = sha256_file(path)
    if expected_sha256 is not None:
        _validate_sha256(f"{label} SHA-256", expected_sha256)
        if actual_sha256 != expected_sha256:
            raise EvaluationEvidenceError(f"{label} SHA-256 mismatch")
    return {"path": str(path), "sha256": actual_sha256}


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationEvidenceError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationEvidenceError(f"{label} must be a JSON object")
    return value


def validate_scenario_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SCENARIO_FIELDS:
        raise EvaluationEvidenceError(
            "scenario contract must contain exactly scenario_id, scenario_overrides, "
            "command_schedule, duration_steps, num_envs, and seed"
        )
    _validate_identifier("scenario_id", value["scenario_id"])
    if not isinstance(value["scenario_overrides"], dict):
        raise EvaluationEvidenceError("scenario_overrides must be an object")
    if not isinstance(value["command_schedule"], list):
        raise EvaluationEvidenceError("command_schedule must be an array")
    for field in ("duration_steps", "num_envs"):
        field_value = value[field]
        if (
            isinstance(field_value, bool)
            or not isinstance(field_value, int)
            or field_value <= 0
        ):
            raise EvaluationEvidenceError(f"{field} must be a positive integer")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise EvaluationEvidenceError("seed must be an integer")

    previous_end = -1
    for index, segment in enumerate(value["command_schedule"]):
        if not isinstance(segment, dict) or set(segment) != {
            "start_step",
            "end_step",
            "command",
        }:
            raise EvaluationEvidenceError(
                f"command schedule segment {index} is invalid"
            )
        start = segment["start_step"]
        end = segment["end_step"]
        command = segment["command"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != previous_end + 1
            or end < start
            or end >= value["duration_steps"]
        ):
            raise EvaluationEvidenceError(
                "command schedule must be ordered, contiguous, and within duration"
            )
        if (
            not isinstance(command, list)
            or len(command) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for item in command
            )
        ):
            raise EvaluationEvidenceError(
                "command schedule values must be finite [vx, vy, yaw_rate]"
            )
        previous_end = end
    if value["command_schedule"] and previous_end != value["duration_steps"] - 1:
        raise EvaluationEvidenceError(
            "command schedule must cover every evaluation step"
        )
    try:
        canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationEvidenceError(f"scenario contract is not finite JSON: {exc}") from exc
    return value


def build_scenario_contract(
    *,
    scenario_id: str,
    scenario_overrides_json: str,
    command_schedule_json: str,
    duration_steps: int,
    num_envs: int,
    seed: int,
) -> dict[str, Any]:
    try:
        scenario_overrides = json.loads(scenario_overrides_json)
    except json.JSONDecodeError as exc:
        raise EvaluationEvidenceError(
            f"invalid scenario_overrides_json at column {exc.colno}: {exc.msg}"
        ) from exc
    try:
        command_schedule = json.loads(command_schedule_json)
    except json.JSONDecodeError as exc:
        raise EvaluationEvidenceError(
            f"invalid command_schedule_json at column {exc.colno}: {exc.msg}"
        ) from exc
    return validate_scenario_contract(
        {
            "scenario_id": scenario_id,
            "scenario_overrides": scenario_overrides,
            "command_schedule": command_schedule,
            "duration_steps": duration_steps,
            "num_envs": num_envs,
            "seed": seed,
        }
    )


def scenario_sha256(contract: dict[str, Any]) -> str:
    validate_scenario_contract(contract)
    return sha256_bytes(canonical_json(contract).encode("utf-8"))


def expected_evaluation_paths(
    repo_root: Path,
    *,
    task: str,
    run_id: str,
    evaluation_id: str,
) -> dict[str, Path]:
    _reject_unsafe_path(repo_root, label="repository root")
    _validate_identifier("task", task)
    _validate_identifier("run_id", run_id)
    _validate_identifier("evaluation_id", evaluation_id)
    evaluation_dir = (
        repo_root
        / "learnings"
        / "policy_tuning"
        / task
        / run_id
        / "evidence"
        / "play"
        / evaluation_id
    )
    return {
        "evaluation_dir": evaluation_dir,
        "result": evaluation_dir / "result.json",
        "telemetry": evaluation_dir / "telemetry.json",
        "video": evaluation_dir / "video.mp4",
    }


def _require_exact_path(actual: Path, expected: Path, *, label: str) -> None:
    _reject_unsafe_path(actual, label=label)
    if actual != expected:
        raise EvaluationEvidenceError(
            f"{label} is outside the current run/evaluation evidence layout"
        )


def _require_new_targets(paths: dict[str, Path]) -> None:
    for label in ("result", "telemetry", "video"):
        path = paths[label]
        _reject_unsafe_path(path, label=f"{label} target")
        if path.exists() or path.is_symlink():
            raise EvaluationEvidenceError(f"{label} target already exists: {path}")


def _load_bound_run_identity(
    path: Path,
    *,
    expected_file_sha256: str,
    repo_root: Path,
    task: str,
    run_id: str,
    seed: int,
    scenario_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    reference = _require_regular_file(
        path,
        label="run identity",
        expected_sha256=expected_file_sha256,
    )
    expected_source_dir = (
        repo_root
        / "learnings"
        / "policy_tuning"
        / task
        / run_id
        / "evidence"
        / "source"
    )
    if path.parent != expected_source_dir or not re.fullmatch(
        r"identity-[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.json",
        path.name,
    ):
        raise EvaluationEvidenceError(
            "run identity is outside the current run evidence/source layout"
        )
    identity = _load_json_object(path, label="run identity")
    if identity.get("version") != 1:
        raise EvaluationEvidenceError("run identity version must be 1")
    payload = {key: value for key, value in identity.items() if key != "identity_sha256"}
    internal_sha256 = sha256_bytes(canonical_json(payload).encode("utf-8"))
    if identity.get("identity_sha256") != internal_sha256:
        raise EvaluationEvidenceError("run identity internal SHA-256 mismatch")
    for field, expected in (("task", task), ("run_id", run_id), ("seed", seed)):
        if identity.get(field) != expected:
            raise EvaluationEvidenceError(f"run identity {field} mismatch")
    source = identity.get("source")
    if not isinstance(source, dict) or source.get("repository_root") != str(repo_root):
        raise EvaluationEvidenceError("run identity repository root mismatch")
    evaluation = identity.get("evaluation_scenario")
    if not isinstance(evaluation, dict):
        raise EvaluationEvidenceError("run identity evaluation_scenario is missing")
    identity_contract = validate_scenario_contract(evaluation.get("contract"))
    expected_scenario_sha256 = scenario_sha256(identity_contract)
    if evaluation.get("sha256") != expected_scenario_sha256:
        raise EvaluationEvidenceError("run identity scenario SHA-256 mismatch")
    if identity_contract != scenario_contract:
        raise EvaluationEvidenceError("run identity scenario contract mismatch")
    return identity, {
        "path": reference["path"],
        "file_sha256": reference["sha256"],
        "identity_sha256": internal_sha256,
    }


@dataclass(frozen=True)
class EvaluationPlan:
    repo_root: Path
    task: str
    run_id: str
    evaluation_id: str
    result_path: Path
    telemetry_path: Path | None
    video_path: Path | None
    canonical_paths: dict[str, Path]
    checkpoint: dict[str, str]
    artifact: dict[str, str]
    artifact_kind: str
    run_identity: dict[str, str]
    scenario_contract: dict[str, Any]
    scenario_sha256: str


def preflight_evaluation(
    *,
    repo_root: Path,
    task: str,
    run_id: str,
    evaluation_id: str,
    result_path: Path,
    telemetry_path: Path | None,
    video_path: Path | None,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    artifact_kind: str,
    artifact_path: Path,
    artifact_sha256: str,
    run_identity_path: Path,
    run_identity_file_sha256: str,
    scenario_contract: dict[str, Any],
) -> EvaluationPlan:
    if artifact_kind not in {"native", "jit", "onnx"}:
        raise EvaluationEvidenceError("artifact_kind must be native, jit, or onnx")
    repo_root = repo_root.absolute()
    canonical_paths = expected_evaluation_paths(
        repo_root,
        task=task,
        run_id=run_id,
        evaluation_id=evaluation_id,
    )
    evaluation_dir = canonical_paths["evaluation_dir"]
    _reject_unsafe_path(evaluation_dir, label="evaluation directory")
    if not evaluation_dir.is_dir():
        raise EvaluationEvidenceError(
            "evaluation directory must be prepared by prepare_evidence_layout.py"
        )
    _require_exact_path(result_path, canonical_paths["result"], label="result_path")
    if telemetry_path is not None:
        _require_exact_path(
            telemetry_path,
            canonical_paths["telemetry"],
            label="telemetry_path",
        )
    if video_path is not None:
        _require_exact_path(video_path, canonical_paths["video"], label="video_path")
    _require_new_targets(canonical_paths)

    checkpoint = _require_regular_file(
        checkpoint_path,
        label="checkpoint",
        expected_sha256=checkpoint_sha256,
    )
    artifact = _require_regular_file(
        artifact_path,
        label="artifact",
        expected_sha256=artifact_sha256,
    )
    scenario_contract = validate_scenario_contract(scenario_contract)
    _, run_identity = _load_bound_run_identity(
        run_identity_path,
        expected_file_sha256=run_identity_file_sha256,
        repo_root=repo_root,
        task=task,
        run_id=run_id,
        seed=scenario_contract["seed"],
        scenario_contract=scenario_contract,
    )
    return EvaluationPlan(
        repo_root=repo_root,
        task=task,
        run_id=run_id,
        evaluation_id=evaluation_id,
        result_path=result_path,
        telemetry_path=telemetry_path,
        video_path=video_path,
        canonical_paths=canonical_paths,
        checkpoint=checkpoint,
        artifact=artifact,
        artifact_kind=artifact_kind,
        run_identity=run_identity,
        scenario_contract=scenario_contract,
        scenario_sha256=scenario_sha256(scenario_contract),
    )


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


class EvaluationPublisher:
    """Claim one evaluation directory and publish result last without overwrite."""

    def __init__(self, plan: EvaluationPlan) -> None:
        self.plan = plan
        self.evaluation_dir = plan.canonical_paths["evaluation_dir"]
        self.claim_path = self.evaluation_dir / ".publish-claim"
        self.attempt_dir = self.evaluation_dir / ".attempt"
        self.video_work_path = self.attempt_dir / "video.mp4"
        self._entered = False
        self._published = False
        self._claim_identity: tuple[int, int] | None = None
        self._attempt_identity: tuple[int, int] | None = None

    def __enter__(self) -> "EvaluationPublisher":
        _require_new_targets(self.plan.canonical_paths)
        try:
            with self.claim_path.open("x", encoding="utf-8") as stream:
                stream.write(self.plan.evaluation_id + "\n")
        except FileExistsError as exc:
            raise EvaluationEvidenceError(
                f"evaluation publication is already claimed: {self.claim_path}"
            ) from exc
        claim_stat = self.claim_path.stat(follow_symlinks=False)
        self._claim_identity = (claim_stat.st_dev, claim_stat.st_ino)
        try:
            self.attempt_dir.mkdir()
        except OSError as exc:
            if self._still_owned(self.claim_path, self._claim_identity):
                self.claim_path.unlink()
            self._claim_identity = None
            raise EvaluationEvidenceError(
                f"cannot create exclusive evaluation attempt: {exc}"
            ) from exc
        attempt_stat = self.attempt_dir.stat(follow_symlinks=False)
        self._attempt_identity = (attempt_stat.st_dev, attempt_stat.st_ino)
        self._entered = True
        return self

    @staticmethod
    def _still_owned(path: Path, identity: tuple[int, int] | None) -> bool:
        if identity is None:
            return False
        try:
            stat_result = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        return (stat_result.st_dev, stat_result.st_ino) == identity

    def _cleanup_owned(self) -> None:
        if self._still_owned(self.attempt_dir, self._attempt_identity):
            shutil.rmtree(self.attempt_dir)
        if self._still_owned(self.claim_path, self._claim_identity):
            self.claim_path.unlink()
        self._attempt_identity = None
        self._claim_identity = None

    def close(self) -> None:
        """Release only this publisher's temporary claim and attempt data."""
        if self._entered:
            self._cleanup_owned()
            self._entered = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def publish(
        self,
        result: dict[str, Any],
        *,
        telemetry: dict[str, Any] | None,
        video_source: Path | None,
    ) -> dict[str, Any]:
        if not self._entered or self._published:
            raise EvaluationEvidenceError("evaluation publisher is not active")
        if (telemetry is None) != (self.plan.telemetry_path is None):
            raise EvaluationEvidenceError("telemetry payload does not match telemetry_path")
        if (video_source is None) != (self.plan.video_path is None):
            raise EvaluationEvidenceError("video payload does not match video_path")
        _require_new_targets(self.plan.canonical_paths)

        telemetry_work = self.attempt_dir / "telemetry.json"
        result_work = self.attempt_dir / "result.json"
        telemetry_reference = None
        if telemetry is not None:
            _write_new_json(telemetry_work, telemetry)
            telemetry_reference = {
                "path": str(self.plan.telemetry_path),
                "sha256": sha256_file(telemetry_work),
            }
        video_reference = None
        if video_source is not None:
            if video_source != self.video_work_path:
                raise EvaluationEvidenceError("video source is outside the evaluation attempt")
            video_work = _require_regular_file(video_source, label="attempt video")
            video_reference = {
                "path": str(self.plan.video_path),
                "sha256": video_work["sha256"],
            }

        published_result = dict(result)
        published_result["result_path"] = str(self.plan.result_path)
        published_result["outputs"] = {
            "telemetry": telemetry_reference,
            "video": video_reference,
        }
        _write_new_json(result_work, published_result)

        sources_and_targets: list[tuple[Path, Path]] = []
        if video_source is not None:
            assert self.plan.video_path is not None
            sources_and_targets.append((video_source, self.plan.video_path))
        if telemetry is not None:
            assert self.plan.telemetry_path is not None
            sources_and_targets.append((telemetry_work, self.plan.telemetry_path))
        created_targets: list[Path] = []
        try:
            for source, target in sources_and_targets:
                os.link(source, target, follow_symlinks=False)
                created_targets.append(target)
            validate_evaluation_bundle(
                result_work,
                canonical_result_path=self.plan.result_path,
            )
            os.link(result_work, self.plan.result_path, follow_symlinks=False)
            created_targets.append(self.plan.result_path)
            validate_evaluation_bundle(self.plan.result_path)
        except (OSError, EvaluationEvidenceError) as exc:
            for target in reversed(created_targets):
                target.unlink(missing_ok=True)
            raise EvaluationEvidenceError(
                f"exclusive evaluation publication failed: {exc}"
            ) from exc
        self._published = True
        return published_result


def close_evaluation_resources(
    publisher: EvaluationPublisher,
    simulation_app: Any,
) -> None:
    """Release publication state before shutting down Isaac Sim."""
    try:
        publisher.close()
    finally:
        simulation_app.close()


def _validate_reference(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise EvaluationEvidenceError(f"{label} reference is invalid")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise EvaluationEvidenceError(f"{label} path is invalid")
    reference = _require_regular_file(
        Path(path_value),
        label=label,
        expected_sha256=value.get("sha256"),
    )
    return reference


def validate_evaluation_bundle(
    result_path: Path,
    *,
    canonical_result_path: Path | None = None,
) -> dict[str, Any]:
    """Revalidate a final bundle or a private result before final publication."""
    result_reference = _require_regular_file(result_path, label="evaluation result")
    result = _load_json_object(result_path, label="evaluation result")
    if result.get("version") != 2 or result.get("status") != "completed":
        raise EvaluationEvidenceError("evaluation result must be completed version 2")
    evaluation = result.get("evaluation")
    inputs = result.get("inputs")
    outputs = result.get("outputs")
    if not isinstance(evaluation, dict) or not isinstance(inputs, dict):
        raise EvaluationEvidenceError("evaluation result binding is incomplete")
    if not isinstance(outputs, dict) or set(outputs) != {"telemetry", "video"}:
        raise EvaluationEvidenceError("evaluation result outputs are incomplete")
    task = _validate_identifier("result task", evaluation.get("task"))
    run_id = _validate_identifier("result run_id", evaluation.get("run_id"))
    evaluation_id = _validate_identifier(
        "result evaluation_id", evaluation.get("evaluation_id")
    )
    _validate_identifier("result candidate_id", evaluation.get("candidate_id"))
    _validate_identifier("result runner", evaluation.get("runner"))
    resource_mode = inputs.get("resource_mode")
    if not isinstance(resource_mode, dict):
        raise EvaluationEvidenceError("result resource mode is missing")
    for field in (
        "training_overlap",
        "idle_gpu_required",
        "video_requested",
        "telemetry_requested",
    ):
        if not isinstance(resource_mode.get(field), bool):
            raise EvaluationEvidenceError(f"result resource mode {field} is invalid")
    run_identity = inputs.get("run_identity")
    if not isinstance(run_identity, dict):
        raise EvaluationEvidenceError("result run identity reference is missing")
    run_identity_path = run_identity.get("path")
    if not isinstance(run_identity_path, str):
        raise EvaluationEvidenceError("result run identity path is invalid")
    identity_document = _load_json_object(
        Path(run_identity_path), label="result run identity"
    )
    source = identity_document.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("repository_root"), str):
        raise EvaluationEvidenceError("result run identity repository root is missing")
    repo_root = Path(source["repository_root"])
    expected_paths = expected_evaluation_paths(
        repo_root,
        task=task,
        run_id=run_id,
        evaluation_id=evaluation_id,
    )
    declared_result_path = canonical_result_path or result_path
    _require_exact_path(
        declared_result_path,
        expected_paths["result"],
        label="result_path",
    )
    if result.get("result_path") != str(declared_result_path):
        raise EvaluationEvidenceError("embedded result_path mismatch")

    scenario = inputs.get("scenario")
    if not isinstance(scenario, dict) or set(scenario) != {"contract", "sha256"}:
        raise EvaluationEvidenceError("result scenario binding is invalid")
    scenario_contract = validate_scenario_contract(scenario["contract"])
    if scenario["sha256"] != scenario_sha256(scenario_contract):
        raise EvaluationEvidenceError("result scenario SHA-256 mismatch")
    loaded_identity, bound_identity = _load_bound_run_identity(
        Path(run_identity_path),
        expected_file_sha256=run_identity.get("file_sha256"),
        repo_root=repo_root,
        task=task,
        run_id=run_id,
        seed=scenario_contract["seed"],
        scenario_contract=scenario_contract,
    )
    if run_identity.get("identity_sha256") != bound_identity["identity_sha256"]:
        raise EvaluationEvidenceError("result run identity SHA-256 mismatch")
    del loaded_identity

    checkpoint = _validate_reference(inputs.get("checkpoint"), label="result checkpoint")
    artifact = inputs.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"kind", "path", "sha256"}:
        raise EvaluationEvidenceError("result artifact reference is invalid")
    artifact_reference = _validate_reference(
        {"path": artifact["path"], "sha256": artifact["sha256"]},
        label="result artifact",
    )

    output_references: dict[str, dict[str, str] | None] = {}
    for label in ("telemetry", "video"):
        value = outputs[label]
        if value is None:
            output_references[label] = None
            continue
        reference = _validate_reference(value, label=f"result {label}")
        _require_exact_path(
            Path(reference["path"]),
            expected_paths[label],
            label=f"result {label}",
        )
        output_references[label] = reference
    if resource_mode["telemetry_requested"] != (
        output_references["telemetry"] is not None
    ):
        raise EvaluationEvidenceError("telemetry resource mode/output mismatch")
    if resource_mode["video_requested"] != (
        output_references["video"] is not None
    ):
        raise EvaluationEvidenceError("video resource mode/output mismatch")
    telemetry_reference = output_references["telemetry"]
    if telemetry_reference is not None:
        telemetry = _load_json_object(
            Path(telemetry_reference["path"]), label="evaluation telemetry"
        )
        if telemetry.get("evaluation") != evaluation:
            raise EvaluationEvidenceError("telemetry evaluation binding mismatch")
        if telemetry.get("inputs") != inputs:
            raise EvaluationEvidenceError("telemetry input binding mismatch")

    return {
        "status": "valid",
        "result": result_reference,
        "checkpoint": checkpoint,
        "artifact": artifact_reference,
        "run_identity": bound_identity,
        "scenario_sha256": scenario["sha256"],
        "outputs": output_references,
    }

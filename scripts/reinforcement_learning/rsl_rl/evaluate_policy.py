#!/usr/bin/env python3
"""Evaluate native, JIT, or ONNX RSL-RL policies in a closed IsaacLab loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from policy_evaluation_telemetry import (
    TELEMETRY_SIGNALS,
    SignalLedger,
    metric_availability_report,
    record_complete_metric,
    required_signals_for_runner,
    telemetry_report,
)

ORIGINAL_ARGV = tuple(sys.argv)

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Exact registered training task")
parser.add_argument(
    "--agent",
    default="rsl_rl_cfg_entry_point",
    help="Agent configuration entry point",
)
parser.add_argument("--artifact_kind", choices=("native", "jit", "onnx"), required=True)
parser.add_argument("--artifact_path", required=True)
parser.add_argument("--artifact_sha256", required=True)
parser.add_argument("--candidate_id", required=True)
parser.add_argument("--checkpoint_sha256", required=True)
parser.add_argument("--scenario_id", required=True)
parser.add_argument("--scenario_overrides_json", default="{}")
parser.add_argument("--command_schedule_json", default="[]")
parser.add_argument("--duration_steps", type=int, required=True)
parser.add_argument("--executor_run_id")
parser.add_argument("--result_path", required=True)
parser.add_argument("--run_id", required=True)
parser.add_argument("--video_path")
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--allow_training_overlap", action="store_true")
parser.add_argument("--follow_robot_camera", action="store_true")
parser.add_argument(
    "--follow_camera_offset_json",
    default="[3.0, 3.0, 2.0]",
    help="World-frame camera eye offset from the selected robot",
)
parser.add_argument("--telemetry_path")
parser.add_argument("--telemetry_env_index", type=int, default=0)
parser.add_argument("--telemetry_stride", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--require_idle_gpu", action="store_true")
cli_args.add_rsl_rl_args(parser)
if "-h" in ORIGINAL_ARGV or "--help" in ORIGINAL_ARGV:
    parser.print_help()
    raise SystemExit(0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = not args_cli.no_video


def _assert_gpu_idle(device: str | None) -> None:
    if not args_cli.require_idle_gpu or device is None or not device.startswith("cuda"):
        return
    gpu_index = device.split(":", 1)[1] if ":" in device else "0"
    check = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            gpu_index,
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            f"cannot verify that GPU {gpu_index} is idle: {check.stderr.strip()}"
        )
    active_pids = [
        line.strip()
        for line in check.stdout.splitlines()
        if line.strip() and line.strip().isdigit()
    ]
    if active_pids:
        raise RuntimeError(
            f"GPU {gpu_index} is not idle; active compute PID(s): "
            f"{', '.join(active_pids)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_artifact_hashes() -> None:
    checkpoint_path = Path(args_cli.checkpoint or "")
    artifact_path = Path(args_cli.artifact_path)
    for label, path, expected in (
        ("checkpoint", checkpoint_path, args_cli.checkpoint_sha256),
        ("artifact", artifact_path, args_cli.artifact_sha256),
    ):
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"{label} path is not an existing absolute file")
        if _sha256(path) != expected:
            raise RuntimeError(f"{label} SHA-256 does not match the approved plan")


def _assert_executor_identity() -> None:
    if (
        args_cli.executor_run_id is not None
        and args_cli.executor_run_id != args_cli.run_id
    ):
        raise RuntimeError(
            "--executor_run_id must exactly match the approved --run_id"
        )


def _assert_resource_mode() -> None:
    if args_cli.require_idle_gpu and args_cli.allow_training_overlap:
        raise RuntimeError(
            "--require_idle_gpu and --allow_training_overlap are mutually exclusive"
        )
    if args_cli.allow_training_overlap:
        if args_cli.num_envs > 4:
            raise RuntimeError("training-overlap evaluation permits at most 4 environments")
        if args_cli.duration_steps > 2000:
            raise RuntimeError("training-overlap evaluation permits at most 2000 steps")
    if args_cli.no_video and args_cli.video_path:
        raise RuntimeError("--video_path must be omitted with --no_video")
    if not args_cli.no_video and not args_cli.video_path:
        raise RuntimeError("--video_path is required unless --no_video is used")
    if args_cli.telemetry_stride <= 0:
        raise RuntimeError("--telemetry_stride must be positive")
    if not 0 <= args_cli.telemetry_env_index < args_cli.num_envs:
        raise RuntimeError("--telemetry_env_index must select an existing environment")


try:
    _assert_artifact_hashes()
    _assert_executor_identity()
    _assert_resource_mode()
    _assert_gpu_idle(args_cli.device)
except RuntimeError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import (
    DistillationRunner,
    OnPolicyRunner,
    OnPolicyRunnerAmp,
    OnPolicyRunnerAmpDwaq,
    OnPolicyRunnerAmpROA,
    OnPolicyRunnerDwaq,
    OnPolicyRunnerROA,
)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


RUNNER_CLASSES = {
    "OnPolicyRunner": OnPolicyRunner,
    "DistillationRunner": DistillationRunner,
    "OnPolicyRunnerDwaq": OnPolicyRunnerDwaq,
    "OnPolicyRunnerAmp": OnPolicyRunnerAmp,
    "OnPolicyRunnerAmpDwaq": OnPolicyRunnerAmpDwaq,
    "OnPolicyRunnerROA": OnPolicyRunnerROA,
    "OnPolicyRunnerAmpROA": OnPolicyRunnerAmpROA,
}


def _parse_overrides(raw: str) -> dict[str, Any]:
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid --scenario_overrides_json at column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(overrides, dict):
        raise ValueError("--scenario_overrides_json must decode to an object")
    return overrides


def _parse_command_schedule(raw: str, duration_steps: int) -> list[dict[str, Any]]:
    try:
        schedule = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid --command_schedule_json at column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(schedule, list):
        raise ValueError("--command_schedule_json must decode to an array")
    previous_end = -1
    for index, segment in enumerate(schedule):
        if not isinstance(segment, dict) or set(segment) != {
            "start_step",
            "end_step",
            "command",
        }:
            raise ValueError(f"command schedule segment {index} is invalid")
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
            or end >= duration_steps
        ):
            raise ValueError(
                "command schedule must be ordered, contiguous, and within duration"
            )
        if (
            not isinstance(command, list)
            or len(command) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in command
            )
        ):
            raise ValueError(
                "command schedule values must be finite [vx, vy, yaw_rate]"
            )
        if not all(np.isfinite(float(value)) for value in command):
            raise ValueError("command schedule values must be finite")
        previous_end = end
    if schedule and previous_end != duration_steps - 1:
        raise ValueError("command schedule must cover every evaluation step")
    return schedule


def _scheduled_command(
    schedule: list[dict[str, Any]],
    step: int,
) -> list[float] | None:
    for segment in schedule:
        if segment["start_step"] <= step <= segment["end_step"]:
            return [float(value) for value in segment["command"]]
    return None


def _set_dotted(root: Any, dotted_path: str, value: Any) -> None:
    parts = dotted_path.split(".")
    if parts and parts[0] == "env":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"invalid empty override path: {dotted_path}")
    current = root
    for part in parts[:-1]:
        if isinstance(current, dict):
            if part not in current:
                raise ValueError(f"override path does not exist: {dotted_path}")
            current = current[part]
        else:
            if not hasattr(current, part):
                raise ValueError(f"override path does not exist: {dotted_path}")
            current = getattr(current, part)
    leaf = parts[-1]
    if isinstance(current, dict):
        if leaf not in current:
            raise ValueError(f"override path does not exist: {dotted_path}")
        current[leaf] = value
    else:
        if not hasattr(current, leaf):
            raise ValueError(f"override path does not exist: {dotted_path}")
        setattr(current, leaf, value)


def _make_runner(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
) -> Any:
    runner_class = RUNNER_CLASSES.get(agent_cfg.class_name)
    if runner_class is None:
        raise ValueError(f"unsupported runner class: {agent_cfg.class_name}")
    return runner_class(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
    )


def _policy_tensor(observations: Any) -> torch.Tensor:
    if isinstance(observations, Mapping):
        if "policy" not in observations:
            raise ValueError("observation mapping does not contain policy")
        tensor = observations["policy"]
    else:
        tensor = observations
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("policy observation must be a torch.Tensor")
    if tensor.ndim > 2:
        tensor = tensor.flatten(1)
    return tensor


class ArtifactPolicy:
    """Run a deployment artifact and return actions on the simulation device."""

    def __init__(self, kind: str, path: Path, device: torch.device):
        self.kind = kind
        self.device = device
        self.jit = None
        self.ort_session = None
        self.ort_input = None
        if kind == "jit":
            self.jit = torch.jit.load(str(path), map_location=device).eval()
        elif kind == "onnx":
            import onnxruntime as ort

            self.ort_session = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            self.ort_input = self.ort_session.get_inputs()[0].name

    def __call__(self, observations: Any) -> torch.Tensor:
        policy_input = _policy_tensor(observations)
        if self.kind == "jit":
            assert self.jit is not None
            output = self.jit(policy_input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise ValueError("JIT policy did not return a tensor")
            return output.to(self.device)
        if self.kind == "onnx":
            assert self.ort_session is not None and self.ort_input is not None
            output = self.ort_session.run(
                None,
                {self.ort_input: policy_input.detach().cpu().numpy()},
            )[0]
            return torch.from_numpy(np.asarray(output)).to(self.device)
        raise ValueError(f"unsupported deployment artifact kind: {self.kind}")


def _timeout_mask(extras: Any, dones: torch.Tensor) -> torch.Tensor:
    if isinstance(extras, dict):
        for key in ("time_outs", "timeouts"):
            value = extras.get(key)
            if isinstance(value, torch.Tensor) and value.shape == dones.shape:
                return value.to(device=dones.device, dtype=torch.bool)
    return torch.zeros_like(dones, dtype=torch.bool)


def _finalize_video(video_path: Path) -> str:
    candidates = sorted(
        video_path.parent.glob(f"{video_path.stem}*.mp4"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        return ""
    source = candidates[-1]
    if source != video_path:
        source.replace(video_path)
    return str(video_path)


def _tensor_row(value: Any, env_index: int) -> list[float] | None:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        return None
    if env_index >= value.shape[0]:
        return None
    row = [float(item) for item in value[env_index].detach().cpu().flatten().tolist()]
    if not row or not all(math.isfinite(item) for item in row):
        raise ValueError("tensor row is empty or non-finite")
    return row


def _required_tensor(
    value: Any,
    name: str,
    *,
    minimum_columns: int = 1,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim < 2:
        raise TypeError(f"{name} must be a batched tensor")
    if value.shape[0] < 1 or value.shape[1] < minimum_columns:
        raise ValueError(f"{name} has an incompatible shape")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _finite_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _joint_names(robot: Any) -> list[str]:
    names = [str(name) for name in getattr(robot, "joint_names", [])]
    if not names:
        raise ValueError("robot joint names are unavailable")
    return names


def _parse_camera_offset(raw: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--follow_camera_offset_json must be valid JSON") from exc
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
        or not all(np.isfinite(float(item)) for item in value)
    ):
        raise ValueError("camera offset must be three finite numbers")
    return [float(item) for item in value]


def _update_follow_camera(env: RslRlVecEnvWrapper, env_index: int, offset: list[float]) -> None:
    robot = env.unwrapped.scene["robot"]
    target = _tensor_row(robot.data.root_pos_w, env_index)
    if target is None or len(target) < 3:
        raise ValueError("robot root position is unavailable for camera follow")
    target = target[:3]
    eye = [target[index] + offset[index] for index in range(3)]
    env.unwrapped.sim.set_camera_view(eye, target)


def _review_windows(
    peak_steps: dict[str, int | None],
    termination_first_steps: dict[str, int],
    duration_steps: int,
    step_dt: float,
) -> list[dict[str, Any]]:
    """Return bounded video windows around objective motion-risk evidence."""
    evidence_by_step: dict[int, list[str]] = {}
    for metric, step in peak_steps.items():
        if step is not None:
            evidence_by_step.setdefault(step, []).append(metric)
    for term_name, step in termination_first_steps.items():
        evidence_by_step.setdefault(step, []).append(
            f"termination:{term_name}"
        )
    windows: list[dict[str, Any]] = []
    radius = 30
    for step in sorted(evidence_by_step):
        start_step = max(0, step - radius)
        end_step = min(duration_steps - 1, step + radius)
        windows.append(
            {
                "start_step": start_step,
                "end_step": end_step,
                "start_seconds": start_step * step_dt,
                "end_seconds": (end_step + 1) * step_dt,
                "evidence": sorted(evidence_by_step[step]),
            }
        )
    return windows


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Run one authorized evaluation matrix cell and write bounded JSON."""
    if args_cli.duration_steps <= 0:
        raise ValueError("--duration_steps must be positive")
    if args_cli.num_envs <= 0:
        raise ValueError("--num_envs must be positive")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    command_schedule = _parse_command_schedule(
        args_cli.command_schedule_json,
        args_cli.duration_steps,
    )
    for dotted_path, value in _parse_overrides(
        args_cli.scenario_overrides_json
    ).items():
        _set_dotted(env_cfg, dotted_path, value)
    if command_schedule:
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.resampling_time_range = (1.0e9, 1.0e9)
        if hasattr(command_cfg, "heading_command"):
            command_cfg.heading_command = False
        if hasattr(command_cfg, "rel_standing_envs"):
            command_cfg.rel_standing_envs = 0.0

    checkpoint_path = Path(args_cli.checkpoint or "")
    artifact_path = Path(args_cli.artifact_path)
    result_path = Path(args_cli.result_path)
    video_path = Path(args_cli.video_path) if args_cli.video_path else None
    telemetry_path = Path(args_cli.telemetry_path) if args_cli.telemetry_path else None
    if not checkpoint_path.is_absolute():
        raise ValueError("--checkpoint must be an absolute native checkpoint path")
    if not artifact_path.is_absolute():
        raise ValueError("--artifact_path must be absolute")
    if not result_path.is_absolute():
        raise ValueError("--result_path must be absolute")
    if video_path is not None and not video_path.is_absolute():
        raise ValueError("--video_path must be absolute")
    if telemetry_path is not None and not telemetry_path.is_absolute():
        raise ValueError("--telemetry_path must be absolute")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"artifact does not exist: {artifact_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
    if telemetry_path is not None:
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    raw_env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if video_path is not None else None,
    )
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    if video_path is not None:
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=str(video_path.parent),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.duration_steps,
            name_prefix=video_path.stem,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

    runner_name = agent_cfg.class_name
    runner = _make_runner(env, agent_cfg)
    runner.load(str(checkpoint_path))
    native_policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_module = runner.alg.policy
    except AttributeError:
        policy_module = runner.alg.actor_critic

    artifact_policy = None
    if args_cli.artifact_kind != "native":
        artifact_policy = ArtifactPolicy(
            args_cli.artifact_kind,
            artifact_path,
            torch.device(env.unwrapped.device),
        )

    command_term = None
    if command_schedule:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        if not hasattr(command_term, "vel_command_b"):
            raise ValueError(
                "base_velocity command term does not support deterministic scheduling"
            )
    observations = env.get_observations()
    camera_offset = _parse_camera_offset(args_cli.follow_camera_offset_json)
    if args_cli.follow_robot_camera:
        _update_follow_camera(env, args_cli.telemetry_env_index, camera_offset)
    previous_actions: torch.Tensor | None = None
    telemetry_samples: list[dict[str, Any]] = []
    telemetry_joint_names: list[str] = []
    telemetry_joint_names_captured = False
    telemetry_expected_samples = (
        args_cli.duration_steps + args_cli.telemetry_stride - 1
    ) // args_cli.telemetry_stride
    telemetry_ledger = None
    if telemetry_path is not None:
        telemetry_expected_counts = {
            name: telemetry_expected_samples for name in TELEMETRY_SIGNALS
        }
        telemetry_expected_counts["joint_names"] = 1
        telemetry_ledger = SignalLedger(
            telemetry_expected_counts,
            required_signals=required_signals_for_runner(runner_name),
        )
    metric_input_names = (
        "command",
        "root_linear_velocity_b",
        "root_angular_velocity_b",
        "projected_gravity_b",
        "joint_velocity",
        "applied_torque",
    )
    metric_ledger = SignalLedger(
        {name: args_cli.duration_steps for name in metric_input_names},
        required_signals=metric_input_names,
    )
    reward_sum = 0.0
    sample_count = 0
    termination_count = 0
    timeout_count = 0
    termination_term_counts: dict[str, int] = {}
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    termination_term_names = (
        tuple(termination_manager.active_terms)
        if termination_manager is not None
        else ()
    )
    tracking_xy_sq_sum = 0.0
    tracking_yaw_sq_sum = 0.0
    tracking_samples = 0
    tilt_sq_sum = 0.0
    tilt_samples = 0
    max_tilt = 0.0
    action_rate_sq_sum = 0.0
    action_rate_samples = 0
    max_abs_action = 0.0
    max_abs_joint_velocity = 0.0
    max_abs_applied_torque = 0.0
    max_abs_action_error = 0.0
    peak_steps: dict[str, int | None] = {
        "max_abs_action": None,
        "max_abs_action_error": None,
        "max_abs_applied_torque": None,
        "max_abs_joint_velocity": None,
        "max_tilt": None,
    }
    termination_first_steps: dict[str, int] = {}
    start_time = time.monotonic()

    for step in range(args_cli.duration_steps):
        if args_cli.follow_robot_camera:
            _update_follow_camera(env, args_cli.telemetry_env_index, camera_offset)
        active_scheduled_command = _scheduled_command(command_schedule, step)
        if active_scheduled_command is not None:
            assert command_term is not None
            command_term.vel_command_b[:] = torch.tensor(
                active_scheduled_command,
                device=env.unwrapped.device,
                dtype=command_term.vel_command_b.dtype,
            )
            observations = env.get_observations()
        with torch.inference_mode():
            native_actions = native_policy(observations)
            if artifact_policy is None:
                actions = native_actions
            else:
                actions = artifact_policy(observations)
                error = float(
                    torch.max(torch.abs(native_actions - actions)).item()
                )
                if error > max_abs_action_error or peak_steps[
                    "max_abs_action_error"
                ] is None:
                    max_abs_action_error = error
                    peak_steps["max_abs_action_error"] = step
            if not torch.isfinite(actions).all():
                raise FloatingPointError("policy produced non-finite actions")
            observations, rewards, dones, extras = env.step(actions)
            if hasattr(policy_module, "reset"):
                policy_module.reset(dones)

        rewards_tensor = torch.as_tensor(rewards)
        dones_tensor = torch.as_tensor(dones, dtype=torch.bool)
        timeouts = _timeout_mask(extras, dones_tensor)
        terminations = dones_tensor & ~timeouts
        reward_sum += float(rewards_tensor.sum().item())
        sample_count += int(rewards_tensor.numel())
        termination_count += int(terminations.sum().item())
        timeout_count += int(timeouts.sum().item())
        for term_name in termination_term_names:
            term_value = termination_manager.get_term(term_name)
            term_count = int(term_value.sum().item())
            termination_term_counts[term_name] = (
                termination_term_counts.get(term_name, 0)
                + term_count
            )
            if term_count and term_name not in termination_first_steps:
                termination_first_steps[term_name] = step
        current_max_action = float(torch.max(torch.abs(actions)).item())
        if (
            current_max_action > max_abs_action
            or peak_steps["max_abs_action"] is None
        ):
            max_abs_action = current_max_action
            peak_steps["max_abs_action"] = step
        if previous_actions is not None:
            delta = actions - previous_actions
            action_rate_sq_sum += float(torch.square(delta).sum().item())
            action_rate_samples += int(delta.numel())
        previous_actions = actions.detach().clone()

        command = metric_ledger.capture(
            "command",
            lambda: _required_tensor(
                env.unwrapped.command_manager.get_command("base_velocity")
                if active_scheduled_command is None
                else torch.tensor(
                    active_scheduled_command,
                    device=actions.device,
                    dtype=actions.dtype,
                ).expand(actions.shape[0], -1),
                "command",
                minimum_columns=3,
            ),
        )
        root_linear_velocity = metric_ledger.capture(
            "root_linear_velocity_b",
            lambda: _required_tensor(
                env.unwrapped.scene["robot"].data.root_lin_vel_b,
                "root_linear_velocity_b",
                minimum_columns=2,
            ),
        )
        root_angular_velocity = metric_ledger.capture(
            "root_angular_velocity_b",
            lambda: _required_tensor(
                env.unwrapped.scene["robot"].data.root_ang_vel_b,
                "root_angular_velocity_b",
                minimum_columns=3,
            ),
        )
        projected_gravity = metric_ledger.capture(
            "projected_gravity_b",
            lambda: _required_tensor(
                env.unwrapped.scene["robot"].data.projected_gravity_b,
                "projected_gravity_b",
                minimum_columns=2,
            ),
        )
        joint_velocity = metric_ledger.capture(
            "joint_velocity",
            lambda: _required_tensor(
                env.unwrapped.scene["robot"].data.joint_vel,
                "joint_velocity",
            ),
        )
        applied_torque = metric_ledger.capture(
            "applied_torque",
            lambda: _required_tensor(
                env.unwrapped.scene["robot"].data.applied_torque,
                "applied_torque",
            ),
        )

        if projected_gravity is not None:
            gravity_xy = projected_gravity[:, :2]
            tilt = torch.linalg.vector_norm(gravity_xy, dim=1)
            tilt_sq_sum += float(torch.square(tilt).sum().item())
            tilt_samples += int(tilt.numel())
            current_max_tilt = float(torch.max(tilt).item())
            if (
                current_max_tilt > max_tilt
                or peak_steps["max_tilt"] is None
            ):
                max_tilt = current_max_tilt
                peak_steps["max_tilt"] = step
        if joint_velocity is not None:
            current_max_joint_velocity = float(torch.max(torch.abs(joint_velocity)).item())
            if (
                current_max_joint_velocity > max_abs_joint_velocity
                or peak_steps["max_abs_joint_velocity"] is None
            ):
                max_abs_joint_velocity = current_max_joint_velocity
                peak_steps["max_abs_joint_velocity"] = step
        if applied_torque is not None:
            current_max_applied_torque = float(torch.max(torch.abs(applied_torque)).item())
            if (
                current_max_applied_torque > max_abs_applied_torque
                or peak_steps["max_abs_applied_torque"] is None
            ):
                max_abs_applied_torque = current_max_applied_torque
                peak_steps["max_abs_applied_torque"] = step

        if (
            command is not None
            and root_linear_velocity is not None
            and root_angular_velocity is not None
        ):
            xy_error = command[:, :2] - root_linear_velocity[:, :2]
            yaw_error = command[:, 2] - root_angular_velocity[:, 2]
            tracking_xy_sq_sum += float(torch.square(xy_error).sum().item())
            tracking_yaw_sq_sum += float(torch.square(yaw_error).sum().item())
            tracking_samples += int(command.shape[0])

        if telemetry_ledger is not None and step % args_cli.telemetry_stride == 0:
            env_index = args_cli.telemetry_env_index
            if not telemetry_joint_names_captured:
                names = telemetry_ledger.capture(
                    "joint_names",
                    lambda: _joint_names(env.unwrapped.scene["robot"]),
                )
                if names is not None:
                    telemetry_joint_names = names
                    telemetry_joint_names_captured = True
            sample = {
                "step": step,
                "sim_time_seconds": step * float(env.unwrapped.step_dt),
                "command": telemetry_ledger.capture(
                    "command", lambda: _tensor_row(command, env_index)
                ),
                "reward": telemetry_ledger.capture(
                    "reward",
                    lambda: _finite_scalar(
                        rewards_tensor[env_index].item(), "reward"
                    ),
                ),
                "done": telemetry_ledger.capture(
                    "done", lambda: bool(dones_tensor[env_index].item())
                ),
                "timeout": telemetry_ledger.capture(
                    "timeout", lambda: bool(timeouts[env_index].item())
                ),
                "root_position_w": telemetry_ledger.capture(
                    "root_position_w",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.root_pos_w, env_index
                    ),
                ),
                "root_quaternion_w": telemetry_ledger.capture(
                    "root_quaternion_w",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.root_quat_w, env_index
                    ),
                ),
                "root_linear_velocity_b": telemetry_ledger.capture(
                    "root_linear_velocity_b",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.root_lin_vel_b, env_index
                    ),
                ),
                "root_angular_velocity_b": telemetry_ledger.capture(
                    "root_angular_velocity_b",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.root_ang_vel_b, env_index
                    ),
                ),
                "projected_gravity_b": telemetry_ledger.capture(
                    "projected_gravity_b",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.projected_gravity_b,
                        env_index,
                    ),
                ),
                "joint_position": telemetry_ledger.capture(
                    "joint_position",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.joint_pos, env_index
                    ),
                ),
                "joint_velocity": telemetry_ledger.capture(
                    "joint_velocity",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.joint_vel, env_index
                    ),
                ),
                "applied_torque": telemetry_ledger.capture(
                    "applied_torque",
                    lambda: _tensor_row(
                        env.unwrapped.scene["robot"].data.applied_torque, env_index
                    ),
                ),
                "action": telemetry_ledger.capture(
                    "action", lambda: _tensor_row(actions, env_index)
                ),
            }
            telemetry_samples.append(sample)

    elapsed = max(time.monotonic() - start_time, 1e-12)
    simulated_seconds = (
        args_cli.duration_steps * float(env.unwrapped.step_dt)
    )
    metric_signal_report = metric_ledger.report()["signals"]

    metric_availability = metric_availability_report(
        metric_signal_report,
        {
            "tracking_xy_rmse": ("command", "root_linear_velocity_b"),
            "tracking_yaw_rmse": ("command", "root_angular_velocity_b"),
            "tilt_rms": ("projected_gravity_b",),
            "max_tilt": ("projected_gravity_b",),
            "max_abs_joint_velocity": ("joint_velocity",),
            "max_abs_applied_torque": ("applied_torque",),
        },
    )
    metrics: dict[str, float] = {
        "mean_reward": reward_sum / max(sample_count, 1),
        "termination_rate": termination_count / max(sample_count, 1),
        "timeout_rate": timeout_count / max(sample_count, 1),
        "max_abs_action": max_abs_action,
        "action_rate_rms": (
            action_rate_sq_sum / max(action_rate_samples, 1)
        )
        ** 0.5,
        "max_abs_action_error": max_abs_action_error,
        "real_time_factor": simulated_seconds / elapsed,
    }
    record_complete_metric(
        metrics,
        metric_availability,
        "tracking_xy_rmse",
        lambda: (tracking_xy_sq_sum / (tracking_samples * 2)) ** 0.5,
    )
    record_complete_metric(
        metrics,
        metric_availability,
        "tracking_yaw_rmse",
        lambda: (tracking_yaw_sq_sum / tracking_samples) ** 0.5,
    )
    record_complete_metric(
        metrics,
        metric_availability,
        "tilt_rms",
        lambda: (tilt_sq_sum / tilt_samples) ** 0.5,
    )
    record_complete_metric(
        metrics, metric_availability, "max_tilt", lambda: max_tilt
    )
    record_complete_metric(
        metrics,
        metric_availability,
        "max_abs_joint_velocity",
        lambda: max_abs_joint_velocity,
    )
    record_complete_metric(
        metrics,
        metric_availability,
        "max_abs_applied_torque",
        lambda: max_abs_applied_torque,
    )
    for term_name, count in termination_term_counts.items():
        normalized_name = re.sub(r"[^a-z0-9]+", "_", term_name.lower()).strip(
            "_"
        )
        metric_value = count / max(sample_count, 1)
        metrics[f"termination_term_{normalized_name}_rate"] = metric_value
        if normalized_name == "illegal_contact":
            metrics["illegal_contact_rate"] = metric_value

    step_dt = float(env.unwrapped.step_dt)
    env.close()
    recorded_video = _finalize_video(video_path) if video_path is not None else ""
    telemetry_evidence = telemetry_report(
        requested=telemetry_path is not None,
        runner=runner_name,
        ledger=telemetry_ledger,
    )
    if telemetry_path is not None:
        telemetry = {
            "version": 2,
            "run_id": args_cli.run_id,
            "candidate_id": args_cli.candidate_id,
            "runner": runner_name,
            "environment_index": args_cli.telemetry_env_index,
            "stride": args_cli.telemetry_stride,
            "step_dt_seconds": step_dt,
            "joint_names": telemetry_joint_names,
            **telemetry_evidence,
            "samples": telemetry_samples,
        }
        telemetry_path.write_text(
            json.dumps(
                telemetry,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    result = {
        "version": 1,
        "run_id": args_cli.run_id,
        "status": "completed",
        "candidate_id": args_cli.candidate_id,
        "runner": runner_name,
        "artifact": args_cli.artifact_kind,
        "scenario_id": args_cli.scenario_id,
        "seed": args_cli.seed,
        "duration_steps": args_cli.duration_steps,
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(artifact_path),
        "video_path": recorded_video,
        "telemetry_path": str(telemetry_path) if telemetry_path is not None else "",
        "telemetry_status": telemetry_evidence["telemetry_status"],
        "missing_required_signals": telemetry_evidence[
            "missing_required_signals"
        ],
        "telemetry_required_for_complete_assessment": telemetry_evidence[
            "telemetry_required_for_complete_assessment"
        ],
        "telemetry": telemetry_evidence,
        "training_overlap": bool(args_cli.allow_training_overlap),
        "metrics": metrics,
        "metric_availability": metric_availability,
        "motion_evidence": {
            "step_dt_seconds": step_dt,
            "peak_steps": peak_steps,
            "termination_first_steps": termination_first_steps,
            "review_windows": _review_windows(
                peak_steps,
                termination_first_steps,
                args_cli.duration_steps,
                step_dt,
            ),
        },
    }
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Evaluation result: {result_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

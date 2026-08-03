#!/usr/bin/env python3
"""Export one RSL-RL checkpoint to JIT/ONNX and prove open-loop parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ORIGINAL_ARGV = tuple(sys.argv)
REPO_ROOT = Path(__file__).resolve().parents[4]
RSL_SCRIPT_DIR = REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
sys.path.insert(0, str(RSL_SCRIPT_DIR))

from isaaclab.app import AppLauncher

import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point")
parser.add_argument("--run_id", required=True)
parser.add_argument("--checkpoint_id", required=True)
parser.add_argument("--checkpoint_sha256", required=True)
parser.add_argument("--export_run_id", required=True)
parser.add_argument("--jit_path", required=True)
parser.add_argument("--onnx_path", required=True)
parser.add_argument("--result_path", required=True)
parser.add_argument("--max_abs_action_error", type=float, required=True)
parser.add_argument("--minimum_parity_samples", type=int, required=True)
parser.add_argument("--history_contract", required=True)
parser.add_argument("--normalization_contract", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--require_idle_gpu", action="store_true")
cli_args.add_rsl_rl_args(parser)
if "-h" in ORIGINAL_ARGV or "--help" in ORIGINAL_ARGV:
    parser.print_help()
    raise SystemExit(0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _assert_gpu_idle(device: str | None) -> None:
    if not args_cli.require_idle_gpu or device is None or not device.startswith("cuda"):
        return
    gpu_index = device.split(":", 1)[1] if ":" in device else "0"
    result = subprocess.run(
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
    if result.returncode != 0:
        raise RuntimeError("cannot verify that the export GPU is idle")
    active = [
        line.strip() for line in result.stdout.splitlines()
        if line.strip().isdigit()
    ]
    if active:
        raise RuntimeError(
            "policy export GPU is not idle; active PID(s): "
            + ", ".join(active)
        )


checkpoint_path = Path(args_cli.checkpoint or "")
jit_path = Path(args_cli.jit_path)
onnx_path = Path(args_cli.onnx_path)
result_path = Path(args_cli.result_path)
if (
    not checkpoint_path.is_absolute()
    or not checkpoint_path.is_file()
    or checkpoint_path.is_symlink()
    or _sha256(checkpoint_path) != args_cli.checkpoint_sha256
):
    parser.error("checkpoint path or SHA-256 is invalid")
if args_cli.minimum_parity_samples <= 0 or args_cli.num_envs < args_cli.minimum_parity_samples:
    parser.error("num_envs must cover minimum_parity_samples")
if args_cli.max_abs_action_error < 0:
    parser.error("max_abs_action_error must be non-negative")
for output in (jit_path, onnx_path, result_path):
    if not output.is_absolute():
        parser.error("all policy export outputs must be absolute")
    if output.exists():
        parser.error(f"refusing to overwrite policy export output: {output}")
if len({jit_path.parent, onnx_path.parent, result_path.parent}) != 1:
    parser.error("JIT, ONNX, and result outputs must share one attempt directory")
result_path.parent.mkdir(parents=True, exist_ok=True)
_assert_gpu_idle(args_cli.device)

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rsl_rl.runners import (  # noqa: E402
    DistillationRunner,
    OnPolicyRunner,
    OnPolicyRunnerAmp,
    OnPolicyRunnerAmpDwaq,
    OnPolicyRunnerAmpROA,
    OnPolicyRunnerDwaq,
    OnPolicyRunnerROA,
)

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import robot_lab.tasks  # noqa: F401,E402


RUNNER_CLASSES = {
    "OnPolicyRunner": OnPolicyRunner,
    "DistillationRunner": DistillationRunner,
    "OnPolicyRunnerDwaq": OnPolicyRunnerDwaq,
    "OnPolicyRunnerAmp": OnPolicyRunnerAmp,
    "OnPolicyRunnerAmpDwaq": OnPolicyRunnerAmpDwaq,
    "OnPolicyRunnerROA": OnPolicyRunnerROA,
    "OnPolicyRunnerAmpROA": OnPolicyRunnerAmpROA,
}


class DWAQDeploymentWrapper(nn.Module):
    def __init__(self, policy: Any, num_obs: int):
        super().__init__()
        self.encoder_backbone = policy.encoder_backbone
        self.encode_mean_vel = policy.encode_mean_vel
        self.encode_mean_latent = policy.encode_mean_latent
        self.actor = policy.actor
        self.actor_obs_normalizer = policy.actor_obs_normalizer
        self.num_obs = num_obs

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        current = history[:, -self.num_obs:]
        features = self.encoder_backbone(history)
        latent = torch.cat(
            (
                self.encode_mean_vel(features),
                self.encode_mean_latent(features),
            ),
            dim=-1,
        )
        actor_input = torch.cat((latent, current), dim=-1)
        if self.actor_obs_normalizer is not None:
            actor_input = self.actor_obs_normalizer(actor_input)
        return self.actor(actor_input)


class ROADeploymentWrapper(nn.Module):
    def __init__(self, policy: Any, num_obs: int):
        super().__init__()
        self.history_encoder = policy.history_encoder
        self.actor = policy.actor
        self.actor_obs_normalizer = policy.actor_obs_normalizer
        self.num_obs = num_obs

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        current = history[:, -self.num_obs:]
        if self.actor_obs_normalizer is not None:
            current = self.actor_obs_normalizer(current)
        hist_latent, code_vel = self.history_encoder(history)
        return self.actor(torch.cat((current, code_vel, hist_latent), dim=-1))


class StandardDeploymentWrapper(nn.Module):
    def __init__(self, policy: Any):
        super().__init__()
        if hasattr(policy, "actor"):
            self.actor = policy.actor
        elif hasattr(policy, "student"):
            self.actor = policy.student
        else:
            raise RuntimeError("policy has neither actor nor student module")
        self.normalizer = getattr(
            policy,
            "actor_obs_normalizer",
            getattr(policy, "student_obs_normalizer", None),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            observations = self.normalizer(observations)
        return self.actor(observations)


def _policy_tensor(observations: Any) -> torch.Tensor:
    if isinstance(observations, Mapping):
        if "policy" not in observations:
            raise RuntimeError("observation mapping does not contain policy")
        tensor = observations["policy"]
    else:
        tensor = observations
    if not isinstance(tensor, torch.Tensor):
        raise RuntimeError("policy observation is not a tensor")
    return tensor


def _cpu_observation_batch(observations: Any, sample_count: int) -> Any:
    if isinstance(observations, torch.Tensor):
        return observations[:sample_count].detach().cpu()
    if isinstance(observations, Mapping):
        return {
            key: _cpu_observation_batch(value, sample_count)
            for key, value in observations.items()
        }
    raise RuntimeError(
        "observation batch contains an unsupported non-tensor value"
    )


def _action_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("policy action is not a tensor")
    return value


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return _bytes_sha256(array.tobytes())


def _onnx_action(path: Path, policy_input: torch.Tensor) -> torch.Tensor:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output = session.run(
        None,
        {input_name: policy_input.detach().cpu().numpy()},
    )[0]
    return torch.from_numpy(np.asarray(output))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(raw_env.unwrapped, DirectMARLEnv):
        raw_env = multi_agent_to_single_agent(raw_env)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
        runner_class = RUNNER_CLASSES.get(agent_cfg.class_name)
        if runner_class is None:
            raise RuntimeError(f"unsupported RSL-RL runner: {agent_cfg.class_name}")
        runner = runner_class(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
        )
        runner.load(str(checkpoint_path))
        native_policy = runner.get_inference_policy(
            device=env.unwrapped.device
        )
        observations = env.get_observations()
        policy_observation = _policy_tensor(observations)
        actual_history_contract = (
            "flat_time_major_history"
            if policy_observation.ndim > 2
            else "current_observation"
        )
        if actual_history_contract != args_cli.history_contract:
            raise RuntimeError(
                "actual policy observation history contract differs from approval"
            )
        policy_input = (
            policy_observation.flatten(1)
            if policy_observation.ndim > 2
            else policy_observation
        )
        with torch.no_grad():
            native_device_action = _action_tensor(native_policy(observations))
        native_device_action = native_device_action.detach().cpu()
        sample_count = min(
            args_cli.minimum_parity_samples,
            policy_input.shape[0],
            native_device_action.shape[0],
        )
        if sample_count < args_cli.minimum_parity_samples:
            raise RuntimeError(
                "environment batch does not cover minimum parity samples"
            )
        policy_input = policy_input[:sample_count].detach().cpu()
        native_device_action = native_device_action[:sample_count]
        cpu_observations = _cpu_observation_batch(
            observations,
            sample_count,
        )
        try:
            policy_module = runner.alg.policy
        except AttributeError:
            policy_module = runner.alg.actor_critic
        policy_module.cpu()
        policy_module.eval()
        with torch.no_grad():
            native_action = _action_tensor(native_policy(cpu_observations))
        native_action = native_action.detach().cpu()
        if native_device_action.shape != native_action.shape:
            raise RuntimeError(
                "Native device and CPU action shapes differ"
            )
        native_device_to_cpu_error = float(
            torch.max(
                torch.abs(native_device_action - native_action)
            ).item()
        )
        runner_name = agent_cfg.class_name
        if "ROA" in runner_name:
            if policy_observation.ndim <= 2:
                raise RuntimeError("ROA export requires a time-major history")
            deployment_model: nn.Module = ROADeploymentWrapper(
                policy_module,
                policy_observation.shape[-1],
            ).eval()
            normalization_contract = "current_frame_only"
        elif "Dwaq" in runner_name:
            if policy_observation.ndim <= 2:
                raise RuntimeError("DWAQ export requires a time-major history")
            deployment_model = DWAQDeploymentWrapper(
                policy_module,
                policy_observation.shape[-1],
            ).eval()
            normalization_contract = "combined_actor_input"
        else:
            if bool(getattr(policy_module, "is_recurrent", False)):
                raise RuntimeError(
                    "automatic recurrent policy export requires a reviewed "
                    "state-reset adapter"
                )
            deployment_model = StandardDeploymentWrapper(policy_module).eval()
            normalization_contract = "backend_export_helper"
        if normalization_contract != args_cli.normalization_contract:
            raise RuntimeError(
                "actual normalization contract differs from approval"
            )
        example_input = policy_input[:1]
        traced = torch.jit.trace(deployment_model, example_input)
        traced.save(str(jit_path))
        torch.onnx.export(
            deployment_model,
            example_input,
            str(onnx_path),
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={
                "obs": {0: "batch"},
                "actions": {0: "batch"},
            },
            opset_version=18,
        )
        jit_model = torch.jit.load(str(jit_path), map_location="cpu").eval()
        with torch.no_grad():
            jit_action = _action_tensor(jit_model(policy_input)).cpu()
        onnx_action = _onnx_action(onnx_path, policy_input).cpu()
        artifacts: dict[str, Any] = {}
        for kind, path, action in (
            ("jit", jit_path, jit_action),
            ("onnx", onnx_path, onnx_action),
        ):
            finite = bool(torch.isfinite(action).all().item())
            if action.shape != native_action.shape:
                raise RuntimeError(f"{kind} action shape differs from Native")
            error = float(torch.max(torch.abs(action - native_action)).item())
            if not finite or error > args_cli.max_abs_action_error:
                raise RuntimeError(
                    f"{kind} action parity gate failed: "
                    f"finite={finite}, max_abs_action_error={error:.9g}, "
                    f"limit={args_cli.max_abs_action_error:.9g}"
                )
            artifacts[kind] = {
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "input_shape": list(policy_input.shape),
                "output_shape": list(action.shape),
                "finite": finite,
                "max_abs_action_error": error,
            }
        result = {
            "version": 2,
            "run_id": args_cli.run_id,
            "checkpoint_id": args_cli.checkpoint_id,
            "export_run_id": args_cli.export_run_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": args_cli.checkpoint_sha256,
            "status": "completed",
            "artifacts": artifacts,
            "parity": {
                "sample_count": sample_count,
                "observation_batch_sha256": _tensor_digest(policy_input),
                "native_output_sha256": _tensor_digest(native_action),
                "native_device_to_cpu_max_abs_action_error": (
                    native_device_to_cpu_error
                ),
                "history_contract": actual_history_contract,
                "normalization_contract": normalization_contract,
            },
        }
        _atomic_write(result_path, result)
    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()

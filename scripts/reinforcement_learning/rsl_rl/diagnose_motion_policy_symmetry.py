"""CPU-only mirror diagnosis from a frozen LW checkpoint and sealed telemetry.

No Isaac Sim, optimizer, rollout, export or configuration mutation. The nominal
observations must reproduce recorded mean actions within --replay-atol. Noisy
randomized observations cannot be recovered from these telemetry bundles: their
results are explicitly noise-free state probes using nominal default positions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from policy_evaluation_evidence import validate_evaluation_bundle

REPO = Path(__file__).resolve().parents[3]
MAPPING = REPO / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/mdp/symmetry.py"
spec = importlib.util.spec_from_file_location("lw_tracking_mirror", MAPPING)
mirror = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirror)
PHASES = {"takeoff": (1.0, 1.45), "landing": (1.45, 1.90)}


class ConfigLoader(yaml.SafeLoader):
    """Read saved config tags as data; never import or instantiate Python objects."""


def _python_tag(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", _python_tag)


def read_config(path):
    return yaml.load(Path(path).read_text(), Loader=ConfigLoader)


def file_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}


def rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid wxyz quaternion.")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def validate_config(env_cfg, agent_cfg):
    policy = env_cfg["observations"]["policy"]
    active = [k for k, v in policy.items() if isinstance(v, dict) and "func" in v]
    if active != [name for name, _ in mirror.POLICY_TERMS]:
        raise ValueError("Saved policy observation order differs from the mirror contract.")
    if policy["history_length"] != 10 or not policy["flatten_history_dim"] or not policy["concatenate_terms"]:
        raise ValueError("Expected ten per-term flattened history slots.")
    for name, _ in mirror.POLICY_TERMS:
        term = policy[name]
        if term.get("scale") is not None or term.get("modifiers") or term.get("clip") != [-100.0, 100.0]:
            raise ValueError("Unsupported observation preprocessing.")
    if tuple(env_cfg["commands"]["motion"]["joint_names"]) != mirror.JOINT_NAMES:
        raise ValueError("Reference joint order differs from the mirror contract.")
    for name in ("joint_pos", "joint_vel"):
        if tuple(policy[name]["params"]["asset_cfg"]["joint_names"]) != mirror.JOINT_NAMES:
            raise ValueError("Observation joint order differs from the mirror contract.")
    for name, names, scale in (("joint_pos", mirror.JOINT_NAMES[:8], .25),
                               ("joint_vel", mirror.JOINT_NAMES[8:], 1.0)):
        action = env_cfg["actions"][name]
        if tuple(action["joint_names"]) != names or action["scale"] != scale or not action["use_default_offset"]:
            raise ValueError("Unsupported action mapping.")
    if agent_cfg["policy"]["actor_obs_normalization"] or agent_cfg["policy"]["activation"] != "elu":
        raise ValueError("Only the saved unnormalized ELU actor is supported.")


def load_actor(checkpoint, agent_cfg):
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = data["model_state_dict"]
    dimensions = [590, *agent_cfg["policy"]["actor_hidden_dims"], 10]
    layers = []
    for i, (a, b) in enumerate(zip(dimensions, dimensions[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(dimensions)-2:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict({k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")}, strict=True)
    actor.eval().requires_grad_(False)
    return actor, data["iter"], state["std"].tolist()


def reconstruct_observations(telemetry, motion, env_cfg):
    """Match input at action f using the preceding ten post-physics states.

    Command/reference orientation advance to f+1 after recording state f. Drop
    the first ten actions of each episode rather than inventing reset history.
    Randomized observation noise and default-pose offsets are NOT reconstructed.
    """
    d = telemetry["motion_diagnostics"]
    history = d["observation_history"]
    expected = [name for name, _ in mirror.POLICY_TERMS]
    if history["term_order"]["policy"] != expected or history["input_shapes"]["policy"] != [1, 590]:
        raise ValueError("Telemetry observation contract differs.")
    if history["flatten_order"] != "per-term history, oldest to newest":
        raise ValueError("Unsupported history flatten order.")
    if history["history_lengths"]["policy"] != dict.fromkeys(expected, 10):
        raise ValueError("Unexpected per-term history lengths.")
    if d["action_sampling"] != {"mode": "mean"} or d["fps"] != 50.0:
        raise ValueError("Expected 50 Hz mean-action telemetry.")
    if d["sampling_phase"] != "post_physics_pre_reset" or telemetry["step_dt_seconds"] != .02:
        raise ValueError("Unsupported telemetry timing.")
    ids = [telemetry["joint_names"].index(n) for n in mirror.JOINT_NAMES]
    defaults = env_cfg["scene"]["robot"]["init_state"]["joint_pos"]
    default_q = np.array([defaults[n] for n in mirror.JOINT_NAMES])
    base_id = d["initial_runtime"]["body_names"].index("base_link")
    frames = []
    samples = telemetry["samples"]
    for row in samples:
        f = row["frame_index"]
        np.testing.assert_allclose(row["reference_joint_position"], motion["joint_pos"][f], atol=1e-7, rtol=0)
        np.testing.assert_allclose(row["reference_joint_velocity"], motion["joint_vel"][f], atol=1e-7, rtol=0)
        if row["done"] or f+1 >= len(motion["joint_pos"]):
            frames.append(None)
            continue
        command = np.r_[motion["joint_pos"][f+1, ids], motion["joint_vel"][f+1, ids]]
        orientation = rotation_matrix(row["root_quaternion_w"]).T @ rotation_matrix(motion["body_quat_w"][f+1, base_id])
        position = np.asarray(row["joint_position"])[ids] - default_q
        position[-2:] = 0.0
        frames.append([np.clip(v, -100, 100) for v in (
            command, orientation[:, :2].reshape(-1), np.asarray(row["root_angular_velocity_b"]),
            position, np.asarray(row["joint_velocity"])[ids], np.asarray(row["action"]))])
    obs, recorded, labels = [], [], []
    for i, row in enumerate(samples):
        window = samples[max(0, i-10):i]
        if len(window) != 10 or any(
            frames[i-10+j] is None or old["episode_id"] != row["episode_id"]
            or old["frame_index"] != row["frame_index"] - 10 + j
            or old["step"] != row["step"] - 10 + j for j, old in enumerate(window)
        ):
            continue
        obs.append(np.concatenate([np.concatenate([frames[j][term] for j in range(i-10, i)]) for term in range(6)]))
        recorded.append(row["action"])
        t = row["frame_index"] / d["fps"]
        phase = next((name for name, (start, end) in PHASES.items() if start <= t < end), "other")
        labels.append({"episode_id": row["episode_id"], "frame_index": row["frame_index"],
                       "reference_time_s": t, "phase": phase, "step": row["step"]})
    if not obs:
        raise ValueError("No complete, reset-free ten-frame history windows.")
    return torch.tensor(np.asarray(obs), dtype=torch.float32), np.asarray(recorded), labels


def error_statistics(error, original):
    """Raw action-unit MSE matches the PPO mirror-loss reduction."""
    mse = float(np.mean(error**2))
    return {"count": len(error), "mse_raw_action": mse, "rms_raw_action": float(np.sqrt(mse)),
            "p95_abs_raw_action": float(np.quantile(abs(error), .95)),
            "max_abs_raw_action": float(abs(error).max()),
            "relative_rms_to_actor_output": float(np.sqrt(mse) / max(np.sqrt(np.mean(original**2)), 1e-12)),
            "position_target_rms_deg": float(np.degrees(.25*np.sqrt(np.mean(error[:, :8]**2)))),
            "wheel_velocity_target_rms_rad_s": float(np.sqrt(np.mean(error[:, 8:]**2))),
            "per_action": {name: {"rms_raw": float(np.sqrt(np.mean(error[:, j]**2))),
                                  "mean_signed_raw": float(error[:, j].mean()),
                                  "max_abs_raw": float(abs(error[:, j]).max()),
                                  "physical_rms": float(np.sqrt(np.mean(error[:, j]**2)) * (180/np.pi*.25 if j < 8 else 1)),
                                  "physical_unit": "deg_position_target" if j < 8 else "rad_s_velocity_target"}
                           for j, name in enumerate(mirror.JOINT_NAMES)}}


def diagnose(actor, obs, recorded, labels, exact, replay_atol):
    with torch.inference_mode():
        pred = actor(obs)
        reflected_pred = actor(mirror.mirror_policy_observations(obs))
        target = mirror.mirror_actions(pred)
    error = (reflected_pred-target).numpy()
    pred = pred.numpy()
    replay_error = pred-recorded
    if not np.isfinite(error).all() or not np.isfinite(pred).all():
        raise ValueError("Non-finite frozen actor output.")
    max_replay = float(abs(replay_error).max())
    if exact and max_replay > replay_atol:
        raise ValueError(f"Nominal reconstruction failed action replay: {max_replay} > {replay_atol}")
    result = {"input_semantics": "action_replay_verified" if exact else "noise_free_nominal_default_probe_only",
              "recorded_action_comparison": {"max_abs": max_replay, "rms": float(np.sqrt(np.mean(replay_error**2))),
                                             "exact_replay_claimed": exact, "tolerance": replay_atol if exact else None},
              "all": error_statistics(error, pred), "phases": {}}
    for phase in (*PHASES, "other"):
        indices = [i for i, row in enumerate(labels) if row["phase"] == phase]
        if not indices:
            raise ValueError(f"No samples for phase {phase}")
        result["phases"][phase] = error_statistics(error[indices], pred[indices])
    result["per_episode"] = {}
    for episode in sorted(set(row["episode_id"] for row in labels)):
        indices = [i for i, row in enumerate(labels) if row["episode_id"] == episode]
        result["per_episode"][episode] = error_statistics(error[indices], pred[indices])
    return result, error, pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-atol", type=float, default=1e-4)
    args = parser.parse_args()
    if not 0 < args.replay_atol <= 1e-4:
        parser.error("Replay tolerance must be positive and at most 1e-4 raw action units.")
    if args.output_dir.exists():
        parser.error("Output directory already exists; evidence must not be overwritten.")
    torch.set_num_threads(2)
    paths = [args.checkpoint, args.env_config, args.agent_config, Path(__file__), MAPPING]
    inputs = [file_binding(p) for p in paths]
    env_cfg, agent_cfg = read_config(args.env_config), read_config(args.agent_config)
    validate_config(env_cfg, agent_cfg)
    actor, iteration, std = load_actor(args.checkpoint, agent_cfg)
    cases, csv_rows, plot_data = {}, [], {}
    for path in args.result:
        path = path.resolve()
        validate_evaluation_bundle(path)
        result = json.loads(path.read_text())
        tp = Path(result["telemetry_path"])
        with gzip.open(tp, "rt") as f:
            telemetry = json.load(f)
        binding = telemetry["inputs"]["checkpoint"]
        if binding["sha256"] != inputs[0]["sha256"]:
            raise ValueError("Telemetry and requested checkpoint differ.")
        if telemetry["telemetry_status"] != "complete" or telemetry["missing_required_signals"]:
            raise ValueError("Incomplete telemetry.")
        diagnostic = telemetry["motion_diagnostics"]
        motion_path = Path(diagnostic["motion_file"]["path"])
        mb = file_binding(motion_path)
        if mb["sha256"] != diagnostic["motion_file"]["sha256"]:
            raise ValueError("Motion file hash differs from telemetry.")
        inputs.extend([file_binding(path), file_binding(tp), mb])
        mode = diagnostic["mode"]
        if mode not in ("nominal_start", "randomized_start") or mode in cases:
            raise ValueError("Expected distinct nominal_start/randomized_start inputs.")
        with np.load(motion_path) as motion:
            obs, recorded, labels = reconstruct_observations(telemetry, motion, env_cfg)
        value, error, prediction = diagnose(actor, obs, recorded, labels, mode == "nominal_start", args.replay_atol)
        value["total_control_samples"] = len(telemetry["samples"])
        value["excluded_initial_history_samples"] = len(telemetry["samples"])-len(labels)
        value["completed_episode_ids"] = [s["episode_id"] for s in telemetry["samples"] if s["done"] and s["termination_terms"].get("motion_finished")]
        cases[mode] = value
        plot_data[mode] = (labels, error)
        for i, label in enumerate(labels):
            row = {"case": mode, **label, "mse_raw_action": float(np.mean(error[i]**2))}
            for j, name in enumerate(mirror.JOINT_NAMES):
                row[name+"_equivariance_error_raw"] = float(error[i, j])
                row[name+"_original_mean_action"] = float(prediction[i, j])
            csv_rows.append(row)
    if "nominal_start" not in cases:
        raise ValueError("A replay-verified nominal case is required for the primary diagnosis.")
    if any(file_binding(item["path"]) != item for item in inputs):
        raise ValueError("An input changed during analysis.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "samples.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(plot_data), figsize=(7*len(plot_data), 7), squeeze=False, constrained_layout=True)
    for col, (mode, (labels, error)) in enumerate(plot_data.items()):
        times = np.array([r["reference_time_s"] for r in labels])
        episodes = np.array([r["episode_id"] for r in labels])
        for ep in sorted(set(episodes)):
            mask = episodes == ep
            axes[0, col].plot(times[mask], np.degrees(.25*np.sqrt(np.mean(error[mask, :8]**2, axis=1))), label=f"Cycle {ep+1}", alpha=.7)
            axes[1, col].plot(times[mask], np.sqrt(np.mean(error[mask, 8:]**2, axis=1)), alpha=.7)
        for row in range(2):
            axes[row, col].axvspan(1.0, 1.45, color="orange", alpha=.12)
            axes[row, col].axvspan(1.45, 1.90, color="green", alpha=.10)
            axes[row, col].grid(alpha=.2)
        axes[0, col].set_title(mode + (" (verified replay)" if mode == "nominal_start" else " (noise-free proxy)"))
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_xlabel("Reference frame time (s)")
    axes[0, 0].set_ylabel("Mirror disagreement: position targets RMS (deg)")
    axes[1, 0].set_ylabel("Mirror disagreement: wheel velocity targets RMS (rad/s)")
    fig.suptitle("Frozen actor equivariance error; orange: takeoff, green: landing\nThese are policy target differences, not measured joint tracking errors")
    fig.savefig(args.output_dir / "phase-errors.png", dpi=150)
    plt.close(fig)
    output = {"version": 1, "kind": "offline_policy_mirror_diagnosis", "iteration": iteration,
              "inputs": inputs, "phases_reference_time_s": PHASES,
              "source_snapshots": {str(path.resolve()): path.read_text() for path in (Path(__file__), MAPPING)},
              "equation": "mu(M_obs(o)) - M_action(mu(o)); raw action MSE over samples and ten actions",
              "mapping_contract": {"policy_terms": mirror.POLICY_TERMS, "history_length": 10,
                                   "joint_action_order": mirror.JOINT_NAMES, "critic_body_order": mirror.BODY_NAMES},
              "checkpoint_exploration_std_raw": std,
              "limitations": ["Randomized telemetry lacks realized observation noise and randomized default joint positions.",
                              "Randomized numbers are noise-free probes at nominal default positions, not exact replay.",
                              "The first ten actions of every episode are excluded; histories never cross resets.",
                              "Nominal cycles repeat; one rollout seed does not establish population performance.",
                              "A mirrored randomized observation represents mirrored latent physical parameters too.",
                              "Mirror errors concern mean action targets, not torque symmetry or closed-loop improvement."],
              "cases": cases, "inputs_unchanged": True,
              "outputs": [file_binding(args.output_dir / name) for name in ("samples.csv", "phase-errors.png")]}
    with (args.output_dir / "diagnostic.json").open("x") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"output": str(args.output_dir.resolve()), "cases": {
        k: {"replay": v["recorded_action_comparison"], "phases": {
            p: {key: stats[key] for key in ("count", "mse_raw_action", "position_target_rms_deg", "wheel_velocity_target_rms_rad_s")}
            for p, stats in v["phases"].items()}} for k, v in cases.items()}}, indent=2))


if __name__ == "__main__":
    main()

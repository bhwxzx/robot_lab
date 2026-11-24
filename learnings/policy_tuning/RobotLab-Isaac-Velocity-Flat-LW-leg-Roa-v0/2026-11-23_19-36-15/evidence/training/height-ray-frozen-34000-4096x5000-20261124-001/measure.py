"""Bounded frozen-policy diagnostic of the training height-reward fallback."""
import copy
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO = Path.cwd().resolve()
OUT = Path(os.environ.get("HEIGHT_RAY_OUTPUT", Path(__file__).resolve().parent)).resolve()
TASK = "RobotLab-Isaac-Velocity-Flat-LW-leg-Roa-v0"
RUN = "2026-11-23_19-36-15"
N = 4096
STEPS = 5000
SEED = 42
RUN_DIR = REPO / "logs/rsl_rl/LW_leg_flat_roa" / RUN
CHECKPOINT = RUN_DIR / "model_34000.pt"
EXPECTED = "ca0269c71f3b983efb7fb8f3643d037c18d36b44afa557da7f09cff4d682f217"
PROV = REPO / "learnings/policy_tuning" / TASK / RUN / "provenance"
CONTEXT = PROV / "context-ff5942de4a7185cc2ff082e70aeb491f32fedf81b3af4aec9882f2201791e38a.json"
CONFIG = PROV / "config-dd9a8f90e060a9c634d8995a65ecced6b5f7d59b70465a2e692f333e3a83bd45.json"
FIELDS = ("invalid_environments", "invalid_rays", "nan_rays", "infinite_rays",
          "finite_out_of_range_rays", "nonzero_raw_height_terms", "nonfinite_raw_height_terms",
          "reset_environments", "timeout_environments")
sys.path.insert(0, str(REPO / ".agents/skills/monitor-tune-isaaclab-training/scripts"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_hash(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode() + b"\x00")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((array.dtype, array.shape)).encode() + b"\x00")
        digest.update(array.tobytes())
    return digest.hexdigest()


def compare(saved, live, path=""):
    if isinstance(saved, dict) and isinstance(live, dict):
        delta = []
        for key in sorted(saved.keys() | live.keys()):
            if key not in saved or key not in live:
                delta.append((path + "/" + key, key in saved, key in live))
            else:
                delta += compare(saved[key], live[key], path + "/" + key)
        return delta
    if saved != live:
        return [(path, saved, live)]
    return []


def checked_inputs():
    from capture_effective_training_config import validate_effective_config_evidence
    from evidence_provenance import read_reference

    def bound(path):
        return read_reference({"path": str(path), "sha256": path.stem.split("-", 1)[1]})
    ctx, cfg = bound(CONTEXT), bound(CONFIG)
    validate_effective_config_evidence(cfg, ctx)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == ctx["source"]["head"]
    assert all(sha(REPO / item["path"]) == item["sha256"] for item in ctx["config_files"])
    assert ctx["task"] == TASK and ctx["seed"] == SEED and sha(CHECKPOINT) == EXPECTED
    for info in cfg["source_files"].values():
        assert sha(info["path"]) == info["sha256"]
    return ctx, cfg


def run(ctx, provenance):
    import gymnasium as gym
    import hydra
    import torch
    from omegaconf import OmegaConf
    from isaaclab.envs.utils.spaces import replace_strings_with_env_cfg_spaces
    from isaaclab.utils import replace_strings_with_slices
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.hydra import register_task_to_hydra
    from rsl_rl.runners import OnPolicyRunnerROA
    import robot_lab.tasks
    from robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards import base_height_l2

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    with hydra.initialize(version_base="1.3", config_path=None):
        env_cfg, agent_cfg = register_task_to_hydra(TASK, "rsl_rl_cfg_entry_point")
        overrides = ctx["training"]["hydra_overrides"]
        resolved = replace_strings_with_slices(OmegaConf.to_container(hydra.compose(config_name=TASK, overrides=overrides), resolve=True))
    env_cfg.from_dict(resolved["env"])
    env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
    agent_cfg.from_dict(resolved["agent"])
    env_cfg.scene.num_envs, env_cfg.seed, env_cfg.log_dir = N, SEED, str(OUT)
    agent_cfg.logger = "wandb"
    agent_cfg.wandb_project = agent_cfg.neptune_project = "LW_leg_locomotion_roa"
    assert agent_cfg.seed == SEED and agent_cfg.class_name == "OnPolicyRunnerROA"
    assert env_cfg.scene.terrain.terrain_type == "generator"
    print("Creating 4096 original training environments", flush=True)
    gym_env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    try:
        saved = {}
        differences = {}
        for key, obj in (("env", env_cfg), ("agent", agent_cfg)):
            saved[key] = yaml.load((RUN_DIR / "params" / (key + ".yaml")).read_text(), Loader=yaml.BaseLoader)
            live = yaml.load(yaml.dump(obj.to_dict(), default_flow_style=False, sort_keys=False), Loader=yaml.BaseLoader)
            differences[key] = compare(saved[key], live)
        print("Resolved configuration differences:", json.dumps(differences, ensure_ascii=False), flush=True)
        assert [d[0] for d in differences["env"]] == ["/log_dir"] and not differences["agent"], differences
        runner = OnPolicyRunnerROA(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(str(CHECKPOINT), load_optimizer=False, map_location=agent_cfg.device)
        model = runner.alg.policy
        runner.eval_mode()
        model.requires_grad_(False)
        initial_weights = freeze_hash(model)
        next_it = runner.current_learning_iteration
        velocity_prob = runner.alg.estimated_velocity_probability()
        assert velocity_prob == 1.0 and not model.is_recurrent
        assert not model.actor_obs_normalization and not model.critic_obs_normalization
        rollout = runner.num_steps_per_env
        period = runner.alg.dagger_update_freq
        assert (next_it, rollout, period) == (34001, 24, 20)
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=env.max_episode_length)
        init_len = [int(env.episode_length_buf.min()), int(env.episode_length_buf.max())]
        assert raw.num_envs == N and raw.step_dt == 0.02 and raw.max_episode_length == 1000
        term = copy.copy(raw.reward_manager.get_term_cfg("base_height_l2"))
        original = term.func
        assert original is base_height_l2 and term.weight == -10 and term.params["target_height"] == .69
        assert term.params["sensor_cfg"].name == "height_scanner_base"
        rows = torch.empty((STEPS, len(FIELDS)), dtype=torch.int64, device=raw.device)
        affected_per_env = torch.zeros(N, dtype=torch.int64, device=raw.device)
        calls = 0
        n_rays = 0
        first_counter = raw.common_step_counter

        def observe(environment, **params):
            nonlocal calls, n_rays
            assert environment.common_step_counter == first_counter + calls + 1
            z = environment.scene.sensors[params["sensor_cfg"].name].data.ray_hits_w[..., 2]
            if calls == 0:
                assert z.ndim == 2 and z.shape[0] == N
                n_rays = z.shape[1]
                print(f"Observed {N} x {n_rays} ray z coordinates per reward call", flush=True)
            assert z.shape == (N, n_rays)
            nan = torch.isnan(z)
            inf = torch.isinf(z)
            large = (z.abs() > 1.e6) & (~inf)
            invalid = nan | inf | large
            bad_env = invalid.any(dim=1)
            reward = original(environment, **params)
            values = (bad_env.sum(), invalid.sum(), nan.sum(), inf.sum(), large.sum(),
                      torch.count_nonzero(reward), (~torch.isfinite(reward)).sum(),
                      environment.reset_buf.sum(), environment.reset_time_outs.sum())
            rows[calls] = torch.stack(values)
            affected_per_env.add_(bad_env)
            calls += 1
            return reward

        term.func = observe
        raw.reward_manager.set_term_cfg("base_height_l2", term)
        states = env.get_observations().to(runner.device)
        start = time.monotonic()
        with torch.inference_mode():
            for step in range(STEPS):
                hist = (next_it + step // rollout) % period == 0
                actions = runner.alg.act(states, hist_encoding=hist)
                states, _, dones, _ = env.step(actions)
                model.reset(dones)
                if (step + 1) % 500 == 0:
                    batch_hits = int((rows[:step + 1, 0] > 0).sum())
                    print(f"Progress {step + 1}/{STEPS}, batch fallbacks {batch_hits}, elapsed {time.monotonic()-start:.1f}s", flush=True)
        data = rows.cpu().numpy()
        counts = affected_per_env.cpu().numpy()
        assert calls == STEPS and n_rays == 9
        assert (data[:, 1] == data[:, 2:5].sum(axis=1)).all()
        assert (data[:, 0] <= data[:, 1]).all() and (data[:, 1] <= n_rays * data[:, 0]).all()
        hit = data[:, 0] > 0
        assert (data[hit, 5] == 0).all() and (data[:, 6] == 0).all()
        assert counts.sum() == data[:, 0].sum()
        final_weights = freeze_hash(model)
        assert final_weights == initial_weights and sha(CHECKPOINT) == EXPECTED
        step_file = OUT / "control-steps.csv.gz"
        with gzip.open(step_file, "xt", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("control_step_1based", "controller", *FIELDS))
            for i, row in enumerate(data):
                hist = (next_it + i // rollout) % period == 0
                writer.writerow((i + 1, "student" if hist else "teacher", *map(int, row)))
        b, e, r = int(hit.sum()), int(data[:, 0].sum()), int(data[:, 1].sum())
        collateral = b * N - e
        report = {
            "status": "completed", "kind": "frozen_training_scale_height_ray_diagnostic",
            "task": TASK, "run_id": RUN, "checkpoint": str(CHECKPOINT), "checkpoint_sha256": EXPECTED,
            "host_id": ctx["host_id"], "git_head": ctx["source"]["head"],
            "context_sha256": sha(CONTEXT), "config_sha256": sha(CONFIG),
            "effective_config_fingerprint": provenance["fingerprints"]["effective_config"],
            "diagnostic_source_sha256": sha(__file__), "per_control_step_data_sha256": sha(step_file),
            "live_config_differences": differences, "inputs": {k: v["sha256"] for k, v in provenance["source_files"].items()},
            "source_model_state_before_sha256": initial_weights, "source_model_state_after_sha256": final_weights,
            "seed": SEED, "controller": "frozen stochastic ROAPPO.act; training teacher/student schedule",
            "next_iteration": next_it, "rollout_length": rollout, "student_period": period,
            "student_control_steps": sum((next_it + i // rollout) % period == 0 for i in range(STEPS)),
            "estimated_velocity_probability": velocity_prob, "optimizer_steps": 0,
            "initial_episode_length_range_steps": init_len, "environment_count": N, "step_dt_s": raw.step_dt,
            "terrain": {n: v.proportion for n, v in env_cfg.scene.terrain.terrain_generator.sub_terrains.items()},
            "raw_height_reward_weight": term.weight, "raw_height_target_m": term.params["target_height"],
            "measurement_phase": "inside original base_height_l2 call, post-physics and before reset",
            "control_steps": calls, "rays_per_environment": n_rays, "environment_steps": calls * N,
            "ray_samples": calls * N * n_rays, "batch_fallback_steps": b, "batch_fallback_fraction": b / calls,
            "invalid_environment_steps": e, "invalid_environment_fraction": e / (calls * N),
            "invalid_rays": r, "invalid_ray_fraction": r / (calls * N * n_rays),
            "nan_rays": int(data[:, 2].sum()), "infinite_rays": int(data[:, 3].sum()),
            "finite_out_of_range_rays": int(data[:, 4].sum()),
            "collaterally_masked_environment_steps": collateral, "collaterally_masked_fraction": collateral / (calls * N),
            "environments_ever_invalid": int((counts > 0).sum()), "max_invalid_environments_on_any_step": int(data[:, 0].max()),
            "nonzero_raw_height_term_environment_steps": int(data[:, 5].sum()), "nonfinite_raw_height_term_environment_steps": int(data[:, 6].sum()),
            "environment_resets": int(data[:, 7].sum()), "environment_timeouts": int(data[:, 8].sum()),
            "wall_seconds_for_rollout_and_output": time.monotonic() - start,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        return report
    finally:
        env.close()


def main():
    if (OUT / "result.json").exists() or (OUT / "control-steps.csv.gz").exists():
        raise FileExistsError("Output already has measurements.")
    ctx, provenance = checked_inputs()
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device="cuda:0").app
    try:
        report = run(ctx, provenance)
        with (OUT / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        print("Completed", json.dumps({k: report[k] for k in ("control_steps", "ray_samples", "batch_fallback_steps", "invalid_environment_steps", "collaterally_masked_environment_steps")}), flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()

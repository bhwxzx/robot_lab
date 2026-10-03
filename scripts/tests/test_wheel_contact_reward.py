"""CPU wheel-support reward contracts without importing or launching Isaac Sim.

Run with: conda run -n isaacsim-5.1 python -B scripts/tests/test_wheel_contact_reward.py
Compile the production reward, shared registration and LW Leg overrides directly
from source; only configuration/environment/sensor interfaces are stubbed.
"""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
BODY_NAMES = ["base_link", "left_wheel_link", "right_foot_link", "left_foot_link", "right_wheel_link"]


def load_reward_and_config():
    """Execute production source without the SDK's simulation-dependent imports."""
    reward_path = TASK_ROOT / "mdp/rewards.py"
    tree = ast.parse(reward_path.read_text(), filename=str(reward_path))
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        or (isinstance(node, ast.FunctionDef) and node.name == "wheel_contact_continuous")
    ]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reward_path), "exec"), namespace)
    reward = namespace["wheel_contact_continuous"]

    shared_path = TASK_ROOT / "tracking_env_cfg.py"
    tree = ast.parse(shared_path.read_text(), filename=str(shared_path))
    rewards_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RewardsCfg")
    registration = next(
        node for node in rewards_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "wheel_contact_continuous" for target in node.targets)
    )
    namespace = {
        "mdp": SimpleNamespace(wheel_contact_continuous=reward),
        "RewTerm": SimpleNamespace,
        "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, body_names=None, **kwargs),
    }
    exec(compile(ast.Module(body=[registration], type_ignores=[]), str(shared_path), "exec"), namespace)
    default_cfg = namespace["wheel_contact_continuous"]

    config_path = TASK_ROOT / "config/LW_Leg/flat_env_cfg.py"
    tree = ast.parse(config_path.read_text(), filename=str(config_path))
    config_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LWLegBeyondMimicFlatEnvCfg"
    )
    post_init = next(
        node for node in config_class.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    overrides = [
        node for node in post_init.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "wheel_contact_continuous" for child in ast.walk(node)
        )
    ]
    # IsaacLab config instances copy nested defaults before their overrides.
    config = SimpleNamespace(rewards=SimpleNamespace(wheel_contact_continuous=copy.deepcopy(default_cfg)))
    exec(
        compile(ast.Module(body=overrides, type_ignores=[]), str(config_path), "exec"),
        {"self": config},
    )
    return reward, default_cfg, config.rewards.wheel_contact_continuous


REWARD, BASE_REWARD_CFG, REWARD_CFG = load_reward_and_config()


class WheelContactRewardTests(unittest.TestCase):
    def setUp(self):
        self.wheel_ids = [BODY_NAMES.index(name) for name in REWARD_CFG.params["sensor_cfg"].body_names]
        self.sensor_data = SimpleNamespace(
            current_contact_time=torch.zeros(3, len(BODY_NAMES)),
            net_forces_w=torch.zeros(3, len(BODY_NAMES), 3),
        )
        self.command = SimpleNamespace(time_steps=torch.full((3,), 100), motion=SimpleNamespace(fps=np.array([50.0])))
        self.env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: {"motion": self.command}[name]),
            scene=SimpleNamespace(sensors={"contact_forces": SimpleNamespace(data=self.sensor_data)}),
            episode_length_buf=torch.zeros(3, dtype=torch.long),
        )
        self.params = {
            **REWARD_CFG.params,
            "sensor_cfg": SimpleNamespace(name="contact_forces", body_ids=self.wheel_ids),
        }
        self.set_contact([0.1, 0.1])

    def set_contact(self, times, vertical_forces=60.0):
        self.sensor_data.current_contact_time[:, self.wheel_ids] = torch.as_tensor(times, dtype=torch.float32)
        self.sensor_data.net_forces_w[:, self.wheel_ids, 2] = torch.as_tensor(vertical_forces, dtype=torch.float32)

    def reward(self, **overrides):
        return REWARD(self.env, **{**self.params, **overrides})

    def test_shared_term_disabled_and_lw_overrides_preserve_shared_defaults(self):
        self.assertIs(BASE_REWARD_CFG.func, REWARD)
        self.assertEqual(BASE_REWARD_CFG.weight, 0.0)
        self.assertEqual(BASE_REWARD_CFG.params["sensor_cfg"].name, "contact_forces")
        self.assertIsNone(BASE_REWARD_CFG.params["sensor_cfg"].body_names)
        self.assertIsNot(BASE_REWARD_CFG.params, REWARD_CFG.params)
        self.assertIsNot(BASE_REWARD_CFG.params["sensor_cfg"], REWARD_CFG.params["sensor_cfg"])
        self.assertEqual(REWARD_CFG.weight, 0.5)
        self.assertEqual(REWARD_CFG.params["start_time_s"], 1.6)
        self.assertEqual(REWARD_CFG.params["min_contact_force"], 10.0)
        self.assertEqual(REWARD_CFG.params["stable_contact_time"], 0.1)

    def test_config_selects_sensor_wheels_and_uses_resolved_body_indices(self):
        self.assertIs(REWARD_CFG.func, REWARD)
        self.assertEqual(REWARD_CFG.weight, 0.5)
        self.assertEqual(self.wheel_ids, [4, 1])
        # Other bodies must not contribute to the support reward.
        self.sensor_data.net_forces_w[:, [0, 2, 3], 2] = 1000.0
        self.sensor_data.current_contact_time[:, [0, 2, 3]] = 10.0
        self.set_contact([0.1, 0.0], [60.0, 0.0])
        torch.testing.assert_close(self.reward(), torch.zeros(3))

    def test_reference_phase_boundary_independent_of_episode_age(self):
        self.command.time_steps[:] = torch.tensor([79, 80, 81])
        self.env.episode_length_buf[:] = torch.tensor([10000, 0, 1])
        torch.testing.assert_close(self.reward(), torch.tensor([0.0, 1.0, 1.0]))

    def test_phase_uses_reference_fps_and_rounds_start_frame_up(self):
        self.command.motion.fps = np.array([60.0])
        self.command.time_steps[:] = torch.tensor([95, 96, 97])
        torch.testing.assert_close(self.reward(), torch.tensor([0.0, 1.0, 1.0]))
        self.command.motion.fps = np.array([50.0])
        self.command.time_steps[:] = torch.tensor([79, 80, 81])
        torch.testing.assert_close(self.reward(start_time_s=1.601), torch.tensor([0.0, 0.0, 1.0]))

    def test_airborne_and_pre_phase_contacts_earn_zero(self):
        self.command.time_steps.fill_(75)
        self.set_contact([1.0, 1.0], 1000.0)
        torch.testing.assert_close(self.reward(), torch.zeros(3))
        self.command.time_steps.fill_(100)
        self.set_contact([0.0, 0.0], 0.0)
        torch.testing.assert_close(self.reward(), torch.zeros(3))

    def test_shorter_contact_duration_ramps_and_saturates(self):
        self.set_contact([[0.005, 0.2], [0.05, 0.09], [0.2, 0.1]])
        torch.testing.assert_close(self.reward(), torch.tensor([0.05, 0.5, 1.0]))

    def test_alternating_single_wheel_support_earns_zero(self):
        # Nonzero cached timers must never override the current force gate.
        self.set_contact([1.0, 1.0], [[60.0, 0.0], [0.0, 60.0], [60.0, 60.0]])
        torch.testing.assert_close(self.reward(), torch.tensor([0.0, 0.0, 1.0]))
        self.set_contact([1.0, 1.0], [[0.0, 60.0], [60.0, 0.0], [60.0, 60.0]])
        torch.testing.assert_close(self.reward(), torch.tensor([0.0, 0.0, 1.0]))

    def test_only_positive_vertical_support_above_threshold_counts(self):
        self.set_contact([1.0, 1.0], [[10.0, 60.0], [-60.0, 60.0], [10.01, 10.01]])
        self.sensor_data.net_forces_w[:, self.wheel_ids, 0] = 1000.0
        torch.testing.assert_close(self.reward(), torch.tensor([0.0, 0.0, 1.0]))

    def test_large_force_does_not_increase_reward_or_bypass_duration(self):
        self.set_contact([0.005, 0.005], 60.0)
        short_reward = self.reward()
        self.set_contact([0.005, 0.005], 3000.0)
        torch.testing.assert_close(self.reward(), short_reward)
        torch.testing.assert_close(short_reward, torch.full((3,), 0.05))
        self.set_contact([5.0, 5.0], 3000.0)
        torch.testing.assert_close(self.reward(), torch.ones(3))

    def test_contact_loss_and_recontact_restart_ramp(self):
        torch.testing.assert_close(self.reward(), torch.ones(3))
        # Sensor timers clear on loss of contact, at physics-step resolution.
        self.set_contact([0.0, 0.2], [0.0, 60.0])
        torch.testing.assert_close(self.reward(), torch.zeros(3))
        self.set_contact([0.005, 0.205])
        torch.testing.assert_close(self.reward(), torch.full((3,), 0.05))

    def test_partial_sensor_reset_and_calls_do_not_leak_or_mutate_state(self):
        self.sensor_data.current_contact_time[1] = 0.0
        self.sensor_data.net_forces_w[1] = 0.0
        before_time = self.sensor_data.current_contact_time.clone()
        before_force = self.sensor_data.net_forces_w.clone()
        before_frames = self.command.time_steps.clone()
        for _ in range(3):
            torch.testing.assert_close(self.reward(), torch.tensor([1.0, 0.0, 1.0]))
        torch.testing.assert_close(self.sensor_data.current_contact_time, before_time, rtol=0, atol=0)
        torch.testing.assert_close(self.sensor_data.net_forces_w, before_force, rtol=0, atol=0)
        torch.testing.assert_close(self.command.time_steps, before_frames, rtol=0, atol=0)
        self.sensor_data.current_contact_time[1, self.wheel_ids] = 0.005
        self.sensor_data.net_forces_w[1, self.wheel_ids, 2] = 60.0
        torch.testing.assert_close(self.reward(), torch.tensor([1.0, 0.05, 1.0]))

    def test_invalid_parameters_fail_explicitly(self):
        for overrides in (
            {"start_time_s": -0.1},
            {"min_contact_force": -1.0},
            {"stable_contact_time": 0.0},
            {"stable_contact_time": -0.1},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.reward(**overrides)

    def test_missing_timer_or_incorrect_body_selection_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            self.reward(sensor_cfg=SimpleNamespace(name="contact_forces", body_ids=[4]))
        self.sensor_data.current_contact_time = None
        with self.assertRaisesRegex(RuntimeError, "track_air_time"):
            self.reward()


if __name__ == "__main__":
    unittest.main(verbosity=2)

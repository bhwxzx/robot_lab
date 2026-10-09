"""CPU roll and reference-phase reward contracts without launching Isaac Sim.

Run with: conda run -n isaacsim-5.1 python -B scripts/tests/test_motion_roll_reward.py
Execute production functions and LW Leg configuration assignments from their AST;
only the simulation-dependent environment and configuration interfaces are stubbed.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
PHASE_TERMS = (
    "motion_takeoff_pitch_ang_vel",
    "motion_takeoff_anchor_ori",
    "wheel_contact_continuous",
    "leg_symmetry",
    "same_foot_x_position",
)


def load_rewards_and_config():
    reward_path = TASK_ROOT / "mdp/rewards.py"
    tree = ast.parse(reward_path.read_text(), filename=str(reward_path))
    functions = {
        "motion_anchor_roll_horizontal_error_exp",
        "_motion_phase_window_weight",
        "_wheel_phase_geometry_weight",
    }
    nodes = [
        node for node in tree.body
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(reward_path), "exec"), namespace)

    config_path = TASK_ROOT / "config/LW_Leg/flat_env_cfg.py"
    tree = ast.parse(config_path.read_text(), filename=str(config_path))
    rewards_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LWLegRewardsCfg")
    registration = next(
        node for node in rewards_class.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "motion_anchor_roll_horizontal" for target in node.targets)
    )
    config_namespace = {
        "RewTerm": SimpleNamespace,
        "mdp": SimpleNamespace(motion_anchor_roll_horizontal_error_exp=namespace["motion_anchor_roll_horizontal_error_exp"]),
    }
    exec(compile(ast.Module(body=[registration], type_ignores=[]), str(config_path), "exec"), config_namespace)

    config_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LWLegBeyondMimicFlatEnvCfg"
    )
    post_init = next(node for node in config_class.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    overrides = [
        node for node in post_init.body
        if isinstance(node, ast.Assign)
        and any(isinstance(child, ast.Attribute) and child.attr in (*PHASE_TERMS, "motion_file") for child in ast.walk(node))
    ]
    config = SimpleNamespace(
        rewards=SimpleNamespace(**{
            name: SimpleNamespace(params={"asset_cfg": SimpleNamespace(), "sensor_cfg": SimpleNamespace()})
            for name in PHASE_TERMS
        }),
        commands=SimpleNamespace(motion=SimpleNamespace()),
    )
    exec(compile(ast.Module(body=overrides, type_ignores=[]), str(config_path), "exec"), {"self": config})
    return namespace, config_namespace["motion_anchor_roll_horizontal"], config


FUNCTIONS, ROLL_CFG, CONFIG = load_rewards_and_config()
REWARD = FUNCTIONS["motion_anchor_roll_horizontal_error_exp"]


def quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """Construct normalized wxyz quaternions for Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = [torch.as_tensor(angle, dtype=torch.float64) / 2.0 for angle in (roll, pitch, yaw)]
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    return torch.stack((
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ), dim=-1).reshape(-1, 4)


def make_env(quaternions):
    command = SimpleNamespace(robot_anchor_quat_w=quaternions)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: {"motion": command}[name]))
    return env, command


class MotionRollRewardTests(unittest.TestCase):
    def test_local_registration_targets_motion_and_uses_radians(self):
        self.assertIs(ROLL_CFG.func, REWARD)
        self.assertEqual(ROLL_CFG.weight, 0.5)
        self.assertEqual(ROLL_CFG.params, {"command_name": "motion", "std": 0.1})

    def test_level_roll_allows_pitch_and_yaw(self):
        env, _ = make_env(quaternion(pitch=[0.0, 0.3, -0.8, 1.0], yaw=[0.0, -1.2, math.pi, 2.6]))
        torch.testing.assert_close(REWARD(env, **ROLL_CFG.params), torch.ones(4, dtype=torch.float64))

    def test_signed_roll_symmetry_and_known_angular_scale(self):
        env, _ = make_env(quaternion(roll=[-0.2, -0.1, 0.0, 0.1, 0.2]))
        expected = torch.tensor([math.exp(-4.0), math.exp(-1.0), 1.0, math.exp(-1.0), math.exp(-4.0)])
        torch.testing.assert_close(REWARD(env, **ROLL_CFG.params), expected.to(torch.float64))

    def test_combined_pitch_and_yaw_preserve_roll_penalty(self):
        rolls = [-0.15, 0.05, 0.1, -0.05]
        pure, _ = make_env(quaternion(roll=rolls))
        combined, _ = make_env(quaternion(roll=rolls, pitch=[0.4, -0.6, 1.0, -1.0], yaw=[-2.4, 1.1, math.pi, 0.8]))
        torch.testing.assert_close(REWARD(combined, **ROLL_CFG.params), REWARD(pure, **ROLL_CFG.params))

    def test_quaternion_sign_does_not_change_reward(self):
        env, command = make_env(quaternion(roll=[-0.1, 0.2], pitch=[0.3, -0.4], yaw=[1.0, -2.0]))
        original = REWARD(env, **ROLL_CFG.params)
        command.robot_anchor_quat_w = -command.robot_anchor_quat_w
        torch.testing.assert_close(REWARD(env, **ROLL_CFG.params), original)

    def test_world_level_target_is_independent_of_reference_and_phase(self):
        env, command = make_env(quaternion(roll=[0.05] * 6))
        original = REWARD(env, **ROLL_CFG.params)
        command.anchor_quat_w = quaternion(roll=[0.8] * 6)
        command.time_steps = torch.tensor([0, 50, 65, 78, 80, 166])
        command.motion = SimpleNamespace(fps=torch.tensor([50.0]))
        env.episode_length_buf = torch.tensor([10000, 0, 1, 100, 200, 300])
        torch.testing.assert_close(REWARD(env, **ROLL_CFG.params), original)

    def test_reward_is_bounded_and_gradient_restores_level_roll(self):
        rolls = torch.tensor([-0.1, 0.0, 0.1], dtype=torch.float64, requires_grad=True)
        env, _ = make_env(quaternion(roll=rolls))
        reward = REWARD(env, **ROLL_CFG.params)
        self.assertTrue(torch.all((reward >= 0.0) & (reward <= 1.0)))
        reward.sum().backward()
        self.assertTrue(torch.isfinite(rolls.grad).all())
        self.assertGreater(rolls.grad[0].item(), 0.0)
        self.assertEqual(rolls.grad[1].item(), 0.0)
        self.assertLess(rolls.grad[2].item(), 0.0)

    def test_invalid_std_fails_explicitly(self):
        env, _ = make_env(quaternion())
        for std in (0.0, -0.1, math.nan, math.inf, -math.inf):
            with self.subTest(std=std), self.assertRaisesRegex(ValueError, "finite, positive"):
                REWARD(env, command_name="motion", std=std)


class ReferenceTimingTests(unittest.TestCase):
    def setUp(self):
        self.env, self.command = make_env(quaternion())
        self.command.motion = SimpleNamespace(fps=torch.tensor([50.0]))

    def test_selected_reference_and_all_five_time_boundaries(self):
        self.assertEqual(
            CONFIG.commands.motion.motion_file,
            "source/robot_lab/robot_lab/datasets/LW/motion_beyondmimic/se3_trajopt/2026-10-10/leg_to_wheel_transform_50hz.npz",
        )
        for name in PHASE_TERMS:
            with self.subTest(term=name):
                params = getattr(CONFIG.rewards, name).params
                key = "end_time_s" if name.startswith("motion_takeoff") else "start_time_s"
                self.assertEqual(params[key], 1.6)

    def test_takeoff_window_preserves_entry_and_finishes_at_frame_80(self):
        self.command.time_steps = torch.tensor([49, 50, 52, 55, 75, 77, 79, 80, 81])
        for name in PHASE_TERMS[:2]:
            with self.subTest(term=name):
                params = getattr(CONFIG.rewards, name).params
                blend = FUNCTIONS["_motion_phase_window_weight"](
                    self.env, command_name="motion", **{key: params[key] for key in ("start_time_s", "end_time_s", "ramp_time_s")}
                )
                torch.testing.assert_close(blend, torch.tensor([0.0, 0.0, 0.352, 1.0, 1.0, 0.648, 0.104, 0.0, 0.0]), atol=1e-6, rtol=1e-5)

    def test_wheel_geometry_ramp_begins_after_landing(self):
        self.command.time_steps = torch.tensor([77, 79, 80, 85, 90, 91])
        self.env.scene = {"robot": SimpleNamespace(data=SimpleNamespace(body_link_pos_w=torch.zeros(6, 2, 3)))}
        for name in ("leg_symmetry", "same_foot_x_position"):
            with self.subTest(term=name):
                params = getattr(CONFIG.rewards, name).params
                blend = FUNCTIONS["_wheel_phase_geometry_weight"](
                    self.env, command_name="motion", asset_cfg=SimpleNamespace(name="robot", body_ids=[0, 1]),
                    start_time_s=params["start_time_s"], ramp_time_s=params["ramp_time_s"],
                )
                torch.testing.assert_close(blend, torch.tensor([0.0, 0.0, 0.0, 0.5, 1.0, 1.0]), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

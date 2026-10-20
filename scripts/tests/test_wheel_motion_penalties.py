"""CPU contracts for post-landing wheel penalties; no Isaac Sim launch required."""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


TASK_ROOT = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
TERMS = ("joint_vel_wheel_l2", "joint_acc_wheel_l2")


def load_functions_and_config():
    path = TASK_ROOT / "mdp/rewards.py"
    names = {"_wheel_phase_weight", "wheel_phase_joint_velocity_l2", "wheel_phase_joint_acceleration_l2"}
    nodes = [n for n in ast.parse(path.read_text()).body
             if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
             or (isinstance(n, ast.FunctionDef) and n.name in names)]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

    path = TASK_ROOT / "config/LW_Leg/flat_env_cfg.py"
    tree = ast.parse(path.read_text())
    rewards = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LWLegRewardsCfg")
    registrations = [n for n in rewards.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in TERMS for t in n.targets)]
    config_namespace = {
        "mdp": SimpleNamespace(**{n: namespace[n] for n in names}),
        "RewTerm": SimpleNamespace,
        "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
    }
    exec(compile(ast.Module(body=registrations, type_ignores=[]), str(path), "exec"), config_namespace)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LWLegBeyondMimicFlatEnvCfg")
    post_init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
    overrides = [n for n in post_init.body if isinstance(n, ast.Assign)
                 and any(isinstance(c, ast.Attribute) and c.attr in TERMS for c in ast.walk(n))]
    config = SimpleNamespace(
        rewards=SimpleNamespace(**{n: copy.deepcopy(config_namespace[n]) for n in TERMS}),
        wheel_joint_names=["right_wheel_joint", "left_wheel_joint"],
    )
    exec(compile(ast.Module(body=overrides, type_ignores=[]), str(path), "exec"), {"self": config})
    return namespace, config.rewards


FUNCTIONS, CONFIG = load_functions_and_config()


class WheelMotionPenaltyTests(unittest.TestCase):
    def setUp(self):
        # Noncontiguous indices ensure leg joints cannot leak into the penalties.
        self.data = SimpleNamespace(joint_vel=torch.zeros(4, 5), joint_acc=torch.zeros(4, 5))
        self.command = SimpleNamespace(time_steps=torch.full((4,), 100), motion=SimpleNamespace(fps=torch.tensor(50.0)))
        self.env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=self.data)},
            command_manager=SimpleNamespace(get_term=lambda name: {"motion": self.command}[name]),
            episode_length_buf=torch.tensor([0, 1, 100, 10000]),
        )

    def penalty(self, name, **overrides):
        cfg = getattr(CONFIG, name)
        return cfg.func(self.env, **{**cfg.params, "asset_cfg": SimpleNamespace(name="robot", joint_ids=[4, 1]), **overrides})

    def test_config_enables_only_two_wheels_with_negative_weights(self):
        for name, function, scale, value, weight in (
            (TERMS[0], "wheel_phase_joint_velocity_l2", "velocity_scale", 10.0, -0.05),
            (TERMS[1], "wheel_phase_joint_acceleration_l2", "acceleration_scale", 100.0, -0.02),
        ):
            cfg = getattr(CONFIG, name)
            self.assertIs(cfg.func, FUNCTIONS[function])
            self.assertEqual(cfg.weight, weight)
            self.assertEqual(cfg.params[scale], value)
            self.assertEqual(cfg.params["asset_cfg"].joint_names, ["right_wheel_joint", "left_wheel_joint"])
            self.assertEqual(cfg.params["start_time_s"], 1.65)
            self.assertEqual(cfg.params["ramp_time_s"], 0.2)

    def test_zero_before_landing_and_smooth_ramp_independent_of_episode_age(self):
        self.command.time_steps[:] = torch.tensor([82, 83, 88, 93])
        self.data.joint_vel.fill_(10.0)
        self.data.joint_acc.fill_(100.0)
        expected = torch.tensor([0.0, 0.00725, 0.57475, 1.0])
        for name in TERMS:
            torch.testing.assert_close(self.penalty(name), expected, atol=1e-6, rtol=1e-5)

    def test_reference_fps_and_exact_start_boundary(self):
        self.command.motion.fps = torch.tensor(100.0)
        self.command.time_steps[:] = torch.tensor([164, 165, 175, 185])
        self.data.joint_vel.fill_(10.0)
        torch.testing.assert_close(self.penalty(TERMS[0]), torch.tensor([0.0, 0.0, 0.5, 1.0]))

    def test_only_wheel_feedback_contributes_and_sign_does_not_matter(self):
        self.data.joint_vel.fill_(10000.0)
        self.data.joint_acc.fill_(100000.0)
        self.data.joint_vel[:, [4, 1]] = torch.tensor([10.0, -20.0])
        self.data.joint_acc[:, [4, 1]] = torch.tensor([-100.0, 200.0])
        for name in TERMS:
            torch.testing.assert_close(self.penalty(name), torch.full((4,), 2.5))
            self.assertTrue(torch.all(getattr(CONFIG, name).weight * self.penalty(name) < 0))

    def test_acceleration_uses_measured_acceleration_instead_of_speed(self):
        self.data.joint_vel.fill_(1000.0)
        torch.testing.assert_close(self.penalty(TERMS[1]), torch.zeros(4))
        self.data.joint_vel.zero_()
        self.data.joint_acc[:, [4, 1]] = 200.0
        torch.testing.assert_close(self.penalty(TERMS[1]), torch.full((4,), 4.0))

    def test_phase_reset_is_per_environment_and_calls_do_not_mutate_feedback(self):
        self.data.joint_vel.fill_(10.0)
        self.data.joint_acc.fill_(100.0)
        self.command.time_steps[1] = 0
        before_vel, before_acc = self.data.joint_vel.clone(), self.data.joint_acc.clone()
        for _ in range(3):
            for name in TERMS:
                torch.testing.assert_close(self.penalty(name), torch.tensor([1.0, 0.0, 1.0, 1.0]))
        torch.testing.assert_close(self.data.joint_vel, before_vel)
        torch.testing.assert_close(self.data.joint_acc, before_acc)

    def test_penalty_gradient_opposes_each_signed_wheel_velocity(self):
        self.data.joint_vel[:, [4, 1]] = torch.tensor([10.0, -10.0])
        self.data.joint_vel.requires_grad_()
        (CONFIG.joint_vel_wheel_l2.weight * self.penalty(TERMS[0])).sum().backward()
        self.assertTrue(torch.all(self.data.joint_vel.grad[:, 4] < 0))
        self.assertTrue(torch.all(self.data.joint_vel.grad[:, 1] > 0))
        self.assertTrue(torch.all(self.data.joint_vel.grad[:, [0, 2, 3]] == 0))

    def test_invalid_scales_timing_fps_and_joint_selection_fail(self):
        for name, scale in ((TERMS[0], "velocity_scale"), (TERMS[1], "acceleration_scale")):
            for value in (0.0, -1.0, math.nan, math.inf):
                with self.subTest(name=name, scale=value), self.assertRaises(ValueError):
                    self.penalty(name, **{scale: value})
            for params in ({"start_time_s": -1.0}, {"start_time_s": math.nan},
                           {"ramp_time_s": 0.0}, {"ramp_time_s": math.inf},
                           {"asset_cfg": SimpleNamespace(name="robot", joint_ids=[4])}):
                with self.subTest(name=name, params=params), self.assertRaises(ValueError):
                    self.penalty(name, **params)
        self.command.motion.fps = torch.tensor(0.0)
        with self.assertRaises(ValueError):
            self.penalty(TERMS[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

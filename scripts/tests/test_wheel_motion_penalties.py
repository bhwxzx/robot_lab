"""CPU wheel phase and penalty contracts, independent of active task rewards."""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch


TASK_ROOT = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
TERMS = ("joint_vel_wheel_l2", "joint_acc_wheel_l2")


def load_functions():
    path = TASK_ROOT / "mdp/rewards.py"
    names = {
        "_wheel_phase_weight", "_wheel_phase_geometry_weight", "wheel_phase_same_feet_x_position",
        "wheel_phase_joint_velocity_l2", "wheel_phase_joint_acceleration_l2",
    }
    nodes = [n for n in ast.parse(path.read_text()).body
             if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
             or (isinstance(n, ast.FunctionDef) and n.name in names)]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

    return namespace


FUNCTIONS = load_functions()
# Exercise the functions even when a task does not register these optional terms.
PARAMS = {
    TERMS[0]: {"velocity_scale": 10.0},
    TERMS[1]: {"acceleration_scale": 100.0},
}
PENALTIES = {
    TERMS[0]: FUNCTIONS["wheel_phase_joint_velocity_l2"],
    TERMS[1]: FUNCTIONS["wheel_phase_joint_acceleration_l2"],
}


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
        params = {
            "command_name": "motion", "start_time_s": 1.65, "ramp_time_s": 0.2,
            "asset_cfg": SimpleNamespace(name="robot", joint_ids=[4, 1]),
            **PARAMS[name], **overrides,
        }
        return PENALTIES[name](self.env, **params)

    def test_zero_ramp_starts_at_first_reference_frame_at_or_after_start(self):
        self.command.time_steps[:] = torch.tensor([82, 83, 100, 0])
        self.data.joint_vel.fill_(10.0)
        self.data.joint_acc.fill_(100.0)
        for name in TERMS:
            torch.testing.assert_close(
                self.penalty(name, ramp_time_s=0.0), torch.tensor([0.0, 1.0, 1.0, 0.0]),
            )

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
        (-self.penalty(TERMS[0])).sum().backward()
        self.assertTrue(torch.all(self.data.joint_vel.grad[:, 4] < 0))
        self.assertTrue(torch.all(self.data.joint_vel.grad[:, 1] > 0))
        self.assertTrue(torch.all(self.data.joint_vel.grad[:, [0, 2, 3]] == 0))

    def test_invalid_scales_timing_fps_and_joint_selection_fail(self):
        for name, scale in ((TERMS[0], "velocity_scale"), (TERMS[1], "acceleration_scale")):
            for value in (0.0, -1.0, math.nan, math.inf):
                with self.subTest(name=name, scale=value), self.assertRaises(ValueError):
                    self.penalty(name, **{scale: value})
            for params in ({"start_time_s": -1.0}, {"start_time_s": math.nan},
                           {"ramp_time_s": -0.1}, {"ramp_time_s": math.nan}, {"ramp_time_s": math.inf},
                           {"asset_cfg": SimpleNamespace(name="robot", joint_ids=[4])}):
                with self.subTest(name=name, params=params), self.assertRaises(ValueError):
                    self.penalty(name, **params)
        self.command.motion.fps = torch.tensor(0.0)
        with self.assertRaises(ValueError):
            self.penalty(TERMS[0])


class WheelPhaseWeightTests(unittest.TestCase):
    def setUp(self):
        self.command = SimpleNamespace(
            time_steps=torch.tensor([0, 52, 62, 63]), motion=SimpleNamespace(fps=torch.tensor(50.0)),
        )
        self.env = SimpleNamespace(
            command_manager=SimpleNamespace(get_term=lambda name: {"motion": self.command}[name]),
            episode_length_buf=torch.tensor([1000, 0, 1, 10000]),
            scene={"robot": SimpleNamespace(data=SimpleNamespace(body_link_pos_w=torch.zeros(4, 5, 3)))},
        )

    def weight(self, **overrides):
        params = {"start_time_s": 0.0, "end_time_s": 1.25, "ramp_time_s": 0.0, **overrides}
        return FUNCTIONS["_wheel_phase_weight"](self.env, command_name="motion", **params)

    def test_zero_ramp_holds_full_weight_until_first_frame_at_or_after_end(self):
        weight = self.weight()
        torch.testing.assert_close(weight, torch.tensor([1.0, 1.0, 1.0, 0.0]))
        self.assertEqual(weight.device, self.command.time_steps.device)
        self.assertTrue(weight.is_floating_point())

    def test_zero_ramp_includes_exact_start_and_excludes_exact_end(self):
        self.command.motion.fps = torch.tensor(100.0)
        self.command.time_steps[:] = torch.tensor([24, 25, 124, 125])
        torch.testing.assert_close(self.weight(start_time_s=0.25), torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_zero_ramp_rounds_non_frame_aligned_boundaries_up(self):
        self.command.time_steps[:] = torch.tensor([12, 13, 62, 63])
        torch.testing.assert_close(self.weight(start_time_s=0.25), torch.tensor([0.0, 1.0, 1.0, 0.0]))

    def test_positive_ramp_preserves_existing_fade_out(self):
        self.command.time_steps[:] = torch.tensor([0, 52, 60, 63])
        torch.testing.assert_close(
            self.weight(ramp_time_s=0.2), torch.tensor([1.0, 1.0, 0.15625, 0.0]), atol=1e-6, rtol=1e-5,
        )

    def test_zero_ramp_reaches_same_feet_x_reward_without_division_by_zero(self):
        asset_cfg = SimpleNamespace(name="robot", body_ids=[4, 1])
        # The unchanged geometry calculation supplies four known separations.
        geometry = Mock(return_value=torch.tensor([0.1, 0.2, 0.3, 0.4]))
        with patch.dict(FUNCTIONS, {"_locomotion_same_feet_x_position": geometry}):
            value = FUNCTIONS["wheel_phase_same_feet_x_position"](
                self.env, command_name="motion", asset_cfg=asset_cfg,
                start_time_s=0.0, end_time_s=1.25, ramp_time_s=0.0,
            )
        geometry.assert_called_once_with(self.env, asset_cfg=asset_cfg)
        torch.testing.assert_close(value, torch.tensor([0.1, 0.2, 0.3, 0.0]))

    def test_zero_ramp_still_validates_end_time_and_reference_fps(self):
        for end in (0.0, -1.0, math.nan, math.inf):
            with self.subTest(end=end), self.assertRaises(ValueError):
                self.weight(end_time_s=end)
        for fps in (0.0, -50.0, math.nan, math.inf):
            self.command.motion.fps = torch.tensor(fps)
            with self.subTest(fps=fps), self.assertRaises(ValueError):
                self.weight()


if __name__ == "__main__":
    unittest.main(verbosity=2)

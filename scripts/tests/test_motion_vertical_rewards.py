"""CPU tests for phase-gated takeoff velocity and flight height rewards.

Run: conda run -n isaacsim-5.1 python -B scripts/tests/test_motion_vertical_rewards.py
Execute production functions and wheel registrations from their AST without
launching Isaac Sim or changing its user preferences.
"""

from __future__ import annotations

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


TASK_ROOT = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
TERMS = ("motion_takeoff_vertical_velocity", "motion_flight_height")


def load_functions_and_config():
    path = TASK_ROOT / "mdp/rewards.py"
    names = {
        "_motion_phase_window_weight",
        "motion_phase_anchor_vertical_velocity_error_huber",
        "motion_phase_anchor_height_error_exp",
    }
    nodes = [
        node for node in ast.parse(path.read_text()).body
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        or (isinstance(node, ast.FunctionDef) and node.name in names)
    ]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)

    path = TASK_ROOT / "config/LW_Wheel/flat_env_cfg.py"
    tree = ast.parse(path.read_text())
    rewards = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LWWheelRewardsCfg")
    registrations = [
        node for node in rewards.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in TERMS for target in node.targets)
    ]
    config_namespace = {"mdp": SimpleNamespace(**{name: namespace[name] for name in names}), "RewTerm": SimpleNamespace}
    exec(compile(ast.Module(body=registrations, type_ignores=[]), str(path), "exec"), config_namespace)
    return {name: config_namespace[name] for name in TERMS}


CONFIG = load_functions_and_config()


def make_env(frames, fps=50.0):
    count = len(frames)
    data = SimpleNamespace(
        body_com_lin_vel_w=torch.full((count, 3, 3), 1000.0, dtype=torch.float64),
        body_link_lin_vel_w=torch.full((count, 3, 3), -1000.0, dtype=torch.float64),
        body_com_pos_w=torch.full((count, 3, 3), 1000.0, dtype=torch.float64),
        root_lin_vel_b=torch.full((count, 3), -1000.0, dtype=torch.float64),
    )
    command = SimpleNamespace(
        motion=SimpleNamespace(fps=torch.tensor(fps)),
        time_steps=torch.tensor(frames),
        robot=SimpleNamespace(data=data),
        robot_anchor_body_index=1,
        anchor_lin_vel_w=torch.zeros(count, 3, dtype=torch.float64),
        anchor_pos_w=torch.zeros(count, 3, dtype=torch.float64),
        robot_anchor_pos_w=torch.zeros(count, 3, dtype=torch.float64),
    )
    command.anchor_lin_vel_w[:, 2] = 2.26
    data.body_com_lin_vel_w[:, 1, 2] = 2.26
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: {"motion": command}[name]),
        episode_length_buf=torch.arange(count) * 10000,
    )
    return env, command


def reward(term, env, **overrides):
    config = CONFIG[term]
    return config.func(env, **(config.params | overrides))


class MotionVerticalRewardTests(unittest.TestCase):
    def test_wheel_only_registration_matches_approved_weights_and_windows(self):
        velocity, height = (CONFIG[name] for name in TERMS)
        self.assertEqual(velocity.weight, -2.0)
        self.assertEqual(velocity.params, {
            "command_name": "motion", "velocity_scale": 1.0,
            "start_time_s": 1.10, "end_time_s": 1.36, "ramp_time_s": 0.04,
        })
        self.assertEqual(height.weight, 2.0)
        self.assertEqual(height.params, {
            "command_name": "motion", "std": 0.10,
            "start_time_s": 1.20, "end_time_s": 1.70, "ramp_time_s": 0.05,
        })
        for relative in ("tracking_env_cfg.py", "config/LW_Leg/flat_env_cfg.py"):
            identifiers = {
                node.id for node in ast.walk(ast.parse((TASK_ROOT / relative).read_text()))
                if isinstance(node, ast.Name)
            }
            self.assertTrue(identifiers.isdisjoint(TERMS))

    def test_velocity_huber_has_quadratic_center_and_linear_tails(self):
        errors = torch.tensor([-5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0], dtype=torch.float64)
        env, command = make_env([63] * len(errors))
        command.robot.data.body_com_lin_vel_w[:, 1, 2] = command.anchor_lin_vel_w[:, 2] + errors
        expected = torch.tensor([4.5, 0.5, 0.125, 0.0, 0.125, 0.5, 4.5], dtype=torch.float64)
        torch.testing.assert_close(reward(TERMS[0], env), expected)
        torch.testing.assert_close(CONFIG[TERMS[0]].weight * reward(TERMS[0], env), -2.0 * expected)

    def test_velocity_normalizes_error_before_huber(self):
        env, command = make_env([63])
        command.robot.data.body_com_lin_vel_w[:, 1, 2] += 2.0
        torch.testing.assert_close(reward(TERMS[0], env, velocity_scale=2.0), torch.tensor([0.5], dtype=torch.float64))

    def test_velocity_uses_selected_anchor_com_world_z_only(self):
        env, command = make_env([63, 63])
        command.robot.data.body_com_lin_vel_w[1, 1, 2] += 1.0
        torch.testing.assert_close(reward(TERMS[0], env), torch.tensor([0.0, 0.5], dtype=torch.float64))

    def test_takeoff_window_includes_peak_and_smoothly_enters_and_exits(self):
        env, command = make_env([0, 54, 55, 56, 57, 63, 66, 67, 68, 100])
        command.robot.data.body_com_lin_vel_w[:, 1, 2] += 1.0
        expected = torch.tensor([0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(reward(TERMS[0], env), expected, atol=1e-6, rtol=1e-5)

    def test_height_tracks_signed_link_height_error_with_ten_cm_scale(self):
        env, command = make_env([75] * 5)
        command.robot_anchor_pos_w[:, 2] = torch.tensor([0.0, 0.1, -0.1, 0.2, -0.2])
        expected = torch.tensor([1.0, math.exp(-1.0), math.exp(-1.0), math.exp(-4.0), math.exp(-4.0)], dtype=torch.float64)
        torch.testing.assert_close(reward(TERMS[1], env), expected)
        torch.testing.assert_close(CONFIG[TERMS[1]].weight * reward(TERMS[1], env), 2.0 * expected)

    def test_height_ignores_xy_and_common_world_origin_offset(self):
        env, command = make_env([75, 75])
        command.robot_anchor_pos_w[:, :2] = 100.0
        command.anchor_pos_w[:, :2] = -100.0
        command.robot_anchor_pos_w[:, 2] = 0.1
        original = reward(TERMS[1], env)
        command.robot_anchor_pos_w[:, 2] += torch.tensor([10.0, -10.0])
        command.anchor_pos_w[:, 2] += torch.tensor([10.0, -10.0])
        torch.testing.assert_close(reward(TERMS[1], env), original)

    def test_flight_height_window_smoothly_enters_and_exits(self):
        env, _ = make_env([59, 60, 61, 62, 63, 75, 82, 83, 84, 85, 86])
        expected = torch.tensor([0.0, 0.0, 0.352, 0.896, 1.0, 1.0, 1.0, 0.896, 0.352, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(reward(TERMS[1], env), expected, atol=1e-6, rtol=1e-5)

    def test_phase_follows_reference_fps_instead_of_episode_age(self):
        frames = [55, 56, 63, 67, 68, 75, 83, 85]
        env50, command50 = make_env(frames)
        env100, command100 = make_env([2 * frame for frame in frames], fps=100.0)
        command50.robot.data.body_com_lin_vel_w[:, 1, 2] += 1.0
        command100.robot.data.body_com_lin_vel_w[:, 1, 2] += 1.0
        env100.episode_length_buf = env50.episode_length_buf.flip(0)
        for name in TERMS:
            with self.subTest(term=name):
                torch.testing.assert_close(reward(name, env50), reward(name, env100))

    def test_per_environment_reference_reset_and_rewards_do_not_modify_inputs(self):
        env, command = make_env([63, 63])
        command.robot.data.body_com_lin_vel_w[:, 1, 2] += 1.0
        command.time_steps[1] = 0
        before = copy.deepcopy(command)
        torch.testing.assert_close(reward(TERMS[0], env), torch.tensor([0.5, 0.0], dtype=torch.float64))
        torch.testing.assert_close(reward(TERMS[1], env), torch.tensor([1.0, 0.0], dtype=torch.float64))
        torch.testing.assert_close(command.time_steps, before.time_steps)
        torch.testing.assert_close(command.robot.data.body_com_lin_vel_w, before.robot.data.body_com_lin_vel_w)
        torch.testing.assert_close(command.robot_anchor_pos_w, before.robot_anchor_pos_w)
        torch.testing.assert_close(command.anchor_pos_w, before.anchor_pos_w)

    def test_velocity_gradient_does_not_saturate_for_large_errors(self):
        env, command = make_env([63] * 4)
        errors = torch.tensor([-2.0, -0.25, 0.25, 2.0], dtype=torch.float64, requires_grad=True)
        command.robot.data.body_com_lin_vel_w[:, 1, 2] = command.anchor_lin_vel_w[:, 2] + errors
        (CONFIG[TERMS[0]].weight * reward(TERMS[0], env)).sum().backward()
        torch.testing.assert_close(errors.grad, torch.tensor([2.0, 0.5, -0.5, -2.0], dtype=torch.float64))

    def test_invalid_scales_fail_explicitly(self):
        env, _ = make_env([63])
        for name, parameter in ((TERMS[0], "velocity_scale"), (TERMS[1], "std")):
            for value in (0.0, -0.1, math.nan, math.inf, -math.inf):
                with self.subTest(term=name, value=value), self.assertRaisesRegex(ValueError, "finite, positive"):
                    reward(name, env, **{parameter: value})


if __name__ == "__main__":
    unittest.main(verbosity=2)

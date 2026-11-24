"""Exercise the production tensor math on CPU, independently of a running Isaac stage."""

import ast
import math
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/foot_rewards.py"
sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
from policy_evaluation_telemetry import foot_ground_measurement


def load_tensor_math():
    module = types.ModuleType("locomotion_foot_reward_math")
    module.__file__ = str(SOURCE)
    module.__dict__.update(wp=wp, torch=torch, SceneEntityCfg=types.SimpleNamespace)
    sys.modules[module.__name__] = module  # Warp needs the original filename for kernel compilation.
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef)
             and node.name in ("_contact_motion", "swing_clearance_score", "feet_clearance_reward")]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    return module


MATH = load_tensor_math()


class FootRewardTests(unittest.TestCase):
    def test_legacy_checkpoint_link_target_is_converted_once(self):
        class RecordedTerm:
            legacy_sole_offset = -.071

            def __call__(self, env, command, asset, target, std, support_force):
                return target

        from unittest.mock import patch
        contacts = types.SimpleNamespace(clearance_term=RecordedTerm())
        with patch.object(MATH, "_ground_for", return_value=contacts, create=True):
            self.assertAlmostEqual(MATH.feet_clearance_reward(None, "base_velocity", None, .196, .05, 2.), .125)
            self.assertAlmostEqual(MATH.feet_clearance_reward(None, "base_velocity", None, .125, .05), .125)

    def test_packed_contacts_match_independent_probe(self):
        wp.init()
        state = torch.zeros(4, 13)
        state[0, 7] = state[0, 11] = 1  # Two contacts with speeds 0 and 2, in different terrain filters.
        state[1, 7] = 2  # Airborne.
        state[3, 7] = .25  # The foot slides with the body.
        counts = torch.tensor([[1, 1], [0, 0], [1, 0], [2, 0]], dtype=torch.int32)
        offsets = torch.tensor([[4, 1], [0, 0], [2, 0], [5, 0]], dtype=torch.int32)
        points = torch.zeros(7, 3)
        points[4, 2], points[1, 2] = -1., 1.
        normals = torch.zeros_like(points)
        normals[:, 2] = 1
        forces = torch.tensor([0., 1., 7., 0., 3., 5., 8.])
        load = torch.zeros(4, 1)
        speed = torch.zeros_like(load)
        wp.launch(MATH._contact_motion, dim=4, device="cpu",
                  inputs=[wp.from_torch(state), wp.from_torch(forces), wp.from_torch(points, dtype=wp.vec3),
                          wp.from_torch(normals, dtype=wp.vec3), wp.from_torch(counts), wp.from_torch(offsets), 0],
                  outputs=[wp.from_torch(load), wp.from_torch(speed)])
        for i, ids in enumerate(([4, 1], [], [2], [5, 6])):
            ref = foot_ground_measurement(0., np.zeros(3), state[i, 7:].numpy(), points[ids].numpy(),
                                          forces[ids].numpy())
            self.assertAlmostEqual(load[i, 0].item(), ref["normal_force_n"])
            self.assertAlmostEqual(speed[i, 0].item(), ref["contact_slip_speed_m_s"] or 0.)
        torch.testing.assert_close(speed[:, 0], torch.tensor([.5, 0, 0, .25]))

    def test_clocked_swing_uses_sole_gap_and_load(self):
        gap = torch.tensor([[.125, 0], [0, .125], [.075, 0], [.125, 0], [0, 0], [.125, 0], [float("inf"), 0]])
        load = torch.tensor([[0, 80], [80, 0], [0, 80], [6, 80], [0, 0], [0, 80], [0, 80]])
        steps = torch.tensor([30, 10, 30, 30, 30, 30, 30])
        cmd = torch.zeros(7, 3)
        cmd[:, 2] = -.5
        cmd[5] = 0
        score = MATH.swing_clearance_score(gap, load, steps, .02, cmd, (1.25, .5, .5), .125, .05, 5.)
        torch.testing.assert_close(score, torch.tensor([1., 1., math.exp(-1), 0., 0., 0., 0.]))

    def test_contact_defined_swing_supports_four_feet(self):
        gap = torch.tensor([[.1, .1, .125, 0.], [.1, .1, .1, .1]])
        load = torch.tensor([[80., 60., 5., 0.], [10., 10., 10., 10.]])
        score = MATH.swing_clearance_score(gap, load, torch.ones(2), .02, torch.ones(2, 3), None, .125, .05, 5.)
        torch.testing.assert_close(score, torch.tensor([.5, 0.]))

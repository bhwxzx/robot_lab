"""Exercise the production tensor math on CPU, independently of a running Isaac stage."""

import ast
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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
    module.__dict__.update(wp=wp, torch=torch, SceneEntityCfg=lambda name: types.SimpleNamespace(name=name))
    sys.modules[module.__name__] = module  # Warp needs the original filename for kernel compilation.
    for path, functions in (
        (SOURCE, ("_contact_motion", "swing_clearance_score", "feet_clearance_reward", "terrain_height_below")),
        (SOURCE.with_name("rewards.py"), ("base_height_l2", "foot_landing_vel", "foot_impact_reduction")),
    ):
        nodes = [node for node in ast.parse(path.read_text()).body
                 if (isinstance(node, ast.FunctionDef) and node.name in functions)
                 or (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


MATH = load_tensor_math()


class FootRewardTests(unittest.TestCase):
    def test_legacy_checkpoint_link_target_is_converted_once(self):
        class RecordedTerm:
            legacy_sole_offset = -.071

            def __call__(self, env, command, asset, target, std, support_force):
                return target

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

    def test_base_height_averages_only_valid_hits_in_each_environment(self):
        base = torch.zeros(4, 3)
        base[:, 2] = torch.tensor([.79, 1.02, .91, .79])
        hits = torch.zeros(4, 4, 3)
        hits[..., 2] = torch.tensor([[0., 0., 0., 0.], [.1, .3, float("nan"), float("inf")],
                                     [1e7, -1e7, float("nan"), -float("inf")], [-.01, .01, 1e7, -1e7]])
        env = types.SimpleNamespace(scene={
            "robot": types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=base)),
            "scanner": types.SimpleNamespace(data=types.SimpleNamespace(ray_hits_w=hits)),
        })
        reward = MATH.base_height_l2(env, .69, sensor_cfg=types.SimpleNamespace(name="scanner"))
        torch.testing.assert_close(reward, torch.tensor([.01, .0169, 0., .01]))
        torch.testing.assert_close(MATH.base_height_l2(env, .69), (base[:, 2] - .69).square())

    def test_landing_depends_on_local_ground_contact_and_velocity(self):
        class Scene(dict):
            pass

        pos = torch.zeros(4, 2, 3)
        pos[..., 0] = torch.tensor([[-1., 1.], [1., -1.], [-1., 1.], [3., 3.]])
        floor = torch.where(pos[..., 0] > 0, .02, 0.)
        pos[..., 2] = floor + .071 + torch.tensor([[.07, .09], [.07, .09], [.07, .07], [.07, .07]])
        vel = torch.zeros_like(pos)
        vel[..., 2] = -1.
        vel[2, 1, 2] = 1.
        forces = torch.zeros_like(pos)
        forces[2, 0, 2] = 5.
        robot = types.SimpleNamespace(data=types.SimpleNamespace(body_pos_w=pos, body_lin_vel_w=vel))
        scene = Scene(robot=robot)
        scene.sensors = {"contact": types.SimpleNamespace(data=types.SimpleNamespace(net_forces_w=forces))}
        env = types.SimpleNamespace(scene=scene)

        def cast(starts, directions, mesh):
            torch.testing.assert_close(starts[..., :2], pos[..., :2])
            torch.testing.assert_close(directions, directions.new_tensor([0., 0., -1.]).expand_as(directions))
            hits = starts.clone()
            hits[..., 2] = torch.where(starts[..., 0] > 2., float("inf"), floor)
            return (hits,)

        feet = types.SimpleNamespace(name="robot", body_ids=slice(None))
        contact = types.SimpleNamespace(name="contact", body_ids=slice(None))
        with patch.object(MATH, "_terrain_height_geometry", return_value=(-float("inf"), [object()]), create=True), \
             patch.object(MATH, "raycast_mesh", side_effect=cast, create=True):
            penalty = MATH.foot_landing_vel(env, feet, contact, .071, .08)
        torch.testing.assert_close(penalty, torch.tensor([1., 1., 0., 0.]))

    def test_impact_uses_terminal_step_and_seeds_only_the_reset_environment(self):
        vel = torch.zeros(2, 2, 3)
        robot = types.SimpleNamespace(data=types.SimpleNamespace(body_lin_vel_w=vel))
        env = types.SimpleNamespace(scene={"robot": robot}, episode_length_buf=torch.tensor([15, 5]),
                                    reset_buf=torch.tensor([False, False]))
        feet = types.SimpleNamespace(name="robot", body_ids=slice(None))
        torch.testing.assert_close(MATH.foot_impact_reduction(env, 2., feet), torch.zeros(2))
        vel[..., 2] = torch.tensor([[-1., -3.], [0., -.5]])
        env.episode_length_buf += 1
        env.reset_buf[0] = True
        torch.testing.assert_close(MATH.foot_impact_reduction(env, 2., feet), torch.tensor([3., .25]))
        vel[..., 2] = torch.tensor([[0., 0.], [-.5, -1.]])
        env.episode_length_buf[:] = torch.tensor([1, 7])
        env.reset_buf.zero_()
        torch.testing.assert_close(MATH.foot_impact_reduction(env, 2., feet), torch.tensor([0., .5]))
        vel[0, :, 2] = torch.tensor([1., .5])
        env.episode_length_buf += 1
        torch.testing.assert_close(MATH.foot_impact_reduction(env, 2., feet), torch.tensor([1.25, 0.]))

"""CPU contracts for LW joint-reference symmetry and the legacy geometry mode.

Run: conda run -n isaacsim-5.1 python -B scripts/tests/test_motion_joint_symmetry_reward.py
Production reward functions are loaded from AST; Isaac Sim is not started.
"""

from __future__ import annotations

import ast
import copy
import inspect
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


TASK = Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
PAIRS = ("hip", "thigh", "shank", "foot")
ORDERED_NAMES = [f"{side}_{part}_joint" for part in PAIRS for side in ("right", "left")]
# Deliberately different from config pair order, including interleaved wheels.
PHYSICAL_NAMES = [
    "left_hip_joint", "right_hip_joint", "left_thigh_joint", "right_thigh_joint",
    "left_shank_joint", "right_shank_joint", "left_foot_joint", "left_wheel_joint",
    "right_foot_joint", "right_wheel_joint",
]
SCALES = (0.1, 0.2, 0.2, 0.2)


def load_functions():
    path = TASK / "mdp/rewards.py"
    selected = {
        "_joint_reference_mirror_error_huber", "wheel_phase_leg_symmetry",
        "_wheel_phase_weight", "_wheel_phase_geometry_weight",
    }
    nodes = [node for node in ast.parse(path.read_text()).body if (
        isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ) or (isinstance(node, ast.FunctionDef) and node.name in selected)]
    namespace = {"math": math, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


FUNCTIONS = load_functions()
REWARD = FUNCTIONS["wheel_phase_leg_symmetry"]


def make_env(count=3, physical_names=PHYSICAL_NAMES):
    asset = SimpleNamespace(data=SimpleNamespace(
        joint_pos=torch.zeros(count, 10, dtype=torch.float64),
        body_link_pos_w=torch.zeros(count, 2, 3, dtype=torch.float64),
    ))
    command = SimpleNamespace(
        robot=asset,
        joint_pos=torch.zeros(count, 10, dtype=torch.float64),
        motion=SimpleNamespace(fps=torch.tensor([50.0])),
        time_steps=torch.tensor(([0, 30, 100] * count)[:count]),
    )
    env = SimpleNamespace(
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_term=lambda name: {"motion": command}[name]),
    )
    sensor_cfg = SimpleNamespace(
        name="robot", joint_names=ORDERED_NAMES.copy(),
        joint_ids=[physical_names.index(name) for name in ORDERED_NAMES],
        preserve_order=True, body_ids=[0, 1],
    )
    return env, asset, command, sensor_cfg


def mirror(env, cfg, scales=SCALES, **kwargs):
    return REWARD(env, "motion", cfg, mode="joint_reference_mirror", joint_scales=scales, **kwargs)


class JointReferenceSymmetryTests(unittest.TestCase):
    def test_matching_asymmetric_reference_has_zero_cost(self):
        env, asset, command, cfg = make_env()
        command.joint_pos[:] = torch.arange(30, dtype=torch.float64).reshape(3, 10) / 10
        asset.data.joint_pos.copy_(command.joint_pos)
        torch.testing.assert_close(mirror(env, cfg), torch.zeros(3, dtype=torch.float64))

    def test_each_joint_pair_uses_its_scale_and_preserves_environment_isolation(self):
        for pair, scale in enumerate(SCALES):
            for side in (0, 1):
                with self.subTest(pair=pair, side=side):
                    env, asset, _, cfg = make_env()
                    asset.data.joint_pos[1, cfg.joint_ids[2 * pair + side]] = scale
                    # Huber(1) / four pairs = 0.125, independently specified.
                    torch.testing.assert_close(mirror(env, cfg), torch.tensor([0.0, 0.125, 0.0], dtype=torch.float64))

    def test_mirrored_deviations_are_not_a_full_joint_tracking_penalty(self):
        env, asset, _, cfg = make_env()
        asset.data.joint_pos[:, cfg.joint_ids[0]] = 0.3
        asset.data.joint_pos[:, cfg.joint_ids[1]] = -0.3
        torch.testing.assert_close(mirror(env, cfg), torch.zeros(3, dtype=torch.float64))

    def test_large_errors_keep_huber_linear_growth(self):
        env, asset, _, cfg = make_env()
        asset.data.joint_pos[0, cfg.joint_ids[6]] = 0.4
        asset.data.joint_pos[1, cfg.joint_ids[6]] = 0.6
        torch.testing.assert_close(mirror(env, cfg), torch.tensor([0.375, 0.625, 0.0], dtype=torch.float64))

    def test_wheel_positions_do_not_contribute(self):
        env, asset, command, cfg = make_env()
        for index in (7, 9):
            asset.data.joint_pos[:, index] = 1e6
            command.joint_pos[:, index] = -1e6
        torch.testing.assert_close(mirror(env, cfg), torch.zeros(3, dtype=torch.float64))

    def test_joint_names_work_with_another_physical_joint_order(self):
        names = PHYSICAL_NAMES[::-1]
        env, asset, command, cfg = make_env(physical_names=names)
        command.joint_pos[:] = torch.arange(30, dtype=torch.float64).reshape(3, 10)
        asset.data.joint_pos.copy_(command.joint_pos)
        asset.data.joint_pos[2, names.index("left_hip_joint")] += 0.1
        torch.testing.assert_close(mirror(env, cfg), torch.tensor([0.0, 0.0, 0.125], dtype=torch.float64))

    def test_full_reference_coverage_and_reset_without_state_leak(self):
        env, asset, command, cfg = make_env()
        asset.data.joint_pos[:, cfg.joint_ids[0]] = 0.1
        # Frames 0, 30, 100 all contribute; geometry gating is irrelevant.
        torch.testing.assert_close(mirror(env, cfg), torch.full((3,), 0.125, dtype=torch.float64))
        command.time_steps[1] = 0
        command.joint_pos[1].copy_(asset.data.joint_pos[1])
        torch.testing.assert_close(mirror(env, cfg), torch.tensor([0.125, 0.0, 0.125], dtype=torch.float64))

    def test_invalid_scales_and_joint_pair_configuration_fail_explicitly(self):
        for scales in (None, (), (0.1,), (0.0, 0.2, 0.2, 0.2), (-0.1, 0.2, 0.2, 0.2),
                       (math.nan, 0.2, 0.2, 0.2), (math.inf, 0.2, 0.2, 0.2)):
            env, _, _, cfg = make_env()
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                mirror(env, cfg, scales)
        for invalid in ("unordered", "duplicate", "cross_pair", "wheel"):
            env, _, _, cfg = make_env()
            if invalid == "unordered":
                cfg.preserve_order = False
            elif invalid == "duplicate":
                cfg.joint_names[2:4] = cfg.joint_names[:2]
            elif invalid == "cross_pair":
                cfg.joint_names[1], cfg.joint_names[3] = cfg.joint_names[3], cfg.joint_names[1]
            else:
                cfg.joint_names[-2:] = ["right_wheel_joint", "left_wheel_joint"]
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mirror(env, cfg)

    def test_reference_robot_and_batch_shape_must_match(self):
        env, _, command, cfg = make_env()
        command.robot = SimpleNamespace()
        with self.assertRaises(ValueError):
            mirror(env, cfg)
        env, _, command, cfg = make_env()
        command.joint_pos = command.joint_pos[:1]
        with self.assertRaises(ValueError):
            mirror(env, cfg)

    def test_inputs_are_not_modified(self):
        env, asset, command, cfg = make_env()
        asset.data.joint_pos[1, cfg.joint_ids[2]] = 0.3
        before = [asset.data.joint_pos.clone(), command.joint_pos.clone(), command.time_steps.clone()]
        config_before = copy.deepcopy(vars(cfg))
        mirror(env, cfg)
        for current, original in zip((asset.data.joint_pos, command.joint_pos, command.time_steps), before):
            torch.testing.assert_close(current, original)
        self.assertEqual(vars(cfg), config_before)

    def test_default_legacy_mode_preserves_geometry_reward_and_phase_ramp(self):
        env, _, command, cfg = make_env()
        command.time_steps = torch.tensor([80, 85, 90])
        calls = []

        def geometry(env, std, asset_cfg):
            calls.append((std, asset_cfg))
            return torch.full((3,), 0.8)

        FUNCTIONS["_locomotion_leg_symmetry"] = geometry
        values = REWARD(env, "motion", cfg, std=0.05, start_time_s=1.6, ramp_time_s=0.2)
        torch.testing.assert_close(values, torch.tensor([0.0, 0.4, 0.8]), atol=1e-6, rtol=1e-5)
        self.assertEqual(calls, [(0.05, cfg)])
        with self.assertRaises(ValueError):
            REWARD(env, "motion", cfg, mode="misspelled")

    def test_wheel_config_selects_joint_mode_but_retains_zero_weight(self):
        path = TASK / "config/LW_Wheel/flat_env_cfg.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LWWheelBeyondMimicFlatEnvCfg")
        joint_names = next(ast.literal_eval(n.value) for n in cls.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "joint_names_without_wheels" for t in n.targets))
        post_init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
        nodes = [n for n in post_init.body if isinstance(n, ast.Assign) and any(
            ast.unparse(t) in ("self.rewards.leg_symmetry.weight", "self.rewards.leg_symmetry.params") for t in n.targets)]
        cfg = SimpleNamespace(joint_names_without_wheels=joint_names,
                              rewards=SimpleNamespace(leg_symmetry=SimpleNamespace()))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), {
            "self": cfg, "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, **kwargs),
        })
        term = cfg.rewards.leg_symmetry
        self.assertEqual(term.weight, 0.0)
        self.assertEqual(term.params["mode"], "joint_reference_mirror")
        self.assertEqual(term.params["asset_cfg"].joint_names, ORDERED_NAMES)
        self.assertTrue(term.params["asset_cfg"].preserve_order)
        self.assertEqual(term.params["joint_scales"], SCALES)
        self.assertNotIn("start_time_s", term.params)
        inspect.signature(REWARD).bind(None, **term.params)


if __name__ == "__main__":
    unittest.main(verbosity=2)

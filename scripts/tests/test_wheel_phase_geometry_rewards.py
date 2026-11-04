"""CPU contracts for double-wheel geometry, using the locomotion implementations.

Run: conda run -n isaacsim-5.1 python -B scripts/tests/test_wheel_phase_geometry_rewards.py
Isaac Sim is not imported or started.
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


ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic"
LOCOMOTION = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/rewards.py"
WHEELS = ["right_wheel_link", "left_wheel_link"]
BODY_NAMES = ["base_link", "left_wheel_link", "right_foot_link", "left_foot_link", "right_wheel_link"]
TERMS = {
    "leg_symmetry": "wheel_phase_leg_symmetry",
    "feet_distance_penalize": "wheel_phase_feet_distance_penalize",
}


def entity(name, **kwargs):
    return SimpleNamespace(name=name, **kwargs)


def quat_apply_inverse(q, v):
    """Test-only wxyz quaternion adapter for the simulation-free function loader."""
    imaginary = -q[..., 1:]
    cross = 2 * torch.cross(imaginary, v, dim=-1)
    return v + q[..., :1] * cross + torch.cross(imaginary, cross, dim=-1)


def load_functions():
    tree = ast.parse((TASK / "mdp/rewards.py").read_text())
    imports = next(n for n in tree.body if isinstance(n, ast.ImportFrom)
                   and n.module == "robot_lab.tasks.manager_based.locomotion.velocity.mdp.rewards")
    aliases = {n.name: n.asname or n.name for n in imports.names if n.name in TERMS}
    if set(aliases) != set(TERMS):
        raise AssertionError("Both geometry wrappers must import the locomotion implementations.")
    namespace = {"torch": torch, "math": math, "SceneEntityCfg": entity,
                 "quat_apply_inverse": quat_apply_inverse}
    loc_nodes = [n for n in ast.parse(LOCOMOTION.read_text()).body
                 if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
                 or (isinstance(n, ast.FunctionDef) and n.name in aliases)]
    exec(compile(ast.Module(body=loc_nodes, type_ignores=[]), str(LOCOMOTION), "exec"), namespace)
    for name, alias in aliases.items():
        namespace[alias] = namespace[name]
    selected = set(TERMS.values()) | {
        "_wheel_phase_weight", "_wheel_phase_geometry_weight", "_wheel_geometry_body_names",
    }
    nodes = [n for n in tree.body if (isinstance(n, ast.ImportFrom) and n.module == "__future__")
             or (isinstance(n, ast.FunctionDef) and n.name in selected)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TASK / "mdp/rewards.py"), "exec"), namespace)
    return namespace


FUNCTIONS = load_functions()


def make_env(count=4):
    data = SimpleNamespace(
        body_link_pos_w=torch.zeros(count, len(BODY_NAMES), 3, dtype=torch.float64),
        root_link_state_w=torch.zeros(count, 13, dtype=torch.float64),
        root_link_quat_w=torch.tensor([1., 0., 0., 0.], dtype=torch.float64).repeat(count, 1),
    )
    # Non-wheel links have a deliberately incompatible spacing.
    data.body_link_pos_w[:, 2, 1] = -5.
    data.body_link_pos_w[:, 3, 1] = 5.

    def find_bodies(names):
        return [i for i, name in enumerate(BODY_NAMES) if name in names], [n for n in BODY_NAMES if n in names]

    asset = SimpleNamespace(data=data, find_bodies=find_bodies)
    command = SimpleNamespace(time_steps=torch.zeros(count, dtype=torch.long),
                              motion=SimpleNamespace(fps=torch.tensor(50.)))
    env = SimpleNamespace(scene={"robot": asset},
                          command_manager=SimpleNamespace(get_term=lambda _: command),
                          episode_length_buf=torch.arange(count) * 100)
    cfg = entity("robot", body_names=WHEELS.copy(), body_ids=[4, 1], preserve_order=True)
    data.body_link_pos_w[:, 4, 1] = -0.258
    data.body_link_pos_w[:, 1, 1] = 0.258
    return env, data, command, cfg


def reward(env, cfg, name, **overrides):
    params = {"std": 0.05} if name == "leg_symmetry" else {
        "min_feet_distance": 0.496, "max_feet_distance": 0.536,
    }
    return FUNCTIONS[TERMS[name]](
        env, **{"command_name": "motion", "asset_cfg": cfg, "start_time_s": 0.,
                "end_time_s": 1.25, "ramp_time_s": 0., **params, **overrides},
    )


class WheelGeometryTests(unittest.TestCase):
    def setUp(self):
        self.env, self.data, self.command, self.cfg = make_env()

    def test_distance_band_bounds_and_signed_linear_violations(self):
        d = torch.tensor([0.496, 0.536, 0.444, 0.55], dtype=torch.float64)
        self.data.body_link_pos_w[:, 4, 1] = -d / 2
        self.data.body_link_pos_w[:, 1, 1] = d / 2
        torch.testing.assert_close(
            -20 * reward(self.env, self.cfg, "feet_distance_penalize"),
            torch.tensor([0., 0., -1.04, -0.28], dtype=torch.float64), atol=1e-12, rtol=1e-12,
        )

    def test_symmetry_uses_base_lateral_distance_with_meter_scale(self):
        self.data.body_link_pos_w[:, 1, 1] += torch.tensor([0., 0.02, 0.05, 0.1])
        expected = torch.tensor([1., math.exp(-0.16), math.exp(-1), math.exp(-4)], dtype=torch.float64)
        torch.testing.assert_close(reward(self.env, self.cfg, "leg_symmetry"), expected)

    def test_symmetric_inward_motion_keeps_symmetry_but_is_penalized_for_distance(self):
        self.data.body_link_pos_w[:, 4, 1] = -0.222
        self.data.body_link_pos_w[:, 1, 1] = 0.222
        torch.testing.assert_close(reward(self.env, self.cfg, "leg_symmetry"), torch.ones(4, dtype=torch.float64))
        torch.testing.assert_close(reward(self.env, self.cfg, "feet_distance_penalize"),
                                   torch.full((4,), 0.052, dtype=torch.float64))

    def test_distance_preserves_world_xy_definition_and_clipping(self):
        self.data.body_link_pos_w[:, 4] = 0.
        self.data.body_link_pos_w[:, 1] = torch.tensor([0.4, 0.3, 10.])
        torch.testing.assert_close(reward(self.env, self.cfg, "feet_distance_penalize"),
                                   torch.zeros(4, dtype=torch.float64))
        self.data.body_link_pos_w[:, 1, 0] = 3.
        torch.testing.assert_close(reward(self.env, self.cfg, "feet_distance_penalize"),
                                   torch.ones(4, dtype=torch.float64))

    def test_translation_and_yaw_do_not_change_either_geometry_value(self):
        self.data.body_link_pos_w[:, 1, 1] = 0.23
        before = {n: reward(self.env, self.cfg, n) for n in TERMS}
        pos = self.data.body_link_pos_w.clone()
        shift = torch.tensor([2., -3., 4.], dtype=torch.float64)
        self.data.body_link_pos_w[:, :, 0] = -pos[:, :, 1] + shift[0]
        self.data.body_link_pos_w[:, :, 1] = pos[:, :, 0] + shift[1]
        self.data.body_link_pos_w[:, :, 2] = pos[:, :, 2] + shift[2]
        self.data.root_link_state_w[:, :3] = shift
        self.data.root_link_quat_w[:] = torch.tensor([math.sqrt(0.5), 0., 0., math.sqrt(0.5)])
        for name in TERMS:
            torch.testing.assert_close(reward(self.env, self.cfg, name), before[name])

    def test_roll_changes_horizontal_distance_but_not_base_frame_symmetry(self):
        angle = math.pi / 3
        for index, y in ((4, -0.258), (1, 0.258)):
            self.data.body_link_pos_w[:, index, 1] = math.cos(angle) * y + math.sin(angle) * 0.5
            self.data.body_link_pos_w[:, index, 2] = math.sin(angle) * y - math.cos(angle) * 0.5
        self.data.root_link_quat_w[:] = torch.tensor([math.cos(angle / 2), math.sin(angle / 2), 0., 0.])
        torch.testing.assert_close(reward(self.env, self.cfg, "leg_symmetry"), torch.ones(4, dtype=torch.float64))
        torch.testing.assert_close(reward(self.env, self.cfg, "feet_distance_penalize"),
                                   torch.full((4,), 0.238, dtype=torch.float64))

    def test_reference_frame_window_and_reset_are_independent_of_episode_age_or_contact(self):
        self.data.body_link_pos_w[:, 4, 1] = -0.22
        self.data.body_link_pos_w[:, 1, 1] = 0.22
        self.command.time_steps[:] = torch.tensor([0, 62, 63, 100])
        # No contact sensor is present: unloading cannot switch these terms off.
        for name, value in (("leg_symmetry", 1.), ("feet_distance_penalize", 0.056)):
            torch.testing.assert_close(reward(self.env, self.cfg, name),
                                       torch.tensor([value, value, 0., 0.], dtype=torch.float64))
        self.command.time_steps[2] = 0
        self.env.episode_length_buf[:] = 10000
        torch.testing.assert_close(reward(self.env, self.cfg, "feet_distance_penalize"),
                                   torch.tensor([0.056, 0.056, 0.056, 0.], dtype=torch.float64))

    def test_nonzero_start_and_exact_end_at_another_fps(self):
        self.command.motion.fps = torch.tensor(100.)
        self.command.time_steps[:] = torch.tensor([24, 25, 124, 125])
        torch.testing.assert_close(reward(self.env, self.cfg, "leg_symmetry", start_time_s=0.25),
                                   torch.tensor([0., 1., 1., 0.], dtype=torch.float64))

    def test_reversing_pair_order_does_not_change_geometry(self):
        before = {n: reward(self.env, self.cfg, n) for n in TERMS}
        self.cfg.body_names.reverse()
        self.cfg.body_ids.reverse()
        for name in TERMS:
            torch.testing.assert_close(reward(self.env, self.cfg, name), before[name])

    def test_invalid_scales_distance_bounds_selection_and_window_fail(self):
        for std in (0., -1., math.nan, math.inf):
            with self.subTest(std=std), self.assertRaises(ValueError):
                reward(self.env, self.cfg, "leg_symmetry", std=std)
        for low, high in ((0., 0.5), (-1., 0.5), (0.5, 0.5), (0.6, 0.5),
                          (math.nan, 0.5), (0.5, math.inf)):
            with self.subTest(bounds=(low, high)), self.assertRaises(ValueError):
                reward(self.env, self.cfg, "feet_distance_penalize", min_feet_distance=low, max_feet_distance=high)
        for name in TERMS:
            for names, ids in ((["right_foot_link", "left_foot_link"], [2, 3]),
                               (WHEELS, [4]), ([WHEELS[0], WHEELS[0]], [4, 4]),
                               (".*_wheel_link", [4, 1])):
                cfg = entity("robot", body_names=names, body_ids=ids)
                with self.subTest(term=name, bodies=names, ids=ids), self.assertRaises(ValueError):
                    reward(self.env, cfg, name)
            for overrides in ({"start_time_s": -1.}, {"end_time_s": 0.},
                              {"ramp_time_s": -1.}, {"end_time_s": math.inf}):
                with self.subTest(term=name, overrides=overrides), self.assertRaises(ValueError):
                    reward(self.env, self.cfg, name, **overrides)

    def test_calls_do_not_mutate_feedback_config_or_reference_progress(self):
        before = {k: v.clone() for k, v in vars(self.data).items()}
        frames = self.command.time_steps.clone()
        config = copy.deepcopy(vars(self.cfg))
        for name in TERMS:
            reward(self.env, self.cfg, name)
        for key, value in before.items():
            torch.testing.assert_close(getattr(self.data, key), value)
        torch.testing.assert_close(self.command.time_steps, frames)
        self.assertEqual(vars(self.cfg), config)


class WheelGeometryConfigTests(unittest.TestCase):
    def test_shared_terms_disabled_and_wheel_overrides_match_approved_config(self):
        path = TASK / "tracking_env_cfg.py"
        cls = next(n for n in ast.parse(path.read_text()).body
                   if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        nodes = [n for n in cls.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in TERMS for t in n.targets)]
        namespace = {"mdp": SimpleNamespace(**{n: FUNCTIONS[n] for n in TERMS.values()}),
                     "RewTerm": SimpleNamespace, "SceneEntityCfg": entity}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        for name in TERMS:
            self.assertEqual(namespace[name].weight, 0.)
        for task, class_name in (("LW_Wheel", "LWWheelBeyondMimicFlatEnvCfg"),
                                 ("LW_Leg", "LWLegBeyondMimicFlatEnvCfg")):
            path = TASK / f"config/{task}/flat_env_cfg.py"
            cls = next(n for n in ast.parse(path.read_text()).body
                       if isinstance(n, ast.ClassDef) and n.name == class_name)
            post = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__post_init__")
            nodes = [n for n in post.body if isinstance(n, ast.Assign)
                     and any(any(ast.unparse(t).startswith(f"self.rewards.{name}.") for name in TERMS)
                             for t in n.targets)]
            cfg = SimpleNamespace(rewards=SimpleNamespace(**{n: copy.deepcopy(namespace[n]) for n in TERMS}))
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
                 {"self": cfg, "SceneEntityCfg": entity})
            for name, function_name in TERMS.items():
                with self.subTest(task=task, term=name):
                    term = getattr(cfg.rewards, name)
                    self.assertIs(term.func, FUNCTIONS[function_name])
                    self.assertEqual(term.weight, {"leg_symmetry": 0.5, "feet_distance_penalize": -20.}[name]
                                     if task == "LW_Wheel" else 0.)
                    expected = {"command_name": "motion", "start_time_s": 0., "end_time_s": 1.25,
                                "ramp_time_s": 0., **({"std": 0.05} if name == "leg_symmetry" else {
                                    "min_feet_distance": 0.496, "max_feet_distance": 0.536})}
                    self.assertEqual({k: v for k, v in term.params.items() if k != "asset_cfg"}, expected)
                    asset = term.params["asset_cfg"]
                    self.assertEqual(asset.name, "robot")
                    self.assertEqual(asset.body_names, WHEELS)
                    self.assertTrue(asset.preserve_order)
                    inspect.signature(term.func).bind(None, **term.params)


if __name__ == "__main__":
    unittest.main(verbosity=2)

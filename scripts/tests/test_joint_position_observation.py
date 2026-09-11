"""Core wheel-position regression checks; no physics environment is created.

Run with: conda run -n isaacsim-5.1 python scripts/tests/test_joint_position_observation.py
Historical configuration audits and live/history captures: learnings/code_validation/README.md.
"""

import itertools
from types import SimpleNamespace
import unittest

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import omni.kit.app
import torch
from omegaconf import OmegaConf

import robot_lab.tasks  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import GaussianNoiseCfg, gaussian_noise
from robot_lab.tasks.manager_based.beyondmimic import mdp as beyond_mdp
from robot_lab.tasks.manager_based.locomotion.velocity import mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.LW.LW_Leg.flat_env_cfg import LWLegFlatAmpRoaEnvCfg


NATIVE_NAMES = [
    "left_hip_joint", "right_hip_joint", "left_thigh_joint", "right_thigh_joint",
    "left_shank_joint", "right_shank_joint", "left_foot_joint", "left_wheel_joint",
    "right_foot_joint", "right_wheel_joint",
]
POLICY_IDS = [1, 0, 3, 2, 5, 4, 8, 6, 9, 7]
WHEEL_NAMES = ["right_wheel_joint", "left_wheel_joint"]
DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
OBSERVATION_FUNCTIONS = [mdp.joint_pos_rel_without_wheel, beyond_mdp.joint_pos_rel_without_wheel]


class JointPositionTests(unittest.TestCase):
    def test_native_indices_are_masked_before_selection(self):
        q = torch.arange(20, dtype=torch.float32).reshape(2, 10) + 1
        default = torch.full_like(q, 0.25)
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=q, default_joint_pos=default))
        env = SimpleNamespace(scene={"robot": asset})
        for function, selected in itertools.product(OBSERVATION_FUNCTIONS, [POLICY_IDS, [6, 1], [9, 6, 7], slice(None), slice(0, 10, 2)]):
            with self.subTest(function=function.__module__, selected=selected):
                before_q, before_default = q.clone(), default.clone()
                cfg = SceneEntityCfg("robot", joint_ids=selected)
                wheels = SceneEntityCfg("robot", joint_ids=[7, 9])
                result = function(env, cfg, wheels)
                selected_ids = list(range(10))[selected] if isinstance(selected, slice) else selected
                for column, native_id in enumerate(selected_ids):
                    expected = torch.zeros(2) if NATIVE_NAMES[native_id] in WHEEL_NAMES else q[:, native_id] - default[:, native_id]
                    torch.testing.assert_close(result[:, column], expected, rtol=0, atol=0)
                result.add_(100)
                torch.testing.assert_close(q, before_q, rtol=0, atol=0)
                torch.testing.assert_close(default, before_default, rtol=0, atol=0)


    def test_wheel_selector_slice(self):
        q = torch.arange(10, dtype=torch.float32)[None, :]
        env = SimpleNamespace(scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos=q, default_joint_pos=torch.zeros_like(q)))})
        result = mdp.joint_pos_rel_without_wheel(
            env, SceneEntityCfg("robot", joint_ids=POLICY_IDS), SceneEntityCfg("robot", joint_ids=slice(7, 10, 2))
        )
        self.assertEqual(result[0, 7].item(), 6)
        self.assertEqual(result[0, 8].item(), 0)
        self.assertEqual(result[0, 9].item(), 0)


    def test_noise_follows_joint_names_and_preserves_input(self):
        for names in [[NATIVE_NAMES[i] for i in POLICY_IDS], NATIVE_NAMES]:
            for device in DEVICES:
                with self.subTest(names=names, device=device):
                    cfg = mdp.UniformJointPositionNoiseCfg(
                        joint_names=names, wheel_joint_names=WHEEL_NAMES, n_min=-0.01, n_max=0.01
                    )
                    data = torch.full((2048, 10), 0.3, device=device)
                    wheels = [i for i, name in enumerate(names) if name in WHEEL_NAMES]
                    legs = [i for i, name in enumerate(names) if name not in WHEEL_NAMES]
                    data[:, wheels] = 0
                    before = data.clone()
                    result = cfg.func(data, cfg)
                    self.assertEqual(torch.count_nonzero(result[:, wheels]).item(), 0)
                    delta = result[:, legs] - data[:, legs]
                    self.assertLessEqual(delta.abs().max().item(), 0.010001)
                    self.assertTrue(torch.all(delta.std(dim=0) > 0.004).item())
                    torch.testing.assert_close(data, before, rtol=0, atol=0)
                    self.assertEqual(result.shape, data.shape)
                    self.assertEqual(result.dtype, data.dtype)
                    self.assertEqual(result.device, data.device)


    def test_gaussian_noise_preserves_nonwheel_distribution(self):
        for names, device in itertools.product([NATIVE_NAMES, [NATIVE_NAMES[i] for i in POLICY_IDS]], DEVICES):
            with self.subTest(names=names, device=device):
                cfg = mdp.GaussianJointPositionNoiseCfg(
                    joint_names=names, wheel_joint_names=WHEEL_NAMES, mean=0.0, std=0.01
                )
                data = torch.zeros((4096, 10), device=device)
                legs = [i for i, name in enumerate(names) if name not in WHEEL_NAMES]
                wheels = [i for i, name in enumerate(names) if name in WHEEL_NAMES]
                torch.manual_seed(123)
                expected = gaussian_noise(data, GaussianNoiseCfg(mean=0.0, std=0.01))
                torch.manual_seed(123)
                actual = cfg.func(data, cfg)
                torch.testing.assert_close(actual[:, legs], expected[:, legs], rtol=0, atol=0)
                self.assertEqual(torch.count_nonzero(actual[:, wheels]).item(), 0)
                self.assertEqual(torch.count_nonzero(data).item(), 0)
                self.assertTrue(torch.all((actual[:, legs].std(dim=0) - 0.01).abs() < 0.001).item())
                restored = cfg.copy()
                restored.from_dict(OmegaConf.to_container(OmegaConf.create(cfg.to_dict()), resolve=True))
                self.assertIs(restored.func, mdp.gaussian_joint_position_noise)
                with self.assertRaisesRegex(ValueError, "observation width"):
                    restored.func(torch.zeros(1, 9), restored)


    def test_noise_config_hydra_roundtrip(self):
        cfg = LWLegFlatAmpRoaEnvCfg().observations.policy.joint_pos.noise
        serialized = OmegaConf.to_container(OmegaConf.create(cfg.to_dict()), resolve=True)
        restored = cfg.copy()
        restored.from_dict(serialized)
        self.assertIs(restored.func, mdp.uniform_joint_position_noise)
        self.assertEqual(restored.joint_names, [NATIVE_NAMES[i] for i in POLICY_IDS])
        self.assertEqual(restored.wheel_joint_names, WHEEL_NAMES)
        self.assertEqual((restored.n_min, restored.n_max), (-0.01, 0.01))
        result = restored.func(torch.zeros(32, 10), restored)
        self.assertEqual(torch.count_nonzero(result[:, 8:10]).item(), 0)


    def test_noise_rejects_wrong_column_count(self):
        cfg = mdp.UniformJointPositionNoiseCfg(joint_names=["left_foot_joint"], wheel_joint_names=WHEEL_NAMES)
        with self.assertRaisesRegex(ValueError, "observation width"):
            cfg.func(torch.zeros(1, 10), cfg)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(JointPositionTests))
    exit_code = 0 if result.wasSuccessful() else 1
    # Kit's fast shutdown can exit before Python reaches SystemExit.
    omni.kit.app.get_app().post_quit(exit_code)
    app.close()
    raise SystemExit(exit_code)

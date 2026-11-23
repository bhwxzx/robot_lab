"""CPU regression tests for Flat LW_Leg ROA symmetry and velocity replay."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from test_roa_training import SyntheticWheelEnv, make_runner, parameters


# Load the tensor transform without launching Isaac Sim for CPU tests.
_path = (Path(__file__).resolve().parents[2] / "source/robot_lab/robot_lab/tasks/manager_based"
         / "locomotion/velocity/mdp/symmetry/lw_leg.py")
_spec = importlib.util.spec_from_file_location("lw_leg_symmetry", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
mirror = _module.compute_roa_symmetric_states


class FlatLegEnv(SyntheticWheelEnv):
    joint_names = [f"{side}_{joint}_joint" for joint in ("hip", "thigh", "shank") for side in ("left", "right")]
    joint_names += ["left_foot_joint", "left_wheel_joint", "right_foot_joint", "right_wheel_joint"]

    def __init__(self):
        self.unwrapped = SimpleNamespace(scene={"robot": SimpleNamespace(joint_names=self.joint_names)})

    def get_observations(self):
        return TensorDict({
            "policy": torch.randn(self.num_envs, 10, 41),
            "critic": torch.randn(self.num_envs, 74),
            "privileged": torch.randn(self.num_envs, 30),
        }, batch_size=[self.num_envs])


class FlatLegSymmetryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.env = FlatLegEnv()

    def test_history_privileged_actions_and_rollout_velocity(self):
        obs = self.env.get_observations()
        obs["policy"][0, :, 39:41] = 0  # Standing gait clock.
        obs["_roa_actor_velocity"] = torch.randn(8, 3)
        obs["_roa_estimated_velocity_mask"] = torch.arange(8).remainder(2).bool().unsqueeze(-1)
        actions = torch.randn(8, 10)
        aug, aug_actions = mirror(self.env, obs, actions)
        for key in obs.keys():
            torch.testing.assert_close(aug[key][:8], obs[key])
        reflected = aug[8:]
        pair_order = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]
        gain_order = [1, 0, 3, 2, 5, 4, 8, 9, 6, 7]
        for key in ("policy", "critic"):
            for start in (9, 19, 29):
                torch.testing.assert_close(reflected[key][..., start:start+10], -obs[key][..., start:start+10][..., pair_order])
            signs = torch.tensor([-1, 1, -1, 1, -1, 1, 1, -1, -1])
            torch.testing.assert_close(reflected[key][..., :9], obs[key][..., :9] * signs)
            torch.testing.assert_close(reflected[key][..., 39:41], -obs[key][..., 39:41])
        polar = torch.tensor([1, -1, 1])
        torch.testing.assert_close(reflected["critic"][:, 41:44], obs["critic"][:, 41:44] * polar)
        for key, start in (("privileged", 0), ("critic", 44)):
            src, dst = obs[key][:, start:], reflected[key][:, start:]
            torch.testing.assert_close(dst[:, 0], src[:, 0])
            torch.testing.assert_close(dst[:, 1:7], src[:, [4, 5, 6, 1, 2, 3]])
            for pos in (7, 17):
                torch.testing.assert_close(dst[:, pos:pos+10], src[:, pos:pos+10][:, gain_order])
            torch.testing.assert_close(dst[:, 27:], src[:, 27:] * polar)
        torch.testing.assert_close(reflected["_roa_actor_velocity"], obs["_roa_actor_velocity"] * polar)
        torch.testing.assert_close(reflected["_roa_estimated_velocity_mask"], obs["_roa_estimated_velocity_mask"])
        torch.testing.assert_close(aug_actions[8:], -actions[:, pair_order])
        restored, restored_actions = mirror(self.env, reflected, aug_actions[8:])
        for key in obs.keys():
            torch.testing.assert_close(restored[key][8:], obs[key])
        torch.testing.assert_close(restored_actions[8:], actions)

    def test_mirror_loss_alone_updates_actor_using_stored_velocity(self):
        with patch("test_roa_training.SyntheticWheelEnv", return_value=self.env):
            runner = make_runner(
                policy_overrides={"vel_offset": 41}, estimated_velocity_schedule=[.5, .5, 0, 1],
                symmetry_cfg=dict(use_data_augmentation=True, use_mirror_loss=True,
                                  mirror_loss_coeff=.1, data_augmentation_func=mirror),
                num_learning_epochs=1, num_mini_batches=1, value_loss_coef=0., entropy_coef=0.,
                priv_reg_coef_schedule=[0., 0., 0, 1],
            )
        alg = runner.alg
        with torch.inference_mode():
            for _ in range(runner.num_steps_per_env):
                obs = self.env.get_observations()
                alg.act(obs)
                alg.process_env_step(obs, torch.zeros(8), torch.zeros(8), {})
            alg.compute_returns(obs)
            alg.storage.advantages.zero_()
            samples = alg.storage.observations.flatten(0, 1)
            mask = samples[alg.VELOCITY_MASK_KEY]
            self.assertTrue(mask.any() and (~mask).any())
            aug, _ = mirror(self.env, samples)
            alg.policy.update_distribution(aug, velocity_override=aug[alg.VELOCITY_INPUT_KEY])
            original, reflected = alg.policy.action_mean.chunk(2)
            _, target = mirror(self.env, actions=original)
            expected = (reflected - target[len(original):]).square().mean().item()
        before = parameters(alg.policy.actor)
        loss = alg.update()["symmetry"]
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(loss, expected, places=6)
        self.assertFalse(torch.equal(before, parameters(alg.policy.actor)))

"""CPU regression tests for ROA branch isolation and checkpoint continuation.

Run with: conda run -n isaacsim-5.1 python scripts/tests/test_roa_training.py
"""
from __future__ import annotations

import io
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))

import torch
from tensordict import TensorDict

from rsl_rl.algorithms.roa_ppo import ROAPPO
from rsl_rl.runners.on_policy_runner_roa import OnPolicyRunnerROA


class SyntheticWheelEnv:
    num_envs = 8
    num_actions = 10
    device = "cpu"

    def get_observations(self):
        return TensorDict(
            {"policy": torch.randn(8, 10, 39), "critic": torch.randn(8, 72), "privileged": torch.randn(8, 33)},
            batch_size=[8],
        )

    def step(self, actions):
        assert actions.shape == (8, 10)
        return self.get_observations(), torch.randn(8), torch.zeros(8, dtype=torch.long), {}


def make_runner(*, policy_overrides=None, **algorithm_overrides):
    cfg = {
        "num_steps_per_env": 2,
        "save_interval": 1000,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"], "privileged": ["privileged"]},
        "policy": {
            "class_name": "ActorCriticROA", "priv_encoder_dims": [64, 20],
            "actor_hidden_dims": [32], "critic_hidden_dims": [32], "vel_offset": 39,
            **(policy_overrides or {}),
        },
        "algorithm": {
            "class_name": "ROAPPO", "num_learning_epochs": 2, "num_mini_batches": 2,
            "dagger_update_freq": 20, "priv_reg_coef_schedule": [0., .1, 2, 5],
            "priv_reg_coef_schedule_resume": None, **algorithm_overrides,
        },
    }
    return OnPolicyRunnerROA(SyntheticWheelEnv(), cfg, device="cpu")


def parameters(module):
    return torch.cat([p.detach().flatten() for p in module.parameters()]).clone()


def checkpoint(runner):
    buffer = io.BytesIO()
    runner.save(buffer, infos={"sentinel": 42})
    buffer.seek(0)
    return buffer


class RoaTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_student_and_teacher_parameter_isolation(self):
        r = make_runner()
        p = r.alg.policy
        teacher = {name: parameters(getattr(p, name)) for name in ["actor", "critic", "priv_encoder"]}
        std = p.std.detach().clone()
        history = parameters(p.history_encoder)
        with patch.object(r.alg, "update", wraps=r.alg.update) as ppo:
            r.learn(1)
            ppo.assert_not_called()
        for name, before in teacher.items():
            self.assertTrue(torch.equal(before, parameters(getattr(p, name))))
        self.assertTrue(torch.equal(std, p.std))
        self.assertFalse(torch.equal(history, parameters(p.history_encoder)))
        self.assertEqual((r.alg.counter, r.current_learning_iteration, r.alg.storage.step), (1, 1, 0))
        history = parameters(p.history_encoder)
        with patch.object(r.alg, "update_dagger", wraps=r.alg.update_dagger) as dagger:
            r.learn(1)
            dagger.assert_not_called()
        self.assertTrue(torch.equal(history, parameters(p.history_encoder)))
        for name, before in teacher.items():
            self.assertFalse(torch.equal(before, parameters(getattr(p, name))))
        self.assertEqual((r.alg.counter, r.current_learning_iteration, r.alg.storage.step), (2, 2, 0))
        # Reach the next student rollout: no stale storage or double increments.
        r.learn(19)
        self.assertEqual((r.alg.counter, r.current_learning_iteration, r.alg.storage.step), (21, 21, 0))
        self.assertTrue(all(torch.isfinite(t).all() for t in p.state_dict().values()))

    def test_regularizer_updates_teacher_without_history_gradients(self):
        r = make_runner(priv_reg_coef_schedule=[1., 1., 0, 1], value_loss_coef=0., entropy_coef=0., schedule="fixed")
        a = r.alg
        with torch.inference_mode():
            for _ in range(r.num_steps_per_env):
                obs = r.env.get_observations()
                a.act(obs, hist_encoding=False)
                a.process_env_step(obs, torch.zeros(8), torch.zeros(8), {})
            a.compute_returns(obs)
            a.storage.advantages.zero_()
        history, teacher = parameters(a.policy.history_encoder), parameters(a.policy.priv_encoder)
        losses = a.update()
        self.assertGreater(losses["priv_reg"], 0)
        self.assertFalse(torch.equal(teacher, parameters(a.policy.priv_encoder)))
        self.assertTrue(torch.equal(history, parameters(a.policy.history_encoder)))
        self.assertTrue(all(p.grad is None for p in a.policy.history_encoder.parameters()))

    def test_resume_matches_both_next_updates(self):
        r = make_runner(dagger_update_freq=2)
        r.learn(2)
        r.alg.learning_rate = .0002
        for group in r.alg.optimizer.param_groups:
            group["lr"] = .0002
        for group in r.alg.hist_encoder_optimizer.param_groups:
            group["lr"] = .0003
        r.alg.priv_reg_coef_schedule = [.02, .2, 1, 8]
        restored = make_runner(dagger_update_freq=2)
        self.assertEqual(restored.load(checkpoint(r)), {"sentinel": 42})
        self.assertEqual(restored.alg.learning_rate, .0002)
        self.assertEqual(restored.alg.priv_reg_coef_schedule, r.alg.priv_reg_coef_schedule)
        for seed in [1234, 5678]:  # student, then teacher
            torch.manual_seed(seed)
            r.learn(1)
            torch.manual_seed(seed)
            restored.learn(1)
            self.assert_nested_equal(r.alg.policy.state_dict(), restored.alg.policy.state_dict())
            self.assert_nested_equal(r.alg.optimizer.state_dict(), restored.alg.optimizer.state_dict())
            self.assert_nested_equal(r.alg.hist_encoder_optimizer.state_dict(), restored.alg.hist_encoder_optimizer.state_dict())
            self.assertEqual(r.alg.counter, restored.alg.counter)
            self.assertEqual(r.current_learning_iteration, restored.current_learning_iteration)

    def test_legacy_migration_and_explicit_override(self):
        r = make_runner()
        r.learn(2)
        legacy = torch.load(checkpoint(r), weights_only=False)
        for key in ["iteration_is_next", "algorithm_counter", "hist_encoder_optimizer_state_dict", "priv_reg_coef_schedule"]:
            legacy.pop(key)
        legacy["iter"] = 20
        buffer = io.BytesIO()
        torch.save(legacy, buffer)
        buffer.seek(0)
        restored = make_runner()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.load(buffer)
        self.assertEqual(len(caught), 4)
        self.assertEqual((restored.alg.counter, restored.current_learning_iteration), (21, 21))
        self.assertEqual(restored.alg.hist_encoder_optimizer.state, {})
        self.assertEqual(restored.alg.priv_reg_coef_schedule, [0., .1, 2, 5])
        again = make_runner(priv_reg_coef_schedule_resume=[.3, .3, 0, 1])
        again.load(checkpoint(restored))
        self.assertEqual((again.alg.counter, again.current_learning_iteration), (21, 21))
        self.assertEqual(again.alg.priv_reg_coef_schedule, [.3, .3, 0, 1])

    def test_load_without_optimizers(self):
        r = make_runner()
        r.learn(2)
        restored = make_runner()
        restored.load(checkpoint(r), load_optimizer=False)
        self.assert_nested_equal(r.alg.policy.state_dict(), restored.alg.policy.state_dict())
        self.assertEqual(restored.alg.optimizer.state, {})
        self.assertEqual(restored.alg.hist_encoder_optimizer.state, {})

    def test_final_filename_and_next_iteration_metadata(self):
        r = make_runner()
        r.log_dir = "unused-mocked-log-dir"
        r._prepare_logging_writer = lambda: None
        r.logger_type = "tensorboard"
        r.log = lambda values: None
        saved = []
        with patch.object(r, "save", side_effect=lambda path: saved.append((Path(path).name, r.current_learning_iteration))), patch(
            "rsl_rl.runners.on_policy_runner_roa.store_code_state", return_value=[]
        ):
            r.learn(2)
        self.assertEqual(saved, [("model_0.pt", 1), ("model_1.pt", 2)])
        state = torch.load(checkpoint(r), weights_only=False)
        self.assertTrue(state["iteration_is_next"])
        self.assertEqual(state["iter"], 2)

    def test_invalid_configuration(self):
        for frequency in [0, -1, 1.5, True]:
            with self.subTest(frequency=frequency), self.assertRaises(ValueError):
                make_runner(dagger_update_freq=frequency)
        for schedule in [[0, .1, 0], [0, .1, 0, 0], [0, .1, -1, 1], [0, float("nan"), 0, 1], [0, True, 0, 1]]:
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                ROAPPO.validate_priv_reg_schedule(schedule)
        with self.assertRaises(ValueError):
            make_runner(priv_reg_coef_schedule_resume=[0, .1, 0])

    def test_dagger_without_optimizer_releases_storage(self):
        r = make_runner()
        r.alg.hist_encoder_optimizer = None
        r.learn(1)
        self.assertEqual((r.alg.counter, r.current_learning_iteration, r.alg.storage.step), (1, 1, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)

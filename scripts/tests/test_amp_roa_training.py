"""CPU AMP-ROA integration checks; no simulator, motion files or training job.

Run with isaacsim-5.1: python -B -m unittest discover -s scripts/tests
-p 'test_amp_roa_training.py' -v
"""
from __future__ import annotations

import ast
import copy
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from rsl_rl.algorithms import ROAPPO, AMPROAPPO
from rsl_rl.runners import OnPolicyRunnerAmpROA
from test_roa_training import parameters, checkpoint
from test_roa_velocity_modes import deployment_wrapper
from test_velocity_diagnostics import metadata


class SyntheticAmpEnv:
    num_envs, num_actions, device, step_dt = 8, 10, "cpu", .02

    def __init__(self):
        self.unwrapped = metadata(leg=True).unwrapped

    def get_observations(self):
        critic = torch.randn(self.num_envs, 74)
        critic[:3, 6:9] = 0
        critic[3:5, 6:9] = .01
        return TensorDict({"policy": torch.randn(self.num_envs, 10, 41),
                           "critic": critic, "privileged": torch.randn(self.num_envs, 30),
                           "amp": torch.randn(self.num_envs, 2, 4)}, [self.num_envs])

    def step(self, actions):
        assert actions.shape == (self.num_envs, self.num_actions)
        dones = torch.zeros(self.num_envs, dtype=torch.long)
        dones[0] = 1  # No terminal observation: exclude this transition from AMP replay.
        return self.get_observations(), torch.rand(self.num_envs), dones, {"time_outs": dones.float()}


class SyntheticExpert:
    observation_dim, frame_dim = 8, 4

    def __init__(self, *args, **kwargs):
        pass

    def feed_forward_generator(self, batches, size):
        for _ in range(batches):
            yield torch.randn(size, 8), torch.randn(size, 8)


def make_amp_runner(enabled=True, schedule=None, diagnostics=None, **overrides):
    cfg = {
        "num_steps_per_env": 2, "save_interval": 1000,
        "obs_groups": {k: [k] for k in ("policy", "critic", "privileged", "amp")},
        "policy": {"class_name": "ActorCriticROA", "priv_encoder_dims": [64, 20],
                   "actor_hidden_dims": [32], "critic_hidden_dims": [32],
                   "vel_offset": 41, "use_velocity_estimation": enabled},
        "algorithm": {"class_name": "AMPROAPPO", "num_learning_epochs": 2, "num_mini_batches": 2,
                      "dagger_update_freq": 20, "priv_reg_coef_schedule": [0., .1, 2, 5],
                      "priv_reg_coef_schedule_resume": None,
                      "estimated_velocity_schedule": schedule,
                      "estimated_velocity_schedule_resume": None, **overrides},
        "velocity_diagnostics": diagnostics,
        "amp_num_preload_transitions": 32, "amp_motion_files": [],
        "amp_discriminator_history_window": True, "amp_reward_coef": 2.,
        "amp_task_reward_lerp": .3, "amp_discr_hidden_dims": [16],
        "amp_replay_buffer_size": 128, "min_normalized_std": [0.] * 10,
    }
    with patch("rsl_rl.runners.on_policy_runner_amp.AMPLoader", SyntheticExpert), redirect_stdout(io.StringIO()):
        return OnPolicyRunnerAmpROA(SyntheticAmpEnv(), cfg, device="cpu")


class AmpRoaTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def same(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.same(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.same(x, y)
        else:
            self.assertEqual(a, b)

    def amp_state(self, r):
        return copy.deepcopy({"disc": r.alg.discriminator.state_dict(),
                              "optimizer": r.alg.amp_optimizer.state_dict(),
                              "normalizer": vars(r.alg.amp_normalizer),
                              "replay_states": r.alg.amp_storage.states,
                              "replay_next": r.alg.amp_storage.next_states,
                              "replay_count": r.alg.amp_storage.num_samples})

    def test_student_teacher_isolation_and_disabled_velocity(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                r = make_amp_runner(enabled, [0., 1., 2, 5], diagnostics={})
                p = r.alg.policy
                teacher = {k: parameters(getattr(p, k)) for k in ("actor", "critic", "priv_encoder")}
                hist, amp = parameters(p.history_encoder), self.amp_state(r)
                with patch.object(r.alg, "update_dagger", wraps=r.alg.update_dagger) as dagger, \
                     patch.object(r.alg.discriminator, "predict_amp_reward", wraps=r.alg.discriminator.predict_amp_reward) as reward:
                    r.learn(1)
                    dagger.assert_called_once()
                    reward.assert_not_called()
                for k, before in teacher.items():
                    self.same(before, parameters(getattr(p, k)))
                self.assertFalse(torch.equal(hist, parameters(p.history_encoder)))
                self.same(amp, self.amp_state(r))
                if not enabled:
                    self.assertIsNone(r.alg.estimated_velocity_schedule)
                    self.assertEqual(r.last_velocity_diagnostics, {})
                    self.assertFalse(any('vel_output' in k for k in p.state_dict()))
                hist = parameters(p.history_encoder)
                before = parameters(r.alg.discriminator)
                with patch.object(r.alg, "update", wraps=r.alg.update) as ppo:
                    r.learn(1)
                    ppo.assert_called_once()
                self.same(hist, parameters(p.history_encoder))
                self.assertFalse(torch.equal(before, parameters(r.alg.discriminator)))
                self.assertEqual(r.alg.amp_storage.num_samples, 14)
                self.assertEqual(r.alg.counter, 2)

    def test_curriculum_cached_inputs_and_exact_log_probability_replay(self):
        for probability in (0., .5, 1.):
            r = make_amp_runner(schedule=[probability, probability, 0, 1])
            a, p = r.alg, r.alg.policy
            with torch.inference_mode():
                obs = r.env.get_observations()
                a.act(obs, amp_obs=obs['amp'].flatten(1))
                self.assertNotIn(a.VELOCITY_INPUT_KEY, obs)
                self.assertNotIn('amp', a.transition.observations)
                cached = a.transition.observations.clone()
                old_log_prob = a.transition.actions_log_prob.clone()
                actions = a.transition.actions.clone()
                mask = cached[a.VELOCITY_MASK_KEY]
                _, estimate = p.infer_hist_latent(obs, return_vel=True)
                self.same(cached[a.VELOCITY_INPUT_KEY], torch.where(mask, estimate, p.get_true_vel(obs)))
                p.history_encoder.vel_output[0].bias.add_(100)
                p.update_distribution(cached, velocity_override=cached[a.VELOCITY_INPUT_KEY])
                self.same(old_log_prob, p.get_actions_log_prob(actions))
            p.zero_grad(set_to_none=True)
            p.update_distribution(cached, velocity_override=cached[a.VELOCITY_INPUT_KEY])
            p.action_mean.sum().backward()
            self.assertTrue(all(x.grad is None for x in p.history_encoder.parameters()))

    def test_zero_probability_matches_legacy_path_and_diagnostics_do_not_change_amp(self):
        runs = []
        for schedule, diagnostics in ((None, None), ([0., 0., 0, 1], None), (None, {})):
            torch.manual_seed(7)
            r = make_amp_runner(schedule=schedule, diagnostics=diagnostics)
            r.learn(3)
            runs.append((r, torch.get_rng_state()))
        for r, rng in runs[1:]:
            self.same(runs[0][0].alg.policy.state_dict(), r.alg.policy.state_dict())
            self.same(runs[0][0].alg.optimizer.state_dict(), r.alg.optimizer.state_dict())
            self.same(runs[0][0].alg.hist_encoder_optimizer.state_dict(), r.alg.hist_encoder_optimizer.state_dict())
            self.same(self.amp_state(runs[0][0]), self.amp_state(r))
            self.same(runs[0][1], rng)
        self.assertEqual(runs[2][0].last_velocity_diagnostics['sample_count'], 16)

    def test_ppo_update_reuses_cached_velocity_after_estimator_changes(self):
        r = make_amp_runner(schedule=[.5, .5, 0, 1], learning_rate=0., desired_kl=None)
        a, p = r.alg, r.alg.policy
        with torch.inference_mode():
            obs = r.env.get_observations()
            for _ in range(r.num_steps_per_env):
                actions = a.act(obs, amp_obs=obs['amp'].flatten(1))
                obs, rewards, dones, extras = r.env.step(actions)
                a.process_env_step(obs, rewards, dones, extras, obs['amp'].flatten(1),
                                   defer_amp_reward=True)
            a.finalize_amp_rollout_rewards()
            a.compute_returns(obs)
            p.history_encoder.vel_output[0].bias.add_(100)
        checked = []
        original = a.storage.mini_batch_generator
        def batches(*args, **kwargs):
            for batch in original(*args, **kwargs):
                checked.append((batch[1], batch[5]))
                yield batch
        act = p.act
        def replay(obs, **kwargs):
            self.same(kwargs['velocity_override'], obs[a.VELOCITY_INPUT_KEY])
            result = act(obs, **kwargs)
            actions, old_log_prob = checked[-1]
            # Shuffling into minibatches may change CPU GEMM rounding; the
            # cached velocity itself above must remain bit-identical.
            torch.testing.assert_close(p.get_actions_log_prob(actions), old_log_prob.flatten(),
                                       rtol=1e-6, atol=4e-6)
            return result
        with patch.object(a.storage, 'mini_batch_generator', batches), patch.object(p, 'act', replay):
            a.update()
        self.assertEqual(len(checked), a.num_learning_epochs * a.num_mini_batches)

    def test_checkpoint_schedules_optimizers_and_next_branch(self):
        r = make_amp_runner(schedule=[0., 1., 2, 5])
        r.learn(2)
        saved = checkpoint(r)
        restored = make_amp_runner()
        restored.load(saved)
        self.assertEqual(restored.alg.estimated_velocity_schedule, [0., 1., 2, 5])
        self.assertEqual(restored.alg.priv_reg_coef_schedule, r.alg.priv_reg_coef_schedule)
        self.assertEqual((restored.current_learning_iteration, restored.alg.counter), (2, 2))
        self.assertIn(restored.alg.VELOCITY_INPUT_KEY, restored.alg.storage.observations)
        self.same(r.alg.policy.state_dict(), restored.alg.policy.state_dict())
        for name in ('optimizer', 'hist_encoder_optimizer', 'amp_optimizer'):
            self.same(getattr(r.alg, name).state_dict(), getattr(restored.alg, name).state_dict())
        self.same(vars(r.alg.amp_normalizer), vars(restored.alg.amp_normalizer))
        with patch.object(restored.alg, 'update', wraps=restored.alg.update) as ppo:
            restored.learn(1)
            ppo.assert_called_once()
        override = make_amp_runner(priv_reg_coef_schedule_resume=[.2, .2, 0, 1],
                                   estimated_velocity_schedule_resume=[.5, 1., 2, 4])
        override.load(checkpoint(r), load_optimizer=False)
        self.assertEqual(override.alg.priv_reg_coef_schedule, [.2, .2, 0, 1])
        self.assertEqual(override.alg.estimated_velocity_probability(), .5)
        self.assertEqual(override.alg.counter, 2)

    def test_legacy_checkpoint_and_saved_none(self):
        r = make_amp_runner()
        r.learn(2)
        target = make_amp_runner(schedule=[0., 1., 2, 5])
        target.load(checkpoint(r))
        self.assertIsNone(target.alg.estimated_velocity_schedule)
        old = torch.load(checkpoint(r), weights_only=False)
        for key in ('priv_reg_coef_schedule', 'estimated_velocity_schedule', 'use_velocity_estimation',
                    'algorithm_counter', 'iteration_is_next'):
            old.pop(key)
        old['iter'] = 19
        buffer = io.BytesIO()
        torch.save(old, buffer)
        buffer.seek(0)
        target = make_amp_runner(schedule=[0., 1., 2, 5])
        with self.assertWarnsRegex(UserWarning, 'lacks regularization schedule'):
            target.load(buffer)
        self.assertEqual((target.current_learning_iteration, target.alg.counter), (20, 20))
        self.assertEqual(target.alg.estimated_velocity_probability(), 1.)
        with patch.object(target.alg, 'update_dagger', wraps=target.alg.update_dagger) as dagger:
            target.learn(1)
            dagger.assert_called_once()

    def test_checkpoint_mode_mismatch_rejected_before_mutation(self):
        enabled, disabled = make_amp_runner(), make_amp_runner(False)
        before = parameters(disabled.alg.policy)
        with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
            disabled.load(checkpoint(enabled))
        self.same(before, parameters(disabled.alg.policy))
        restored = make_amp_runner(False)
        restored.load(checkpoint(disabled))
        self.same(before, parameters(restored.alg.policy))

    def test_grouped_diagnostics_and_curriculum_logging(self):
        r = make_amp_runner(schedule=[1., 1., 0, 1], diagnostics={})
        r.log_dir, r.writer, r.logger_type = 'unused', Mock(), 'tensorboard'
        with patch.object(r, 'log'), patch.object(r, 'save'), \
             patch('rsl_rl.runners.on_policy_runner_amp_roa.store_code_state', return_value=[]):
            r.learn(2)
        tags = [call.args[0] for call in r.writer.add_scalar.call_args_list]
        for branch in ('teacher', 'student'):
            self.assertIn(f'VelocityDiagnostics/{branch}/all/rmse_vx_m_s', tags)
            self.assertIn(f'VelocityCurriculum/{branch}/estimated_fraction', tags)
        self.assertEqual(r.last_velocity_curriculum['estimated_fraction'], 1.)
        self.assertEqual(r.last_velocity_diagnostics['sample_count'], 16)

    def test_shared_validation_and_task_config(self):
        self.assertIs(AMPROAPPO.update_dagger, ROAPPO.update_dagger)
        for invalid in ([0., 2., 0, 1], [0., 1., 0, 0], [0., 1., .5, 2]):
            with self.assertRaises(ValueError):
                make_amp_runner(schedule=invalid)
        for invalid in (0, True, -1, 1.5):
            with self.assertRaises(ValueError):
                make_amp_runner(dagger_update_freq=invalid)
        with self.assertRaises(ValueError):
            make_amp_runner(priv_reg_coef_schedule_resume=[0., .1, 0, 0])
        path = ROOT / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/LW/LW_Leg/agents/rsl_rl_amp_roa_cfg.py'
        tree = ast.parse(path.read_text())
        values = [ast.literal_eval(n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.keyword) and n.arg == 'estimated_velocity_schedule']
        self.assertEqual(values, [[0., 1., 2000, 3000]])

    def test_amp_policy_native_jit_onnx_parity_with_and_without_velocity(self):
        import onnxruntime
        for enabled in (True, False):
            r = make_amp_runner(enabled)
            p = r.alg.policy.eval()
            wrapper = deployment_wrapper()(p, 41).eval()
            sample = torch.zeros(1, 410)
            jit = torch.jit.trace(wrapper, sample)
            buffer = io.BytesIO()
            torch.onnx.export(wrapper, sample, buffer, dynamo=False, input_names=['history'],
                              output_names=['actions'], dynamic_axes={'history': {0: 'batch'}, 'actions': {0: 'batch'}})
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            onnx = onnxruntime.InferenceSession(buffer.getvalue(), sess_options=options,
                                               providers=['CPUExecutionProvider'])
            for reset in (False, True):
                obs = r.env.get_observations()
                if reset:
                    p.reset(torch.ones(8))
                    obs['policy'].zero_()
                history = obs['policy'].flatten(1)
                with torch.no_grad():
                    expected = p.act_inference(obs, hist_encoding=True)
                    torch.testing.assert_close(wrapper(history), expected)
                    torch.testing.assert_close(jit(history), expected)
                exported = onnx.run(None, {'history': history.numpy()})[0]
                torch.testing.assert_close(torch.from_numpy(exported), expected, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

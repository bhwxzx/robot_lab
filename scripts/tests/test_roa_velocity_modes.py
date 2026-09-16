"""CPU tests for ROA velocity ablation, curriculum, checkpoint and export.

Run in isaacsim-5.1 with python -B -m unittest discover -s scripts/tests
-p 'test_roa_velocity_modes.py' -v. No simulator, training job or disk export.
"""
from __future__ import annotations

import ast
import copy
import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from test_roa_training import make_runner, checkpoint, parameters

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
from rsl_rl.algorithms.roa_ppo import ROAPPO
from policy_evaluation_telemetry import capture_roa_diagnostics, roa_diagnostic_signals
from policy_evaluation_evidence import validate_roa_velocity_mode, EvaluationEvidenceError


def deployment_wrapper():
    """Load the actual export class without importing/launching Isaac Sim."""
    path = ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ROADeploymentWrapper")
    module = types.ModuleType("roa_export_test_wrapper")
    module.__file__ = str(path)
    module.torch, module.nn = torch, torch.nn
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.ROADeploymentWrapper


class RoaVelocityModeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def runner(self, enabled=True, schedule=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return make_runner(policy_overrides={"use_velocity_estimation": enabled},
                               estimated_velocity_schedule=schedule, **kwargs)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def assert_training_equal(self, left, right):
        for part in ["policy", "optimizer", "hist_encoder_optimizer"]:
            self.assert_nested_equal(getattr(left.alg, part).state_dict(), getattr(right.alg, part).state_dict())
        self.assertEqual(left.alg.counter, right.alg.counter)
        self.assertEqual(left.current_learning_iteration, right.current_learning_iteration)

    def collect_teacher(self, runner):
        observations = runner.env.get_observations()
        with torch.inference_mode():
            for _ in range(runner.num_steps_per_env):
                original_keys = set(observations.keys())
                actions = runner.alg.act(observations, hist_encoding=False)
                self.assertEqual(set(observations.keys()), original_keys)
                observations, rewards, dones, extras = runner.env.step(actions)
                runner.alg.process_env_step(observations, rewards, dones, extras)
            runner.alg.compute_returns(observations)

    def test_disabled_architecture_has_no_speed_head_or_actor_speed_input(self):
        runner = self.runner(enabled=False)
        policy = runner.alg.policy
        self.assertFalse(any("vel_output" in name for name in policy.state_dict()))
        observations = runner.env.get_observations()
        changed = observations.clone()
        changed["critic"][:, 39:42] += 100
        inputs = []
        handle = policy.actor.register_forward_pre_hook(lambda module, args: inputs.append(args[0].clone()))
        try:
            for history in [False, True]:
                before = policy.act_inference(observations, hist_encoding=history)
                after = policy.act_inference(changed, hist_encoding=history)
                torch.testing.assert_close(before, after, rtol=0, atol=0)
        finally:
            handle.remove()
        self.assertTrue(all(x.shape == (8, 59) for x in inputs))
        self.assertEqual(policy.get_true_vel(observations).shape, (8, 3))
        with self.assertRaisesRegex(ValueError, "disabled"):
            policy.infer_hist_latent(observations, return_vel=True)
        with self.assertRaisesRegex(ValueError, "override"):
            policy.act(observations, velocity_override=torch.zeros(8, 3))

    def test_disabled_training_keeps_latent_updates_and_skips_velocity_diagnostics(self):
        runner = self.runner(enabled=False, schedule=[0., 1., 0, 2])
        self.assertIsNone(runner.alg.estimated_velocity_schedule)
        runner.cfg["velocity_diagnostics"] = {}  # No manager metadata needed when disabled.
        actor = parameters(runner.alg.policy.actor)
        history = parameters(runner.alg.policy.history_encoder)
        losses = []
        original = runner.alg.update_dagger

        def update():
            value = original()
            losses.append(value)
            return value

        with patch.object(runner.alg, "update_dagger", side_effect=update):
            runner.learn(1)
        self.assertEqual(set(losses[0]), {"hist_latent"})
        self.assertTrue(torch.equal(actor, parameters(runner.alg.policy.actor)))
        self.assertFalse(torch.equal(history, parameters(runner.alg.policy.history_encoder)))
        history = parameters(runner.alg.policy.history_encoder)
        runner.learn(1)
        self.assertTrue(torch.equal(history, parameters(runner.alg.policy.history_encoder)))
        self.assertFalse(torch.equal(actor, parameters(runner.alg.policy.actor)))
        self.assertEqual(runner.last_velocity_diagnostics, {})
        self.assertEqual(runner.last_velocity_curriculum, {})

    def test_schedule_boundaries_and_invalid_settings(self):
        runner = self.runner(schedule=[.1, .9, 2, 4])
        for counter, expected in [(0, .1), (2, .1), (3, .3), (4, .5), (6, .9), (10, .9)]:
            runner.alg.counter = counter
            self.assertAlmostEqual(runner.alg.estimated_velocity_probability(), expected)
        for invalid in [[], [0, 1, 0], [-.1, 1, 0, 1], [0, 1.1, 0, 1], [1, 0, 0, 1],
                        [0, 1, -1, 1], [0, 1, 0, 0], [0, 1, 1.2, 1], [0, 1, 0, 1.2],
                        [False, 1, 0, 1], [0, float("nan"), 0, 1], [0, 1, 0, float("inf")]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                ROAPPO.validate_velocity_schedule(invalid)
        with self.assertRaises(ValueError):
            self.runner(enabled="false")
        runner.alg.symmetry = {"use_mirror_loss": True}
        with self.assertRaisesRegex(ValueError, "symmetry"):
            runner.alg.set_estimated_velocity_schedule([0, 1, 0, 1])

    def test_rollout_stores_exact_chosen_velocity_and_replays_without_reestimation(self):
        for probability in [0., .5, 1.]:
            with self.subTest(probability=probability):
                runner = self.runner(schedule=[probability, probability, 0, 1])
                algorithm = runner.alg
                self.collect_teacher(runner)
                batch = algorithm.storage.observations.flatten(0, 1)
                stored_velocity = batch[algorithm.VELOCITY_INPUT_KEY]
                mask = batch[algorithm.VELOCITY_MASK_KEY]
                with torch.no_grad():
                    _, predicted = algorithm.policy.infer_hist_latent(batch, return_vel=True)
                    actual = algorithm.policy.get_true_vel(batch)
                    torch.testing.assert_close(stored_velocity, torch.where(mask, predicted, actual), atol=1e-7, rtol=1e-6)
                if probability in (0., 1.):
                    self.assertTrue((mask == bool(probability)).all())
                else:
                    self.assertTrue(mask.any() and (~mask).any())
                self.assertFalse(stored_velocity.requires_grad)
                # A later estimator change must not change the replayed PPO input.
                with torch.no_grad():
                    algorithm.policy.history_encoder.vel_output[0].bias.add_(10)
                with torch.no_grad():
                    algorithm.policy.update_distribution(batch, velocity_override=stored_velocity)
                    torch.testing.assert_close(algorithm.policy.action_mean, algorithm.storage.mu.flatten(0, 1),
                                               rtol=1e-5, atol=1e-6)
                    log_prob = algorithm.policy.get_actions_log_prob(algorithm.storage.actions.flatten(0, 1))
                    torch.testing.assert_close(log_prob, algorithm.storage.actions_log_prob.flatten(),
                                               rtol=1e-5, atol=1e-6)
                history = parameters(algorithm.policy.history_encoder)
                actor = parameters(algorithm.policy.actor)
                original = algorithm.policy.infer_hist_latent

                def latent_only(obs, return_vel=False):
                    self.assertFalse(return_vel, "PPO must not recompute cached velocity")
                    return original(obs, return_vel=return_vel)

                with patch.object(algorithm.policy, "infer_hist_latent", side_effect=latent_only):
                    algorithm.update()
                self.assertTrue(torch.equal(history, parameters(algorithm.policy.history_encoder)))
                self.assertFalse(torch.equal(actor, parameters(algorithm.policy.actor)))
                self.assertTrue(all(p.grad is None for p in algorithm.policy.history_encoder.parameters()))

    def test_zero_probability_matches_legacy_updates_and_rng(self):
        legacy = self.runner()
        torch.manual_seed(42)
        curriculum = self.runner(schedule=[0., 0., 0, 1])
        for seed in [20, 30]:
            torch.manual_seed(seed)
            legacy.learn(1)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            curriculum.learn(1)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assert_training_equal(legacy, curriculum)

    def test_curriculum_logging_preserves_student_teacher_schedule(self):
        runner = self.runner(schedule=[0., 1., 0, 2])
        runner.log_dir = "unused-mocked-log-dir"
        runner._prepare_logging_writer = lambda: None
        runner.logger_type = "tensorboard"
        runner.writer, runner.log = Mock(), Mock()
        with patch.object(runner, "save"), patch("rsl_rl.runners.on_policy_runner_roa.store_code_state", return_value=[]), \
             patch.object(runner.alg, "update_dagger", wraps=runner.alg.update_dagger) as dagger, \
             patch.object(runner.alg, "update", wraps=runner.alg.update) as ppo:
            runner.learn(3)
        self.assertEqual((dagger.call_count, ppo.call_count), (1, 2))
        logs = {call.args[0:3:2]: call.args[1] for call in runner.writer.add_scalar.call_args_list}
        self.assertEqual(logs[("VelocityCurriculum/student/probability", 0)], 1.)
        self.assertEqual(logs[("VelocityCurriculum/teacher/probability", 1)], .5)
        self.assertEqual(logs[("VelocityCurriculum/teacher/estimated_fraction", 2)], 1.)
        self.assertEqual(runner.last_velocity_curriculum["sample_count"], 16)

    def test_resume_restores_course_and_matches_subsequent_updates(self):
        original = self.runner(schedule=[0., 1., 0, 8], dagger_update_freq=2)
        original.learn(3)
        restored = self.runner(dagger_update_freq=2)
        restored.load(checkpoint(original))
        self.assertEqual(restored.alg.estimated_velocity_schedule, [0., 1., 0, 8])
        self.assertAlmostEqual(restored.alg.estimated_velocity_probability(), 3 / 8)
        self.assertIn(restored.alg.VELOCITY_INPUT_KEY, restored.alg.storage.observations)
        for seed in [50, 60]:  # Teacher mixture, then student update.
            torch.manual_seed(seed)
            original.learn(1)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            restored.learn(1)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assert_training_equal(original, restored)
        override = self.runner(estimated_velocity_schedule_resume=[.2, .6, 0, 10])
        override.load(checkpoint(original))
        self.assertEqual(override.alg.counter, original.alg.counter)
        self.assertEqual(override.alg.estimated_velocity_schedule, [.2, .6, 0, 10])

    def test_checkpoint_architecture_guard_and_legacy_loading(self):
        enabled, disabled = self.runner(), self.runner(enabled=False)
        for target, source in [(enabled, disabled), (disabled, enabled)]:
            before = copy.deepcopy(target.alg.policy.state_dict())
            with self.assertRaisesRegex(ValueError, "architecture mismatch"):
                target.load(checkpoint(source))
            self.assert_nested_equal(before, target.alg.policy.state_dict())
        disabled.learn(2)
        restored = self.runner(enabled=False)
        restored.load(checkpoint(disabled))
        self.assert_training_equal(disabled, restored)
        legacy = torch.load(checkpoint(enabled), weights_only=False)
        legacy.pop("use_velocity_estimation")
        legacy.pop("estimated_velocity_schedule")
        buffer = io.BytesIO()
        torch.save(legacy, buffer)
        buffer.seek(0)
        enabled.load(buffer)
        self.assertIsNone(enabled.alg.estimated_velocity_schedule)

    def test_disabled_telemetry_omits_estimates_and_rejects_velocity_ablations(self):
        runner = self.runner(enabled=False)
        obs = runner.env.get_observations()
        actual = obs["critic"][:, 39:42] / 2
        values = capture_roa_diagnostics(runner.alg.policy, obs, actual, 2.)
        self.assertEqual(set(values), set(roa_diagnostic_signals("student", False)))
        self.assertNotIn("roa_student_velocity_b", values)
        self.assertNotIn("roa_teacher_velocity_b", values)
        for mode in ["student_true_velocity", "student_teacher_latent"]:
            with self.assertRaises(ValueError):
                roa_diagnostic_signals(mode, False)
        with self.assertRaises(ValueError):
            capture_roa_diagnostics(runner.alg.policy, obs, actual, 2., include_ablations=True)

    def test_disabled_evidence_requires_matching_bound_configuration(self):
        self.assertTrue(validate_roa_velocity_mode({}, None))
        with self.assertRaises(EvaluationEvidenceError):
            validate_roa_velocity_mode({"use_velocity_estimation": False}, None)
        for configured, declared, valid in [("false", False, True), ("true", True, True),
                                            ("true", False, False), ("false", True, False),
                                            ('"false"', False, False)]:
            config = {"source_files": {"agent": {"content_utf8": f"policy:\n  use_velocity_estimation: {configured}\n"}}}
            with self.subTest(configured=configured, declared=declared), \
                 patch("policy_evaluation_evidence._validate_reference", return_value={"path": "/unused"}), \
                 patch("policy_evaluation_evidence._load_json_object", return_value=config):
                if valid:
                    self.assertEqual(validate_roa_velocity_mode({"use_velocity_estimation": declared}, {}), declared)
                else:
                    with self.assertRaises(EvaluationEvidenceError):
                        validate_roa_velocity_mode({"use_velocity_estimation": declared}, {})
        with self.assertRaises(EvaluationEvidenceError):
            validate_roa_velocity_mode({"use_velocity_estimation": "false"}, None)

    def test_native_wrapper_jit_and_onnx_parity_both_architectures(self):
        import onnx
        import onnxruntime

        wrapper_class = deployment_wrapper()
        for enabled in [True, False]:
            with self.subTest(enabled=enabled):
                runner = self.runner(enabled=enabled)
                policy = runner.alg.policy.eval()
                wrapper = wrapper_class(policy, 39).eval()
                example = torch.zeros(1, 390)
                traced = torch.jit.trace(wrapper, example)
                jit_buffer = io.BytesIO()
                torch.jit.save(traced, jit_buffer)
                jit_buffer.seek(0)
                traced = torch.jit.load(jit_buffer)
                onnx_buffer = io.BytesIO()
                torch.onnx.export(wrapper, example, onnx_buffer, dynamo=False,
                                  input_names=["history"], output_names=["actions"],
                                  dynamic_axes={"history": {0: "batch"}, "actions": {0: "batch"}})
                onnx.checker.check_model(onnx.load_model_from_string(onnx_buffer.getvalue()))
                session_options = onnxruntime.SessionOptions()
                session_options.intra_op_num_threads = 1
                session_options.inter_op_num_threads = 1
                session = onnxruntime.InferenceSession(onnx_buffer.getvalue(), sess_options=session_options,
                                                       providers=["CPUExecutionProvider"])
                for reset in [False, True]:
                    observations = runner.env.get_observations()
                    if reset:
                        policy.reset(torch.ones(8))
                        observations["policy"].zero_()
                    history = observations["policy"].flatten(1, 2)
                    with torch.no_grad():
                        expected = policy.act_inference(observations, hist_encoding=True)
                        torch.testing.assert_close(wrapper(history), expected)
                        torch.testing.assert_close(traced(history), expected)
                    exported = session.run(None, {"history": history.numpy()})[0]
                    torch.testing.assert_close(torch.from_numpy(exported), expected, rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    unittest.main(verbosity=2)

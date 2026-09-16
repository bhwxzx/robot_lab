"""CPU grouped-statistics and ROA runner integration tests (no simulator).

Run: conda run --no-capture-output -n isaacsim-5.1 python -B -m unittest discover \
    -s scripts/tests -p 'test_velocity_diagnostics.py' -v
"""
from __future__ import annotations

import copy
import io
import math
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rsl_rl"))

import torch

from rsl_rl.utils.velocity_diagnostics import (
    GroupedVelocityDiagnostics,
    VelocityDiagnosticConfig,
    critic_diagnostic_layout,
)
from test_roa_training import make_runner


def metadata(*, leg=False):
    """Match current IsaacLab manager fields, including resolved tensor scales."""
    names = ["base_ang_vel", "projected_gravity", "velocity_commands",
             "joint_pos", "joint_vel", "actions"]
    dims = [(3,), (3,), (3,), (10,), (10,), (10,)]
    if leg:
        names.append("gait_phase")
        dims.append((2,))
    names += ["base_lin_vel", "privileged"]
    dims += [(3,), (30,)]
    terms = {name: NS(scale=torch.tensor(scale), clip=(-100., 100.), history_length=0, modifiers=None)
             for name, scale in [("base_ang_vel", .25), ("velocity_commands", 1.), ("base_lin_vel", 2.)]}
    group = NS(enable_corruption=False, concatenate_terms=True, concatenate_dim=-1,
               history_length=None, **terms)
    manager = NS(active_terms={"critic": names}, group_obs_term_dim={"critic": dims}, cfg=NS(critic=group))
    # Deliberately different: the runtime manager configuration is authoritative.
    return NS(unwrapped=NS(observation_manager=manager, cfg=NS(observations=NS(critic=None))))


def accumulate(command, actual, predicted, yaw, **kwargs):
    accumulator = GroupedVelocityDiagnostics()
    accumulator.add(command, actual, predicted, yaw, **kwargs)
    return accumulator


class GroupedVelocityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_group_boundaries_and_independent_motion_partition(self):
        command = torch.tensor([
            [0., 0., 0.], [1e-6, -1e-6, 1e-6], [1.01e-6, 0., 0.],
            [.1, 0., .1], [0., 0., -.1], [.100001, 0., 0.],
            [0., 0., .100001], [.08, .08, 0.],
        ], dtype=torch.float64)
        actual = torch.zeros(8, 3, dtype=torch.float64)
        actual[0, 0] = .3  # Zero command is not equivalent to standing still.
        actual[1] = torch.tensor([.02, 0., 4.], dtype=torch.float64)  # Vertical motion is not part of this gate.
        actual[3, 1] = .020001
        yaw = torch.tensor([0., .05, .050001, 0., 0., 0., 0., 0.], dtype=torch.float64)
        r = accumulate(command, actual, actual, yaw).report()
        for group, count in [("command_zero", 2), ("command_low", 3), ("command_other", 3),
                             ("state_near_stationary", 5), ("state_moving", 3)]:
            self.assertEqual(r[f"{group}/sample_count"], count)
            self.assertEqual(r[f"{group}/fraction_valid"], count / 8)
        self.assertEqual(r["command_zero/near_stationary_fraction"], .5)

    def test_known_signed_bias_axis_rmse_and_means(self):
        actual = torch.tensor([[1., 2., 3.], [3., 4., 5.]], dtype=torch.float64)
        error = torch.tensor([[1., -2., 0.], [-3., 4., 2.]], dtype=torch.float64)
        r = accumulate(torch.zeros_like(actual), actual, actual + error, torch.zeros(2)).report()
        for axis, true_mean, estimate_mean, bias, rmse in [
            ("vx", 2., 1., -1., math.sqrt(5)), ("vy", 3., 4., 1., math.sqrt(10)),
            ("vz", 4., 5., 1., math.sqrt(2)),
        ]:
            self.assertEqual(r[f"all/actual_{axis}_mean_m_s"], true_mean)
            self.assertEqual(r[f"all/estimated_{axis}_mean_m_s"], estimate_mean)
            self.assertEqual(r[f"all/bias_{axis}_m_s"], bias)
            self.assertAlmostEqual(r[f"all/rmse_{axis}_m_s"], rmse)
        self.assertNotIn("command_low/rmse_vx_m_s", r)

    def test_nonfinite_and_clipped_samples_are_excluded_without_double_counting(self):
        command, actual, predicted = [torch.zeros(6, 3) for _ in range(3)]
        yaw = torch.zeros(6)
        command[0, 0] = float("nan")
        actual[1, 1] = float("inf")
        predicted[2, 2] = float("-inf")
        yaw[3] = float("nan")
        clipped = torch.tensor([True, False, False, False, True, False])
        r = accumulate(command, actual, predicted, yaw, clipped=clipped).report()
        self.assertEqual((r["sample_count"], r["invalid_sample_count"], r["clipped_sample_count"]), (6, 5, 2))
        self.assertEqual(r["valid_sample_count"], 1)
        self.assertEqual(r["valid_fraction"], 1 / 6)
        self.assertEqual(r["all/rmse_vx_m_s"], 0)
        self.assertTrue(all(math.isfinite(value) for value in r.values()))

    def test_empty_and_all_invalid_have_no_error_metrics(self):
        for n in (0, 4):
            with self.subTest(n=n):
                x = torch.full((n, 3), float("nan"))
                r = accumulate(x, x, x, torch.zeros(n)).report()
                self.assertEqual(r["sample_count"], n)
                self.assertEqual(r["valid_sample_count"], 0)
                self.assertFalse(any("rmse" in key or "bias" in key or "fraction_valid" in key for key in r))
                if n == 0:
                    self.assertNotIn("valid_fraction", r)

    def test_chunk_equivalence_and_weighted_distributed_reduction(self):
        command = torch.zeros(11, 3, dtype=torch.float64)
        command[1:4, 0] = .05
        command[4:, 0] = 1.
        actual = torch.randn(11, 3, dtype=torch.float64)
        predicted = actual + torch.arange(11, dtype=torch.float64)[:, None]
        yaw = torch.zeros(11)
        inputs = (command, actual, predicted, yaw)
        full = accumulate(*inputs).report()
        chunked = GroupedVelocityDiagnostics()
        for start in range(0, 11, 3):
            chunked.add(*(x[start:start + 3] for x in inputs))
        local = accumulate(*(x[:2] for x in inputs))
        remote = accumulate(*(x[2:] for x in inputs))
        before = local.stats.clone()

        def reduce(packed, op):
            self.assertEqual(op, torch.distributed.ReduceOp.SUM)
            packed += torch.cat((remote.counts, remote.stats.flatten()))

        with patch("torch.distributed.all_reduce", side_effect=reduce) as collective:
            reduced = local.report(distributed=True)
            collective.assert_called_once()
        for report in (chunked.report(), reduced):
            self.assertEqual(report.keys(), full.keys())
            for key in full:
                self.assertAlmostEqual(report[key], full[key], places=12, msg=key)
        torch.testing.assert_close(before, local.stats, rtol=0, atol=0)
        self.assertEqual(local.report()["sample_count"], 2)

    def test_invalid_config_and_shapes(self):
        for kwargs in [{"chunk_size": v} for v in [0, -1, True, 1.5]] + [
            {field: value} for field in ["zero_command_epsilon", "low_command_xy", "low_command_yaw",
                                       "near_stationary_xy", "near_stationary_yaw"]
            for value in [0, -1, True, float("nan"), float("inf")]
        ] + [{"zero_command_epsilon": .1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VelocityDiagnosticConfig(**kwargs)
        with self.assertRaises(ValueError):
            accumulate(torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 3), torch.zeros(2))
        with self.assertRaises(ValueError):
            accumulate(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2),
                       clipped=torch.zeros(2))


class CriticMetadataTests(unittest.TestCase):
    def test_wheel_and_leg_offsets_and_runtime_scales(self):
        for leg, offset in [(False, 39), (True, 41)]:
            with self.subTest(leg=leg):
                env = metadata(leg=leg)
                layout = critic_diagnostic_layout(env, NS(vel_offset=offset))
                self.assertEqual(layout["base_lin_vel"].offset, offset)
                critic = torch.zeros(2, offset + 33)
                critic[:, offset:offset + 3] = torch.tensor([2., -4., 6.])
                actual, clipped = layout["base_lin_vel"].decode(critic)
                torch.testing.assert_close(actual, torch.tensor([[1., -2., 3.]] * 2, dtype=torch.float64))
                self.assertFalse(clipped.any())
                self.assertEqual(layout["base_ang_vel"].scale.item(), .25)

    def test_vector_and_none_scales_and_clip_precision(self):
        env = metadata()
        manager = env.unwrapped.observation_manager
        group = manager.cfg.critic
        group.base_lin_vel.scale = (.1, 2., 3.)
        group.base_lin_vel.clip = (-.7, .7)
        group.velocity_commands.scale = None
        manager.cfg = {"critic": group}
        layout = critic_diagnostic_layout(env, NS(vel_offset=39))
        critic = torch.zeros(3, 72)
        critic[:, 39:42] = torch.tensor([[.7, 0., 0.], [0., -.7, 0.], [.1, .2, .3]]) * torch.tensor([.1, 2., 3.])
        actual, clipped = layout["base_lin_vel"].decode(critic)
        self.assertEqual(clipped.tolist(), [True, True, False])
        torch.testing.assert_close(actual[2], torch.tensor([.1, .2, .3], dtype=torch.float64))
        self.assertEqual(layout["velocity_commands"].scale.item(), 1.)

    def test_unsupported_metadata_is_rejected(self):
        mutations = [
            lambda m: setattr(m.cfg.critic, "enable_corruption", True),
            lambda m: setattr(m.cfg.critic, "history_length", 1),
            lambda m: setattr(m.cfg.critic, "concatenate_terms", False),
            lambda m: setattr(m.cfg.critic, "concatenate_dim", 1),
            lambda m: setattr(m.cfg.critic.base_lin_vel, "history_length", 1),
            lambda m: setattr(m.cfg.critic.base_lin_vel, "modifiers", [object()]),
            lambda m: m.active_terms["critic"].__setitem__(6, "missing_velocity"),
            lambda m: m.group_obs_term_dim["critic"].pop(),
            lambda m: m.group_obs_term_dim["critic"].__setitem__(6, (1, 3)),
            lambda m: m.group_obs_term_dim["critic"].__setitem__(6, (4,)),
        ]
        for field, values in [("scale", [0., -1., float("nan"), float("inf"), (1., 2.)]),
                              ("clip", [(1., -1.), (float("nan"), 1.), (0.,)])]:
            mutations += [lambda m, f=field, v=value: setattr(m.cfg.critic.base_lin_vel, f, v) for value in values]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index), self.assertRaises(ValueError):
                env = metadata()
                mutate(env.unwrapped.observation_manager)
                critic_diagnostic_layout(env, NS(vel_offset=39))
        with self.assertRaises(ValueError):
            critic_diagnostic_layout(metadata(), NS(vel_offset=41))


class RunnerVelocityDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def runner(self, chunk_size=5):
        with redirect_stdout(io.StringIO()):
            runner = make_runner()
        runner.cfg["velocity_diagnostics"] = {"chunk_size": chunk_size}
        runner.env.unwrapped = metadata().unwrapped
        return runner

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

    def test_enabled_diagnostics_preserve_updates_optimizers_and_rng(self):
        enabled = self.runner()
        torch.manual_seed(42)
        disabled = self.runner()
        disabled.cfg["velocity_diagnostics"] = None
        for seed in [123, 456]:  # Student DAgger, then teacher PPO.
            torch.manual_seed(seed)
            enabled.learn(1)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(seed)
            disabled.learn(1)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            for component in ["policy", "optimizer", "hist_encoder_optimizer"]:
                self.assert_nested_equal(getattr(enabled.alg, component).state_dict(),
                                         getattr(disabled.alg, component).state_dict())
            self.assertEqual(enabled.alg.counter, disabled.alg.counter)
            self.assertEqual(enabled.current_learning_iteration, disabled.current_learning_iteration)
            self.assertEqual(enabled.last_velocity_diagnostics["sample_count"], 16)
            self.assertEqual(disabled.last_velocity_diagnostics, {})

    def test_pre_action_storage_physical_units_chunking_and_no_mutation(self):
        r = self.runner()
        storage = r.alg.storage
        storage.step = 1  # Ignore unused capacity, even if it contains nonfinite data.
        storage.observations[1:].apply_(lambda x: x.fill_(float("nan")))
        critic = storage.observations[0]["critic"]
        critic.zero_()
        critic[:, 39:42] = torch.tensor([.01, -.005, .03]) * 2
        critic[:, 2] = .06 * .25  # Must be classified as moving after inverse scale.
        critic[0, 39] = 200.  # Saturated true velocity must not be reported as truth.
        critic[1, 6] = -100.  # Saturated command.
        critic[2, 0] = 25.  # Saturated angular velocity.
        storage.observations[0]["policy"].zero_()
        before = storage.observations.clone()
        state = copy.deepcopy(r.alg.policy.state_dict())
        for parameter in r.alg.policy.parameters():
            parameter.grad = torch.ones_like(parameter)
        layout = critic_diagnostic_layout(r.env, r.alg.policy)
        calls = []

        def predict(batch, return_vel):
            self.assertTrue(return_vel)
            calls.append(len(batch))
            return None, torch.tensor([.02, -.025, .06]).expand(len(batch), 3) * 2

        with patch.object(r.alg.policy, "infer_hist_latent", side_effect=predict), patch.object(
            r.env, "get_observations", side_effect=AssertionError("must use stored pre-action observations")
        ):
            small = r._collect_velocity_diagnostics((VelocityDiagnosticConfig(chunk_size=3), layout))
            self.assertEqual(calls, [3, 3, 2])
            calls.clear()
            large = r._collect_velocity_diagnostics((VelocityDiagnosticConfig(chunk_size=32), layout))
            self.assertEqual(calls, [8])
        self.assert_nested_equal(small, large)
        self.assertEqual((small["sample_count"], small["valid_sample_count"], small["clipped_sample_count"]), (8, 5, 3))
        self.assertEqual(small["command_zero/sample_count"], 5)
        self.assertEqual(small["state_near_stationary/sample_count"], 0)
        for axis, bias in [("vx", .01), ("vy", -.02), ("vz", .03)]:
            self.assertAlmostEqual(small[f"all/bias_{axis}_m_s"], bias)
            self.assertAlmostEqual(small[f"all/rmse_{axis}_m_s"], abs(bias))
        for key in before.keys():
            torch.testing.assert_close(before[key], storage.observations[key], rtol=0, atol=0, equal_nan=True)
        self.assert_nested_equal(state, r.alg.policy.state_dict())
        self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in r.alg.policy.parameters()))

    def test_empty_rollout_and_disabled_collection_skip_inference(self):
        r = self.runner()
        layout = critic_diagnostic_layout(r.env, r.alg.policy)
        with patch.object(r.alg.policy, "infer_hist_latent") as inference:
            self.assertEqual(r._collect_velocity_diagnostics(None), {})
            report = r._collect_velocity_diagnostics((VelocityDiagnosticConfig(), layout))
            inference.assert_not_called()
        self.assertEqual(report["sample_count"], 0)
        self.assertFalse(any("rmse" in key for key in report))

    def test_logging_controller_tags_once_per_rollout_before_update(self):
        r = self.runner()
        r.log_dir = "unused-mocked-log-dir"
        r._prepare_logging_writer = lambda: None
        r.logger_type = "tensorboard"
        r.writer = Mock()
        r.log = Mock()
        original_collect = r._collect_velocity_diagnostics
        original_student, original_teacher = r.alg.update_dagger, r.alg.update
        events = []

        def collect(setup):
            self.assertEqual(r.alg.storage.step, 2)
            events.append("diagnostics")
            return original_collect(setup)

        def update(fn, name):
            events.append(name)
            return fn()

        with patch.object(r, "save"), patch("rsl_rl.runners.on_policy_runner_roa.store_code_state", return_value=[]), \
             patch.object(r, "_collect_velocity_diagnostics", side_effect=collect), \
             patch.object(r.alg, "update_dagger", side_effect=lambda: update(original_student, "student")), \
             patch.object(r.alg, "update", side_effect=lambda: update(original_teacher, "teacher")):
            r.learn(2)
        self.assertEqual(events, ["diagnostics", "student", "diagnostics", "teacher"])
        counts = [call.args for call in r.writer.add_scalar.call_args_list if call.args[0] in {
            "VelocityDiagnostics/student/sample_count", "VelocityDiagnostics/teacher/sample_count"}]
        self.assertEqual(counts, [("VelocityDiagnostics/student/sample_count", 16., 0),
                                  ("VelocityDiagnostics/teacher/sample_count", 16., 1)])
        learning_rates = [call.args for call in r.writer.add_scalar.call_args_list
                          if call.args[0] == "VelocityDiagnostics/history_learning_rate"]
        self.assertEqual(len(learning_rates), 2)
        self.assertTrue(all(call[1] == r.alg.hist_encoder_optimizer.param_groups[0]["lr"] for call in learning_rates))
        self.assertEqual(r.alg.storage.step, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

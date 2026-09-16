"""CPU checks for paired ROA diagnostics, physical units and immutable snapshots."""
from pathlib import Path
import sys
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
from rsl_rl.modules.actor_critic_roa import ActorCriticROA
from policy_evaluation_telemetry import (
    BASE_REQUIRED_SIGNALS, ROA_DIAGNOSTIC_SIGNALS, ROA_ABLATION_SIGNALS,
    roa_diagnostic_signals, SignalLedger,
    capture_roa_diagnostics, telemetry_report,
)
from policy_evaluation_evidence import build_scenario_contract, EvaluationEvidenceError, scenario_sha256


class RoaEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.actual = torch.tensor([[.1, -.2, .3], [.4, -.5, .6]])
        self.obs = TensorDict({"policy": torch.randn(2, 10, 39), "critic": torch.randn(2, 72),
                               "privileged": torch.randn(2, 33)}, batch_size=[2])
        self.obs["critic"][:, 39:42] = self.actual * 2
        self.policy = ActorCriticROA(self.obs, {"policy": ["policy"], "critic": ["critic"],
                                    "privileged": ["privileged"]}, 10,
                                    actor_hidden_dims=[32], critic_hidden_dims=[32],
                                    priv_encoder_dims=[64, 20], vel_offset=39).eval()

    def test_same_observations_physical_units_and_branch_actions(self):
        before = {k: v.clone() for k, v in self.obs.items()}
        state = {k: v.clone() for k, v in self.policy.state_dict().items()}
        snapshot = capture_roa_diagnostics(self.policy, self.obs, self.actual, 2.)
        torch.testing.assert_close(snapshot["roa_teacher_velocity_b"], self.actual)
        torch.testing.assert_close(snapshot["roa_true_velocity_b"], self.actual)
        with torch.inference_mode():
            _, scaled_prediction = self.policy.infer_hist_latent(self.obs, return_vel=True)
            torch.testing.assert_close(snapshot["roa_student_velocity_b"], scaled_prediction / 2.)
            for mode in ["teacher", "student"]:
                expected = self.policy.act_inference(self.obs, hist_encoding=mode == "student")
                torch.testing.assert_close(snapshot[f"roa_{mode}_action"], expected, rtol=0, atol=0)
        for k, v in before.items():
            torch.testing.assert_close(self.obs[k], v, rtol=0, atol=0)
        for k, v in state.items():
            torch.testing.assert_close(self.policy.state_dict()[k], v, rtol=0, atol=0)
        self.actual.add_(100)  # emulate simulator buffers being updated by env.step
        torch.testing.assert_close(snapshot["roa_true_velocity_b"], before["critic"][:, 39:42] / 2.)

    def test_invalid_scale_and_nonfinite_data_rejected(self):
        for scale in [0., -1., float("nan"), [1., 2.]]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                capture_roa_diagnostics(self.policy, self.obs, self.actual, scale)
        self.actual[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            capture_roa_diagnostics(self.policy, self.obs, self.actual, 2.)

    def test_ablation_replaces_only_selected_actor_input_block(self):
        inputs = []
        def capture(module, args):
            inputs.append(args[0].detach().clone())
        hook = self.policy.actor.register_forward_pre_hook(capture)
        try:
            data = capture_roa_diagnostics(self.policy, self.obs, self.actual, 2., include_ablations=True)
        finally:
            hook.remove()
        self.assertEqual(len(inputs), 4)
        teacher, student, true_velocity, teacher_latent = inputs
        for vector in inputs[1:]:
            torch.testing.assert_close(vector[:, :39], teacher[:, :39], rtol=0, atol=0)
        torch.testing.assert_close(true_velocity[:, 39:42], teacher[:, 39:42], rtol=0, atol=0)
        torch.testing.assert_close(true_velocity[:, 39:42], self.actual * 2, rtol=0, atol=0)
        torch.testing.assert_close(true_velocity[:, 42:], student[:, 42:], rtol=0, atol=0)
        torch.testing.assert_close(teacher_latent[:, 39:42], student[:, 39:42], rtol=0, atol=0)
        torch.testing.assert_close(teacher_latent[:, 42:], teacher[:, 42:], rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(data["roa_student_true_velocity_action"], self.policy.actor(true_velocity), rtol=0, atol=0)
            torch.testing.assert_close(data["roa_student_teacher_latent_action"], self.policy.actor(teacher_latent), rtol=0, atol=0)

    def test_old_diagnostics_remain_valid_and_ablation_actions_are_required(self):
        self.assertEqual(roa_diagnostic_signals("student"), ROA_DIAGNOSTIC_SIGNALS)
        required = roa_diagnostic_signals("student_true_velocity")
        self.assertTrue(set(ROA_ABLATION_SIGNALS).issubset(required))
        ledger = SignalLedger({name: 1 for name in required}, required_signals=required)
        for name in ROA_DIAGNOSTIC_SIGNALS:
            ledger.capture(name, lambda: [0.])
        self.assertEqual(ledger.report()["status"], "partial")
        self.assertEqual(set(ledger.report()["missing_required_signals"]), set(ROA_ABLATION_SIGNALS))

    def test_missing_diagnostics_cannot_report_complete(self):
        required = BASE_REQUIRED_SIGNALS | frozenset(ROA_DIAGNOSTIC_SIGNALS)
        ledger = SignalLedger({name: 1 for name in required}, required_signals=required)
        for name in BASE_REQUIRED_SIGNALS:
            ledger.capture(name, lambda: [0.])
        report = telemetry_report(requested=True, runner="OnPolicyRunnerROA", ledger=ledger,
                                  additional_required_signals=ROA_DIAGNOSTIC_SIGNALS)
        self.assertEqual(report["telemetry_status"], "partial")
        self.assertTrue(report["telemetry_required_for_complete_assessment"])
        self.assertEqual(set(report["missing_required_signals"]), set(ROA_DIAGNOSTIC_SIGNALS))

    def test_controller_is_hash_bound_and_unknown_modes_rejected(self):
        import json
        def scenario(overrides):
            return build_scenario_contract(scenario_id="paired", scenario_overrides_json=json.dumps(overrides),
                                           command_schedule_json="[]", duration_steps=100, num_envs=1, seed=42)
        teacher = scenario({"evaluation.roa_mode": "teacher"})
        student = scenario({"evaluation.roa_mode": "student"})
        self.assertNotEqual(scenario_sha256(teacher), scenario_sha256(student))
        modes = ["teacher", "student", "student_true_velocity", "student_teacher_latent"]
        self.assertEqual(len({scenario_sha256(scenario({"evaluation.roa_mode": mode})) for mode in modes}), 4)
        scenario({})  # historical scenarios retain their existing contract
        for overrides in [{"evaluation.roa_mode": "bad"}, {"evaluation.roa_mode": True},
                          {"evaluation.typo": "student"}]:
            with self.assertRaises(EvaluationEvidenceError):
                scenario(overrides)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""CPU checks for deterministic DWAQ velocity intervention and evidence binding."""
from pathlib import Path
import contextlib
import io
import json
import sys
import unittest

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rsl_rl"))
sys.path.insert(0, str(ROOT / "scripts/reinforcement_learning/rsl_rl"))
from rsl_rl.modules.actor_critic_dwaq import ActorCriticDwaq
from policy_evaluation_telemetry import (
    BASE_REQUIRED_SIGNALS, DWAQ_DIAGNOSTIC_SIGNALS,
    SignalLedger, capture_dwaq_diagnostics, telemetry_report,
)
from policy_evaluation_evidence import build_scenario_contract, scenario_sha256, EvaluationEvidenceError


class DwaqEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.actual = torch.tensor([[.1, -.2, .3], [.4, -.5, .6]])
        self.obs = TensorDict({"policy": torch.randn(2, 5, 39), "critic": torch.randn(2, 42)}, [2])
        self.obs["critic"][:, 39:42] = self.actual * 2
        with contextlib.redirect_stdout(io.StringIO()):
            self.policy = ActorCriticDwaq(
                self.obs, {"policy": ["policy"], "critic": ["critic"]}, 10,
                actor_hidden_dims=[32], critic_hidden_dims=[32], vae_hidden_dims=[32, 16],
                actor_obs_normalization=True,
            )
        # Nontrivial normalization makes an incorrectly post-normalization swap detectable.
        self.policy.actor_obs_normalizer(torch.randn(50, 58) * 3 + 2)
        self.policy.eval()

    def test_original_action_exact_and_only_velocity_input_replaced(self):
        inputs = []
        hook = self.policy.actor_obs_normalizer.register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().clone()))
        state = {k: v.clone() for k, v in self.policy.state_dict().items()}
        observations = {k: v.clone() for k, v in self.obs.items()}
        rng = torch.get_rng_state().clone()
        try:
            data = capture_dwaq_diagnostics(self.policy, self.obs, self.actual, 2.)
        finally:
            hook.remove()
        self.assertEqual(len(inputs), 2)
        torch.testing.assert_close(inputs[0][:, 3:], inputs[1][:, 3:], rtol=0, atol=0)
        torch.testing.assert_close(inputs[1][:, :3], self.actual * 2, rtol=0, atol=0)
        torch.testing.assert_close(data["dwaq_estimated_velocity_b"], inputs[0][:, :3] / 2)
        with torch.inference_mode():
            torch.testing.assert_close(data["dwaq_estimated_velocity_action"],
                                       self.policy.act_inference(self.obs), rtol=0, atol=0)
            expected = self.policy.actor(self.policy.actor_obs_normalizer(inputs[1]))
            torch.testing.assert_close(data["dwaq_true_velocity_action"], expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for k, v in state.items():
            torch.testing.assert_close(v, self.policy.state_dict()[k], rtol=0, atol=0)
        for k, v in observations.items():
            torch.testing.assert_close(v, self.obs[k], rtol=0, atol=0)
        self.actual.add_(100)
        torch.testing.assert_close(data["dwaq_true_velocity_b"], observations["critic"][:, 39:42] / 2)

    def test_axis_scales_and_bad_labels(self):
        scale = torch.tensor([2., 3., 4.])
        self.obs["critic"][:, 39:42] = self.actual * scale
        data = capture_dwaq_diagnostics(self.policy, self.obs, self.actual, scale)
        torch.testing.assert_close(data["dwaq_label_velocity_b"], self.actual)
        self.obs["critic"][:, 39:42] += 1
        with self.assertRaisesRegex(ValueError, "label differs"):
            capture_dwaq_diagnostics(self.policy, self.obs, self.actual, scale)

    def test_invalid_scale_and_nonfinite_outputs_rejected(self):
        for scale in [0., -1., float("nan"), [1., 2.]]:
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                capture_dwaq_diagnostics(self.policy, self.obs, self.actual, scale)
        with torch.no_grad():
            self.policy.encode_mean_vel.bias[0] = float("nan")
        with self.assertRaises(ValueError):
            capture_dwaq_diagnostics(self.policy, self.obs, self.actual, 2.)

    def test_controller_hash_bound_and_mixed_modes_rejected(self):
        def scenario(overrides):
            return build_scenario_contract(scenario_id="paired", scenario_overrides_json=json.dumps(overrides),
                command_schedule_json="[]", duration_steps=100, num_envs=1, seed=42)
        self.assertNotEqual(scenario_sha256(scenario({"evaluation.dwaq_mode": "estimated_velocity"})),
                            scenario_sha256(scenario({"evaluation.dwaq_mode": "true_velocity"})))
        for overrides in [{"evaluation.dwaq_mode": "bad"}, {"evaluation.dwaq_mode": True},
                          {"evaluation.roa_mode": "student", "evaluation.dwaq_mode": "true_velocity"}]:
            with self.assertRaises(EvaluationEvidenceError):
                scenario(overrides)

    def test_missing_diagnostics_cannot_report_complete(self):
        required = BASE_REQUIRED_SIGNALS | frozenset(DWAQ_DIAGNOSTIC_SIGNALS)
        ledger = SignalLedger({name: 1 for name in required}, required_signals=required)
        for name in BASE_REQUIRED_SIGNALS:
            ledger.capture(name, lambda: [0.])
        result = telemetry_report(requested=True, runner="OnPolicyRunnerDwaq", ledger=ledger,
                                  additional_required_signals=DWAQ_DIAGNOSTIC_SIGNALS)
        self.assertEqual(result["telemetry_status"], "partial")
        self.assertEqual(set(result["missing_required_signals"]), set(DWAQ_DIAGNOSTIC_SIGNALS))


if __name__ == "__main__":
    unittest.main(verbosity=2)

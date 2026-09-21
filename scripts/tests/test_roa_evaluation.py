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
    RoaSensorDelay, ROA_SENSOR_DELAY_OPTION, ROA_SENSOR_COLUMNS,
    ROA_CURRENT_COLUMNS,
)
from policy_evaluation_evidence import (
    build_scenario_contract, EvaluationEvidenceError, scenario_sha256, validate_roa_sensor_delay,
)


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

    def test_zero_sensor_delay_preserves_history_actions_inputs_and_rng(self):
        adapter = RoaSensorDelay(0, .02)
        before = self.obs.clone()
        rng = torch.get_rng_state().clone()
        state = {k: v.clone() for k, v in self.policy.state_dict().items()}
        with torch.inference_mode():
            for step in range(12):
                if step == 5:
                    adapter.reset(torch.tensor([True, False]))
                delayed, snapshot = adapter.advance(self.obs, step)
                self.assertIs(delayed, self.obs)
                torch.testing.assert_close(self.policy.act_inference(delayed, hist_encoding=True),
                                           self.policy.act_inference(self.obs, hist_encoding=True), rtol=0, atol=0)
                torch.testing.assert_close(snapshot['roa_sensor_age_s'], torch.zeros(2, 1, dtype=torch.float64))
        for key in before.keys():
            torch.testing.assert_close(before[key], self.obs[key], rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        for key, value in state.items():
            torch.testing.assert_close(value, self.policy.state_dict()[key], rtol=0, atol=0)

    def test_one_step_delay_full_history_and_per_environment_reset(self):
        adapter = RoaSensorDelay(1, .02)
        expected = [[], []]
        previous = None
        for step in range(14):
            frame = torch.arange(78, dtype=torch.float32).reshape(2, 39) + step * 1000
            self.obs['policy'] = frame[:, None].repeat(1, 10, 1)
            untouched = self.obs.clone()
            reset = [step in (0, 6), step == 0]
            if step == 6:
                adapter.reset(torch.tensor([True, False]))
            result, snapshot = adapter.advance(self.obs, step)
            for env in range(2):
                target = frame[env].clone()
                source = frame[env] if reset[env] else previous[env]
                target[list(ROA_SENSOR_COLUMNS)] = source[list(ROA_SENSOR_COLUMNS)]
                expected[env] = [target.clone()] * 10 if reset[env] else (expected[env] + [target])[-10:]
                torch.testing.assert_close(result['policy'][env], torch.stack(expected[env]), rtol=0, atol=0)
                torch.testing.assert_close(snapshot['roa_delayed_policy_frame'][env, list(ROA_CURRENT_COLUMNS)],
                                           frame[env, list(ROA_CURRENT_COLUMNS)], rtol=0, atol=0)
                self.assertEqual(snapshot['roa_sensor_source_step'][env].item(), step if reset[env] else step - 1)
            for key in untouched.keys():
                torch.testing.assert_close(self.obs[key], untouched[key], rtol=0, atol=0)
            for key in ('critic', 'privileged'):
                torch.testing.assert_close(result[key], self.obs[key], rtol=0, atol=0)
            previous = frame.clone()
            self.obs['policy'].fill_(-999)  # caller mutation cannot corrupt retained frames

    def test_sensor_delay_rejects_duplicate_advance_and_wrong_contract(self):
        for steps in (True, -1, 2, 1.0):
            with self.assertRaises(ValueError):
                RoaSensorDelay(steps, .02)
        with self.assertRaises(ValueError):
            RoaSensorDelay(1, .01)
        adapter = RoaSensorDelay(1, .02)
        adapter.advance(self.obs, 0)
        for step in (0, 2):
            with self.assertRaises(ValueError):
                adapter.advance(self.obs, step)
        with self.assertRaises(ValueError):
            adapter.reset(torch.tensor([True]))
        malformed = self.obs.clone()
        malformed['policy'] = torch.zeros(2, 5, 39)
        with self.assertRaises(ValueError):
            adapter.advance(malformed, 1)

    def test_sensor_delay_contract_and_lossless_evidence_validation(self):
        import copy
        import json
        def contract(overrides):
            return build_scenario_contract(scenario_id='age', scenario_overrides_json=json.dumps(overrides),
                                           command_schedule_json='[]', duration_steps=5, num_envs=1, seed=42)
        cases = [contract({'evaluation.roa_mode': 'student', ROA_SENSOR_DELAY_OPTION: n}) for n in (0, 1)]
        self.assertNotEqual(scenario_sha256(cases[0]), scenario_sha256(cases[1]))
        for overrides in ({ROA_SENSOR_DELAY_OPTION: 1},
                          {'evaluation.roa_mode': 'teacher', ROA_SENSOR_DELAY_OPTION: 1},
                          *({'evaluation.roa_mode': 'student', ROA_SENSOR_DELAY_OPTION: n} for n in (True, 2, 1.0, -1))):
            with self.assertRaises(EvaluationEvidenceError):
                contract(overrides)
        adapter = RoaSensorDelay(1, .02)
        samples = []
        for step in range(5):
            self.obs['policy'].fill_(float(step))
            _, snapshot = adapter.advance(self.obs, step)
            samples.append({'step': step, 'done': step == 2,
                            **{k: v[0].tolist() for k, v in snapshot.items()}})
            if step == 2:
                adapter.reset(torch.tensor([True, False]))
        telemetry = {'step_dt_seconds': .02, 'stride': 1,
                     'inputs': {'resource_mode': {'telemetry_stride': 1}}, 'samples': samples}
        validate_roa_sensor_delay(telemetry, 1)
        for field, value in [('roa_sensor_source_step', [1]), ('roa_sensor_age_s', [.04]),
                             ('roa_delayed_policy_frame', [999.] * 39), ('step', 9), ('done', None)]:
            broken = copy.deepcopy(telemetry)
            broken['samples'][1][field] = value
            with self.assertRaises(EvaluationEvidenceError):
                validate_roa_sensor_delay(broken, 1)
        broken = copy.deepcopy(telemetry)
        broken['samples'][3]['roa_sensor_source_step'] = [2]
        with self.assertRaises(EvaluationEvidenceError):
            validate_roa_sensor_delay(broken, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

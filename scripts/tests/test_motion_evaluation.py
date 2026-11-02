"""Motion evaluation contracts without launching Isaac Sim."""
import copy
import ast
import contextlib
import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reinforcement_learning/rsl_rl"))
from policy_evaluation_evidence import EvaluationEvidenceError, build_scenario_contract, validate_motion_telemetry, validate_scenario_contract
from policy_evaluation_telemetry import (
    MOTION_REQUIRED_SIGNALS, MOTION_VECTOR_WIDTHS, MOTION_JOINT_SIGNALS,
    MOTION_BODY_SIGNALS, SignalLedger, telemetry_report, summarize_motion_samples,
    MOTION_PHYSICS_VECTOR_WIDTHS, MOTION_PHYSICS_JOINT_SIGNALS, summarize_motion_physics,
    LandingPullPulse, summarize_landing_pull,
)


class MotionEvaluationTests(unittest.TestCase):
    def pull_config(self):
        path = Path(self.directory.name) / "motion.npz"
        return {"delta_v_m_s": .5, "angle_deg": 45., "duration_s": .05,
                "body_name": "base_link", "point_b_m": [.2015, 0., .1263],
                "airborne_time_s": .05, "contact_threshold_n": 10.,
                "geometry": {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}

    def test_pull_requires_airborne_then_both_wheels_and_next_step(self):
        pulse = LandingPullPulse(self.pull_config(), .005)
        pulse.observe(0, 0, [0, 0, 20, 20])
        self.assertIsNone(pulse.start_step)
        for step in range(1, 11):
            pulse.observe(step, 0, [0, 0, 0, 0])
        pulse.observe(11, 0, [0, 0, 20, 0])
        self.assertIsNone(pulse.start_step)
        pulse.observe(12, 0, [0, 0, 20, 20])
        self.assertFalse(pulse.active(12, 0))
        self.assertEqual([i for i in range(40) if pulse.active(i, 0)], list(range(13, 23)))
        self.assertFalse(pulse.active(13, 1))
        for step in range(40, 51):
            pulse.observe(step, 1, [0, 0, 0, 0])
        pulse.observe(51, 1, [0, 0, 20, 20])
        self.assertEqual(pulse.start_step, 13)

    def test_pull_rejects_fractional_physics_duration(self):
        config = self.pull_config()
        config["duration_s"] = .052
        with self.assertRaisesRegex(ValueError, "integer"):
            LandingPullPulse(config, .005)

    def test_pull_contract_requires_full_physics_and_geometry_hash(self):
        scenario = copy.deepcopy(self.scenario)
        scenario["scenario_overrides"]["evaluation.landing_pull"] = self.pull_config()
        with self.assertRaisesRegex(EvaluationEvidenceError, "full physics"):
            validate_scenario_contract(scenario)
        scenario["scenario_overrides"]["evaluation.motion_physics_window"] = [0, 2]
        validate_scenario_contract(scenario)
        scenario["scenario_overrides"]["evaluation.landing_pull"]["geometry"]["sha256"] = "0" * 64
        with self.assertRaises(EvaluationEvidenceError):
            validate_scenario_contract(scenario)

    def test_pull_telemetry_rejects_force_after_withdrawal(self):
        records = self.add_physics()
        config = self.pull_config()
        config.update(duration_s=.01, airborne_time_s=.005)
        self.scenario["scenario_overrides"]["evaluation.landing_pull"] = config
        self.telemetry["motion_diagnostics"]["initial_runtime"] = {"masses_kg": [1.]}
        pulse = LandingPullPulse(config, .005)
        force = np.array([1., 0., 1.]) / np.sqrt(2) * 50.
        for r in records:
            if 1 <= r["physics_step"] < 12:
                r["contact_forces_w"] = [0.] * 8 + [20., 0., 0., 20.]
            active = pulse.active(r["physics_step"], r["episode_id"])
            applied = force if active else np.zeros(3)
            r.update(pull_active=active, pull_forward_w=[1., 0., 0.],
                     pull_force_w=applied.tolist(), pull_torque_w=np.cross(config["point_b_m"], applied).tolist(),
                     pull_point_w=config["point_b_m"], pull_body_com_w=[0., 0., 0.],
                     root_position_w=[0., 0., 0.], root_linear_velocity_w=[0., 0., 0.],
                     root_quaternion_w=[1., 0., 0., 0.],
                     joint_position_target=[0., 0.], joint_velocity_target=[0., 0.])
            pulse.observe(r["physics_step"], r["episode_id"], np.asarray(r["contact_forces_w"]).reshape(4, 3)[:, 2])
            if r["substep"] == 3:
                self.samples[r["control_step"]]["contact_forces_w"] = r["contact_forces_w"]
        diagnostic = {"config": config, "total_mass_kg": 1., "forward_w": [1., 0., 0.],
                      "summary": summarize_landing_pull(records, config, .005)}
        self.result["landing_pull"] = self.telemetry["landing_pull"] = diagnostic
        self.result["metrics"] = summarize_motion_samples(self.samples, self.telemetry["joint_names"], [10, 10])
        self.result["physics_window_metrics"] = summarize_motion_physics(records, [10, 10], .005)
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)
        self.assertEqual(diagnostic["summary"]["applied_substeps"], 2)
        self.assertEqual(diagnostic["summary"]["horizons"], {})
        records[4]["pull_force_w"][0] = 1.
        with self.assertRaisesRegex(EvaluationEvidenceError, "force/torque/schedule"):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "motion.npz"
        np.savez(path, fps=np.array([50]), joint_pos=np.zeros((3, 2)))
        self.scenario = {"scenario_id": "motion-test", "scenario_overrides": {
            "evaluation.motion_mode": "nominal_start",
            "evaluation.motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }, "command_schedule": [], "duration_steps": 4, "num_envs": 1, "seed": 42}
        widths = {**MOTION_VECTOR_WIDTHS, **{k: 2 for k in MOTION_JOINT_SIGNALS},
                  **MOTION_BODY_SIGNALS, "command": 4, "action": 2, "contact_forces_w": 12}
        self.samples = []
        for step, frame in enumerate((0, 1, 2, 0)):
            sample = {k: [0.0] * width for k, width in widths.items()}
            sample.update(step=step, sim_time_seconds=(step + 1) * .02, frame_index=frame,
                          episode_id=int(step == 3), episode_start_frame=0,
                          reward=1.0, done=step == 2, timeout=step == 2,
                          termination_terms={"motion_finished": step == 2, "time_out": False, "anchor_pos": False})
            sample["joint_position_error"] = [0.0, 1000.0]  # wheel angle must be excluded
            sample["action"] = [float(step if step < 3 else 1000)] * 2
            self.samples.append(sample)
        ledger = SignalLedger({k: 4 for k in MOTION_REQUIRED_SIGNALS}, required_signals=MOTION_REQUIRED_SIGNALS)
        for sample in self.samples:
            for k in MOTION_REQUIRED_SIGNALS:
                ledger.capture(k, lambda k=k: sample[k])
        evidence = telemetry_report(requested=True, runner="OnPolicyRunner", ledger=ledger,
                                    additional_required_signals=MOTION_REQUIRED_SIGNALS)
        diagnostic = {"mode": "nominal_start", "sampling_phase": "post_physics_pre_reset",
                      "motion_file": {"path": str(path), "sha256": self.scenario['scenario_overrides']['evaluation.motion_sha256']},
                      "frame_count": 3, "fps": 50.0, "body_names": ["base_link"],
                      "contact_body_names": ["right_foot_link", "left_foot_link", "right_wheel_link", "left_wheel_link"]}
        self.telemetry = {**evidence, "motion_diagnostics": diagnostic, "samples": self.samples,
                          "joint_names": ["leg_joint", "wheel_joint"], "joint_effort_limits": [10, 10],
                          "step_dt_seconds": .02}
        self.result = {"artifact": "native", "runner": "OnPolicyRunner", "telemetry_status": "complete",
                       "motion_diagnostics": diagnostic, "inputs": {"resource_mode": {
                           "telemetry_stride": 1, "video_requested": False, "idle_gpu_required": True,
                           "training_overlap": False}},
                       "metrics": summarize_motion_samples(self.samples, self.telemetry['joint_names'], [10, 10])}

    def test_success_censor_and_reset_exclusion(self):
        metrics = self.result['metrics']
        self.assertEqual(metrics['successful_motion_episodes'], 1)
        self.assertEqual(metrics['censored_episodes'], 1)
        self.assertEqual(metrics['position_joint_rmse_rad'], 0)
        self.assertEqual(metrics['action_rate_rms'], 1)
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def add_action_sampling(self):
        self.scenario['scenario_overrides']['evaluation.motion_action_mode'] = 'sample'
        checkpoint = Path(self.directory.name) / 'checkpoint.pt'
        torch.save({'model_state_dict': {'std': torch.tensor([.14, .2])}}, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.result['inputs']['checkpoint'] = {'path': str(checkpoint), 'sha256': digest}
        value = {'mode': 'sample', 'distribution': 'Normal', 'std_source': 'checkpoint',
                 'checkpoint_sha256': digest, 'noise_std_type': 'scalar', 'action_std': [.14, .2]}
        self.result['motion_diagnostics']['action_sampling'] = value
        return value

    def test_action_sampling_contract_and_checkpoint_binding(self):
        value = self.add_action_sampling()
        validate_scenario_contract(self.scenario)
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)
        value['checkpoint_sha256'] = '2' * 64
        with self.assertRaisesRegex(EvaluationEvidenceError, 'checkpoint-bound'):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_action_sampling_rejects_finite_std_tamper(self):
        value = self.add_action_sampling()
        value['action_std'] = [.15, .2]
        with self.assertRaisesRegex(EvaluationEvidenceError, 'differs from checkpoint'):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_invalid_action_sampling_mode_and_missing_motion_rejected(self):
        self.scenario['scenario_overrides']['evaluation.motion_action_mode'] = 'training'
        with self.assertRaises(EvaluationEvidenceError):
            validate_scenario_contract(self.scenario)
        self.scenario['scenario_overrides'] = {'evaluation.motion_action_mode': 'sample'}
        with self.assertRaisesRegex(EvaluationEvidenceError, 'requires motion'):
            validate_scenario_contract(self.scenario)

    def test_action_sampling_standard_deviations_are_positive_finite_and_complete(self):
        value = self.add_action_sampling()
        for bad in ([0, .2], [-.1, .2], [float('nan'), .2], [.1], [True, .2]):
            value['action_std'] = bad
            with self.subTest(std=bad), self.assertRaisesRegex(EvaluationEvidenceError, 'checkpoint-bound'):
                validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_explicit_action_mode_requires_matching_telemetry_but_legacy_mean_still_valid(self):
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)
        self.scenario['scenario_overrides']['evaluation.motion_action_mode'] = 'sample'
        with self.assertRaisesRegex(EvaluationEvidenceError, 'mode differs'):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)
        self.result['motion_diagnostics']['action_sampling'] = {'mode': 'mean'}
        with self.assertRaisesRegex(EvaluationEvidenceError, 'mode differs'):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)
        self.scenario['scenario_overrides']['evaluation.motion_action_mode'] = 'mean'
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_simultaneous_failure_is_not_success(self):
        self.samples[2]['termination_terms']['anchor_pos'] = True
        metrics = summarize_motion_samples(self.samples, self.telemetry['joint_names'], [10, 10])
        self.assertEqual(metrics['successful_motion_episodes'], 0)
        self.assertEqual(metrics['failed_episodes'], 1)

    def test_missing_contact_rejected(self):
        self.samples[1]['contact_forces_w'] = None
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_training_overlap_preserves_physics_validation(self):
        self.result['inputs']['resource_mode'].update(idle_gpu_required=False, training_overlap=True)
        self.add_physics()
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)
        self.telemetry['physics_samples'].pop()
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_invalid_resource_modes_rejected(self):
        for idle, overlap in ((False, False), (True, True), (1, False), (False, None)):
            with self.subTest(idle=idle, overlap=overlap):
                self.result['inputs']['resource_mode'].update(idle_gpu_required=idle, training_overlap=overlap)
                with self.assertRaisesRegex(EvaluationEvidenceError, 'resource contract'):
                    validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_training_overlap_duration_limit_rejected(self):
        self.result['inputs']['resource_mode'].update(idle_gpu_required=False, training_overlap=True)
        self.scenario['duration_steps'] = 2001
        with self.assertRaisesRegex(EvaluationEvidenceError, 'resource contract'):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_frame_skip_rejected(self):
        self.samples[1]['frame_index'] = 2
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_metric_tamper_rejected(self):
        self.result['metrics']['frame_zero_success_fraction'] = .5
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_bound_motion_changed_rejected(self):
        path = Path(self.result['motion_diagnostics']['motion_file']['path'])
        np.savez(path, fps=np.array([60]), joint_pos=np.zeros((3, 2)))
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_scenario_requires_motion_hash(self):
        with self.assertRaises(EvaluationEvidenceError):
            build_scenario_contract(scenario_id="test", scenario_overrides_json='{"evaluation.motion_mode":"nominal_start"}',
                                    command_schedule_json='[]', duration_steps=4, num_envs=1, seed=42)

    def add_physics(self):
        self.scenario['scenario_overrides']['evaluation.motion_physics_window'] = [0, 2]
        self.telemetry['motion_diagnostics']['physics_window'] = {
            'frame_range_inclusive': [0, 2], 'physics_dt_seconds': .005, 'decimation': 4,
            'sampling_phase': 'after_scene_update_before_reset',
            'wheel_body_names': ['right_wheel_link', 'left_wheel_link'],
        }
        physics = []
        for parent in self.samples:
            for sub in range(4):
                step = parent['step'] * 4 + sub
                s = {'control_step': parent['step'], 'substep': sub, 'physics_step': step,
                     'sim_time_seconds': (step + 1) * .005,
                     **{k: parent[k] for k in ('frame_index', 'episode_id', 'episode_start_frame')}}
                s.update({k: [0.0] * w for k, w in MOTION_PHYSICS_VECTOR_WIDTHS.items()})
                s.update({k: [0.0] * 2 for k in MOTION_PHYSICS_JOINT_SIGNALS})
                physics.append(s)
        self.telemetry['physics_samples'] = physics
        self.result['physics_window_metrics'] = summarize_motion_physics(physics, [10, 10], .005)
        return physics

    def test_physics_includes_terminal_substeps_and_reset(self):
        self.add_physics()
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_physics_missing_substep_rejected(self):
        self.add_physics().pop(5)
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_physics_stale_endpoint_rejected(self):
        self.add_physics()[3]['applied_torque'][0] = 1.0
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_physics_stale_velocity_rejected(self):
        self.add_physics()[1]['wheel_center_velocity_before_w'][2] = -2.0
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_physics_substep_peak_not_lost(self):
        physics = self.add_physics()
        physics[0]['contact_forces_w'][8] = 3000.0
        physics[0]['computed_torque'][0] = 12.0
        physics[0]['applied_torque'][0] = 10.0
        metrics = summarize_motion_physics(physics, [10, 10], .005)
        self.assertEqual(metrics['max_single_wheel_normal_force_n'], 3000.0)
        self.assertEqual(metrics['computed_over_limit_joint_samples'], 1)
        self.result['physics_window_metrics'] = metrics
        validate_motion_telemetry(self.result, self.telemetry, self.scenario)

    def test_physics_metric_tamper_rejected(self):
        self.add_physics()
        self.result['physics_window_metrics']['sample_count'] = 15
        with self.assertRaises(EvaluationEvidenceError):
            validate_motion_telemetry(self.result, self.telemetry, self.scenario)


class FrozenMotionActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Extract the pure selector without importing the simulator CLI entrypoint.
        path = Path(__file__).resolve().parents[1] / 'reinforcement_learning/rsl_rl/evaluate_policy.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_select_motion_action_policy')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        cls.select = staticmethod(namespace['_select_motion_action_policy'])
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rsl_rl'))
        from rsl_rl.modules import ActorCritic
        cls.policy_class = ActorCritic

    def policy(self, noise_type='scalar'):
        obs = {'policy': torch.ones(2, 3)}
        with contextlib.redirect_stdout(io.StringIO()):
            policy = self.policy_class(obs, {'policy': ['policy'], 'critic': ['policy']}, 2,
                                       actor_hidden_dims=[4], critic_hidden_dims=[4],
                                       init_noise_std=.15, noise_std_type=noise_type,
                                       actor_obs_normalization=True)
        policy.eval()
        return policy, obs

    def test_mean_keeps_original_callable_outputs_and_rng(self):
        policy, obs = self.policy()
        original = policy.act_inference
        rng = torch.random.get_rng_state().clone()
        selected, metadata = self.select(original, policy, 'mean', '1' * 64)
        self.assertIs(selected, original)
        self.assertEqual(metadata, {'mode': 'mean'})
        self.assertTrue(torch.equal(selected(obs), original(obs)))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_sampling_matches_native_ppo_and_leaves_model_state_unchanged(self):
        for noise_type in ('scalar', 'log'):
            with self.subTest(noise_type=noise_type):
                policy, obs = self.policy(noise_type)
                state = {k: v.clone() for k, v in policy.state_dict().items()}
                selected, metadata = self.select(policy.act_inference, policy, 'sample', '1' * 64)
                torch.manual_seed(71)
                expected = policy.act(obs)
                torch.manual_seed(71)
                actual = selected(obs)
                self.assertTrue(torch.equal(expected, actual))
                self.assertFalse(torch.equal(actual, policy.act_inference(obs)))
                np.testing.assert_allclose(metadata['action_std'], [.15, .15], rtol=1e-6)
                self.assertEqual(metadata['noise_std_type'], noise_type)
                for key, value in policy.state_dict().items():
                    self.assertTrue(torch.equal(value, state[key]), key)

    def test_invalid_checkpoint_std_rejected(self):
        policy, _ = self.policy()
        with torch.no_grad():
            policy.std[0] = 0
        with self.assertRaisesRegex(ValueError, 'standard deviations'):
            self.select(policy.act_inference, policy, 'sample', '1' * 64)


if __name__ == '__main__':
    unittest.main()

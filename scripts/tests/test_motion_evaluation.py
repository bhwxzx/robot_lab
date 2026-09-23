"""Motion evaluation contracts without launching Isaac Sim."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reinforcement_learning/rsl_rl"))
from policy_evaluation_evidence import EvaluationEvidenceError, build_scenario_contract, validate_motion_telemetry
from policy_evaluation_telemetry import (
    MOTION_REQUIRED_SIGNALS, MOTION_VECTOR_WIDTHS, MOTION_JOINT_SIGNALS,
    MOTION_BODY_SIGNALS, SignalLedger, telemetry_report, summarize_motion_samples,
    MOTION_PHYSICS_VECTOR_WIDTHS, MOTION_PHYSICS_JOINT_SIGNALS, summarize_motion_physics,
)


class MotionEvaluationTests(unittest.TestCase):
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

    def test_simultaneous_failure_is_not_success(self):
        self.samples[2]['termination_terms']['anchor_pos'] = True
        metrics = summarize_motion_samples(self.samples, self.telemetry['joint_names'], [10, 10])
        self.assertEqual(metrics['successful_motion_episodes'], 0)
        self.assertEqual(metrics['failed_episodes'], 1)

    def test_missing_contact_rejected(self):
        self.samples[1]['contact_forces_w'] = None
        with self.assertRaises(EvaluationEvidenceError):
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


if __name__ == '__main__':
    unittest.main()

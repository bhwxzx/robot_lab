"""CPU diagnostics contracts; filesystem writes are mocked, no IsaacSim launch.

Run: conda run -n isaacsim-5.1 python -B scripts/tests/test_motion_sampling_diagnostics.py
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/mdp"
spec = importlib.util.spec_from_file_location("sampling_diagnostics", MDP / "sampling_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MotionSamplingDiagnostics = module.MotionSamplingDiagnostics


class MemoryFile(io.StringIO):
    def __exit__(self, *args):
        return False


class MemoryFilesystem:
    """Exercise x/a mode and directory collision behavior without temporary files."""

    def __init__(self):
        self.directories = set()
        self.files = {}

    def mkdir(self, path, *, parents, exist_ok):
        if path in self.directories and not exist_ok:
            raise FileExistsError(path)
        self.directories.add(path)

    def open(self, path, mode, *, encoding):
        if mode == "x":
            if path in self.files:
                raise FileExistsError(path)
            self.files[path] = MemoryFile()
        elif mode != "a":
            raise AssertionError(f"Unexpected file mode: {mode}")
        stream = self.files[path]
        stream.seek(0, io.SEEK_END)
        return stream

    def rows(self, recorder):
        return [json.loads(line) for line in self.files[recorder.path].getvalue().splitlines()]


def load_command():
    """Use actual command hooks/sampler; stub only base reset/compute and physics."""
    class Base:
        def reset(self, env_ids=None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs)
            # Same time_left RNG call as IsaacLab's CommandTerm._resample().
            self.time_left[env_ids] = self.time_left[env_ids].uniform_(10.0, 20.0)
            self._adaptive_sampling(env_ids)
            return {"base_reset": 1.0}

        def compute(self, dt):
            self._update_command()

    names = {
        "_get_sampling_diagnostics", "_record_sampling_diagnostics", "reset", "compute",
        "flush_sampling_diagnostics", "_adaptive_sampling", "_update_command",
    }
    tree = ast.parse((MDP / "commands.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {
        "CommandTerm": Base, "torch": torch, "math": math,
        "MotionSamplingDiagnostics": MotionSamplingDiagnostics,
        "sample_uniform": lambda low, high, size, device: torch.empty(size, device=device).uniform_(low, high),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, original], type_ignores=[])),
                 str(MDP / "commands.py"), "exec"), namespace)
    return namespace["MotionCommand"]


Command = load_command()
TERMS = {"anchor_pos": False, "ground_contact": False, "motion_finished": True, "time_out": True}


class SamplingDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.fs = MemoryFilesystem()
        fs = self.fs
        self.mkdir_patch = patch.object(Path, "mkdir", lambda path, **kwargs: fs.mkdir(path, **kwargs))
        self.open_patch = patch.object(Path, "open", lambda path, mode, **kwargs: fs.open(path, mode, **kwargs))
        self.mkdir_patch.start()
        self.open_patch.start()
        self.addCleanup(self.mkdir_patch.stop)
        self.addCleanup(self.open_patch.stop)

    def recorder(self, **kwargs):
        return MotionSamplingDiagnostics(**{
            "output_dir": "/diagnostics/new-run", "num_envs": 2, "frame_count": 170,
            "fps": 50.0, "bin_count": 4, "term_timeouts": TERMS, "device": "cpu",
            "window_steps": 3, **kwargs,
        })

    def observe(self, recorder, step, frames=(42, 43), fired=None):
        flags = {name: torch.zeros(recorder.num_envs, dtype=torch.bool) for name in TERMS}
        for name, env_ids in (fired or {}).items():
            flags[name][env_ids] = True
        terminated = flags["anchor_pos"] | flags["ground_contact"]
        recorder.observe(step=step, frames=torch.tensor(frames), terminated=terminated, term_values=flags)

    def test_initial_reset_and_duplicate_step_have_no_extra_exposure(self):
        d = self.recorder()
        self.observe(d, 0)
        self.observe(d, 1, fired={"anchor_pos": [0]})
        self.observe(d, 1, frames=(0, 0), fired={"anchor_pos": [0, 1]})
        d.flush()
        row = self.fs.rows(d)[1]
        self.assertEqual(sum(row["raw_by_frame"]["exposure_steps"]), 2)
        self.assertEqual(row["raw_by_frame"]["failures"][42], 1)
        self.assertEqual(row["control_steps"], 1)
        self.assertTrue(row["partial"])

    def test_boundaries_exposure_rates_and_unobserved_bins(self):
        d = self.recorder(num_envs=8)
        self.observe(d, 1, frames=(0, 42, 43, 84, 85, 127, 128, 169),
                     fired={"anchor_pos": [1, 2, 6]})
        d.flush()
        row = self.fs.rows(d)[1]
        bins = row["by_sampling_failure_bin"]
        self.assertEqual(bins["exposure_steps"], [2, 2, 2, 2])
        self.assertEqual(bins["failures_per_10000_steps"], [5000, 5000, 0, 5000])
        phase = row["by_phase_0p1s"]
        self.assertEqual(phase["exposure_steps"][8], 2)  # 0.84 and 0.86 seconds
        self.assertIsNone(phase["failures_per_10000_steps"][1])
        self.assertEqual(len(phase["exposure_steps"]), 34)

    def test_point_one_second_boundaries(self):
        d = self.recorder(num_envs=4)
        self.observe(d, 1, frames=(4, 5, 59, 60))
        d.flush()
        phase = self.fs.rows(d)[1]["by_phase_0p1s"]["exposure_steps"]
        self.assertEqual([phase[i] for i in (0, 1, 11, 12)], [1, 1, 1, 1])

    def test_simultaneous_causes_and_normal_completion_are_distinct(self):
        d = self.recorder()
        self.observe(d, 1, fired={"anchor_pos": [0], "ground_contact": [0], "motion_finished": [0, 1]})
        d.flush()
        row = self.fs.rows(d)[1]["by_sampling_failure_bin"]
        self.assertEqual(sum(row["failures"]), 1)
        self.assertEqual(sum(row["timeout_only"]), 1)
        self.assertEqual(sum(row["motion_finished_without_failure"]), 1)
        self.assertEqual(sum(row["termination_counts"]["motion_finished"]), 2)
        self.assertEqual(sum(row["termination_counts"]["anchor_pos"]), 1)
        self.assertEqual(sum(row["termination_counts"]["ground_contact"]), 1)

    def test_disjoint_windows_conserve_counts_and_preserve_episode_age(self):
        d = self.recorder(window_steps=2)
        d.record_starts(torch.tensor([0, 1]), torch.tensor([41, 42]), torch.full((4,), 0.25))
        self.observe(d, 1)
        self.observe(d, 2)
        self.observe(d, 3, fired={"anchor_pos": [0]})
        d.flush()
        d.flush()  # No duplicate empty tail.
        rows = self.fs.rows(d)[1:]
        self.assertEqual([row["control_steps"] for row in rows], [2, 1])
        self.assertEqual([row["partial"] for row in rows], [False, True])
        self.assertEqual(sum(sum(row["raw_by_frame"]["exposure_steps"]) for row in rows), 6)
        self.assertEqual(rows[1]["raw_by_frame"]["failures_by_age_steps"][42][2], 1)
        self.assertEqual(rows[1]["raw_by_frame"]["failures_by_start_frame"][41], 1)
        self.assertEqual(sum(rows[1]["raw_by_frame"]["reset_starts"]), 0)

    def test_resample_resets_age_and_records_selected_not_advanced_frame(self):
        d = self.recorder()
        self.observe(d, 1)
        d.record_starts(torch.tensor([0]), torch.tensor([70]), torch.full((4,), 0.25))
        self.observe(d, 2, frames=(71, 44), fired={"anchor_pos": [0]})
        d.flush()
        raw = self.fs.rows(d)[1]["raw_by_frame"]
        self.assertEqual(raw["reset_starts"][70], 1)
        self.assertEqual(raw["failures_by_start_frame"][70], 1)
        self.assertEqual(raw["failures"][71], 1)
        self.assertEqual(raw["failures_by_age_steps"][71][0], 1)

    def test_existing_directory_is_rejected_without_overwriting(self):
        d = self.recorder()
        before = self.fs.files[d.path].getvalue()
        with self.assertRaises(FileExistsError):
            self.recorder()
        self.assertEqual(self.fs.files[d.path].getvalue(), before)

    def test_invalid_settings_rejected_before_creating_files(self):
        for overrides in ({"window_steps": 0}, {"window_steps": 1.5}, {"fps": float("nan")},
                          {"fps": 0}, {"frame_count": 0}, {"output_dir": ""}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.recorder(**overrides)
        self.assertFalse(self.fs.files)
        self.assertFalse(self.fs.directories)

    def test_missing_step_is_rejected(self):
        d = self.recorder()
        with self.assertRaisesRegex(ValueError, "consecutive"):
            self.observe(d, 2)

    def test_write_failure_retains_counts_for_explicit_retry(self):
        d = self.recorder()
        self.observe(d, 1, fired={"anchor_pos": [0]})
        with patch.object(Path, "open", side_effect=OSError("disk unavailable")):
            with self.assertRaisesRegex(OSError, "disk unavailable"):
                d.flush()
        self.assertEqual(d.window_control_steps, 1)
        self.assertEqual(d.counts["failures"].sum().item(), 1)
        d.flush()
        self.assertEqual(len(self.fs.rows(d)), 2)

    def test_age_bucket_endpoints_and_overflow(self):
        d = self.recorder(num_envs=6)
        d.ages[:] = torch.tensor([0, 4, 5, 9, 99, 100])
        self.observe(d, 1, frames=(42,) * 6, fired={"anchor_pos": list(range(6))})
        d.flush()
        row = self.fs.rows(d)[1]
        ages = row["raw_by_frame"]["failures_by_age_steps"][42]
        self.assertEqual([ages[i] for i in (0, 4, 5, 8, 9)], [1, 1, 2, 1, 1])
        self.assertEqual(row["unknown_start_failures"], 6)

    def make_command(self, enabled):
        c = Command()
        c.cfg = SimpleNamespace(
            sampling_diagnostics_dir="/diagnostics/command" if enabled else None,
            sampling_diagnostics_window_steps=100,
            adaptive_uniform_ratio=0.1, adaptive_kernel_size=1, adaptive_alpha=0.001,
        )
        c.num_envs, c.device, c.bin_count = 2, "cpu", 4
        c.motion = SimpleNamespace(time_step_total=170, fps=50.0)
        flags = {name: torch.zeros(2, dtype=torch.bool) for name in TERMS}
        manager = SimpleNamespace(
            active_terms=list(TERMS), terminated=torch.zeros(2, dtype=torch.bool),
            get_term=flags.__getitem__, get_term_cfg=lambda name: SimpleNamespace(time_out=TERMS[name]),
        )
        c._env = SimpleNamespace(common_step_counter=0, termination_manager=manager)
        c._sampling_diagnostics = None
        c.time_steps = torch.zeros(2, dtype=torch.long)
        c.time_left = torch.zeros(2)
        c.bin_failed_count = torch.tensor([0.1, 0.7, 0.15, 0.05])
        c._current_bin_failed = torch.zeros(4)
        c.kernel = torch.ones(1)
        c.metrics = {name: torch.zeros(2) for name in ("sampling_entropy", "sampling_top1_prob", "sampling_top1_bin")}
        c._refresh_relative_body_targets = lambda: None
        c._resample_command = lambda ids: c._adaptive_sampling(ids) if len(ids) else None
        return c, flags

    def test_enabled_disabled_sampling_rng_and_command_outputs_are_identical(self):
        def rollout(enabled):
            torch.manual_seed(713)
            c, flags = self.make_command(enabled)
            c.reset()
            snapshots = []
            for step in range(1, 13):
                c._env.common_step_counter = step
                for value in flags.values():
                    value.zero_()
                c._env.termination_manager.terminated.zero_()
                if step % 3 == 0:
                    flags["anchor_pos"][0] = True
                    c._env.termination_manager.terminated[0] = True
                    c.reset(torch.tensor([0]))
                c.compute(0.02)
                snapshots.append((c.time_steps.clone(), c.bin_failed_count.clone(),
                                  {name: value.clone() for name, value in c.metrics.items()}))
            return c, snapshots, torch.random.get_rng_state().clone()

        disabled, expected, rng_off = rollout(False)
        self.assertIsNone(disabled._sampling_diagnostics)
        self.assertFalse(self.fs.files)
        enabled, actual, rng_on = rollout(True)
        self.assertTrue(torch.equal(rng_off, rng_on))
        for (frames_a, counts_a, metrics_a), (frames_b, counts_b, metrics_b) in zip(expected, actual):
            self.assertTrue(torch.equal(frames_a, frames_b))
            self.assertTrue(torch.equal(counts_a, counts_b))
            for name in metrics_a:
                self.assertTrue(torch.equal(metrics_a[name], metrics_b[name]))
        enabled.flush_sampling_diagnostics()
        row = self.fs.rows(enabled._sampling_diagnostics)[1]
        self.assertEqual(sum(row["raw_by_frame"]["exposure_steps"]), 24)
        self.assertEqual(sum(row["raw_by_frame"]["failures"]), 4)

    def test_command_reset_captures_terminating_frame_before_replacement(self):
        c, flags = self.make_command(True)
        c.reset()
        c.time_steps[:] = torch.tensor([84, 128])
        c._env.common_step_counter = 1
        c._env.termination_manager.terminated[0] = True
        flags["ground_contact"][0] = True
        c.reset(torch.tensor([0]))
        c.compute(0.02)
        c.flush_sampling_diagnostics()
        raw = self.fs.rows(c._sampling_diagnostics)[1]["raw_by_frame"]
        self.assertEqual(raw["failures"][84], 1)
        self.assertEqual(raw["exposure_steps"][128], 1)
        self.assertEqual(sum(raw["exposure_steps"]), 2)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Compare an approved wheel ROA checkpoint with immutable historical evidence.

Static mode executes one new model on the archived Sep24 input protocol. Closed
mode reads completed Native bundles; neither mode launches training or exports.
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import assess_wheel_roa_scale as legacy
from evidence_provenance import capture_evaluator_source, read_reference
from run_evaluation_batch import validate_contract
from summarize_evaluation_batch import validate_batch

ROOT = legacy.ROOT
TASK = legacy.TASK
MAX_VECTORS = 50000


def load_inputs(args):
    contract_path = Path(args.contract).resolve()
    contract = json.loads(contract_path.read_bytes())
    identity = validate_contract(contract)
    if identity["task"] != TASK or identity["seed"] != 42:
        raise ValueError("Expected the approved wheel-flat ROA task and seed 42")
    root = legacy.BASE / identity["run_id"]
    config = read_reference(contract["effective_config"])
    checkpoint = legacy.read_ref(contract["checkpoint"])
    if checkpoint.name != "model_49999.pt":
        raise ValueError("This assessment is bound to the approved final checkpoint")
    import yaml
    log = ROOT / "logs/rsl_rl/LW_wheel_flat_roa" / identity["run_id"]
    env = yaml.load((log / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    agent = yaml.load((log / "params/agent.yaml").read_text(), Loader=yaml.BaseLoader)
    if agent["policy"]["use_velocity_estimation"] != "false":
        raise ValueError("Expected the independently trained no-velocity policy")
    scale = legacy.resolve_scale(env)
    if len(contract["cases"]) != 6 or sum(c["scenario"]["duration_steps"] for c in contract["cases"]) != 30000:
        raise ValueError("The approved Native budget is six cases and 30000 steps")
    if any(c["scenario"]["num_envs"] != 1 or c["video"] for c in contract["cases"]):
        raise ValueError("Expected one environment and no video")
    previous = legacy.BASE / args.previous_run
    previous_static = previous / "evidence/analysis/static-rightturn-scale-comparison-20260928-001/metrics.json"
    previous_closed = previous / "evidence/analysis/flat-roa-scale-assess-20260928-001/metrics.json"
    return {
        "contract": contract, "identity": identity, "config": config,
        "contract_ref": legacy.ref(contract_path), "root": root, "scale": scale,
        "previous_static": previous_static, "previous_closed": previous_closed,
    }


def source_snapshot(root):
    return capture_evaluator_source(root, ROOT, [Path(__file__).resolve(), Path(legacy.__file__).resolve()])


def checked_json(path):
    value = json.loads(path.read_bytes())
    legacy.verify_historical_references(value)
    return value


def configure_kernel(inputs, args):
    # These overrides are process-local, recorded in the result, and never
    # change the existing helper's bytes or any historical artifact.
    legacy.RUN = inputs["identity"]["run_id"]
    legacy.LABEL = args.label
    legacy.STATIC_ID = args.static_id
    legacy.CHECKPOINT_SHA = inputs["contract"]["checkpoint"]["sha256"]


def gain_comparison(current, baseline):
    result = {}
    for stage, window in current["windows"].items():
        old = baseline["windows"][stage]
        values = {}
        for kind in ("q", "dq"):
            key = "full_current_" + kind
            numerator = window["sensitivity"][key]["spectral_norm_median"]
            denominator = old["sensitivity"][key]["spectral_norm_median"]
            values[kind + "_gain_ratio"] = numerator / denominator if denominator else None
            if stage in ("onset_left", "growth", "all_turn"):
                ratios = [values[kind + "_gain_ratio"]]
                for alignment in current["alternate_alignments"]:
                    num = current["alternate_alignments"][alignment][stage][key]["spectral_norm_median"]
                    den = baseline["alternate_alignments"][alignment][stage][key]["spectral_norm_median"]
                    ratios.append(num / den if den else None)
                values[kind + "_gain_ratio_across_alignments_min_max"] = [min(ratios), max(ratios)] if all(v is not None for v in ratios) else None
        for key in ("hip_target_abs_peak_rad", "hip_target_first_difference_rms_rad"):
            num, den = window["outputs"]["baseline"][key], old["outputs"]["baseline"][key]
            values[key + "_ratio"] = num / den if den else None
        result[stage] = values
    return result


def static_analysis(inputs, args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.monotonic()
    configure_kernel(inputs, args)
    source = source_snapshot(inputs["root"])
    if legacy.sha(legacy.OLD_STATIC) != legacy.STATIC_SHA:
        raise ValueError("Archived Sep24 protocol changed")
    historical = checked_json(legacy.OLD_STATIC)
    previous = checked_json(inputs["previous_static"])
    old_auto_path = legacy.OLD_STATIC.with_name("autograd-verification.json")
    checked_json(old_auto_path)
    if historical["method"]["primary_alignment"] != previous["method"]["primary_alignment"]:
        raise ValueError("Static alignment contracts differ")
    if historical["data_refs"] != previous["data_refs"]:
        raise ValueError("Static hardware data references differ")
    kernel = legacy.static_kernel(historical)
    data_repo = Path(historical["data_repository"]["root"])
    archived_head = historical["data_repository"]["head"]
    current_head = subprocess.check_output(["git", "-C", str(data_repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(data_repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Hardware data repository must remain clean")
    # All archived input references have already passed their original hashes.
    # Rebind only the repository revision guard, preserving the data protocol.
    kernel = legacy.replace_once(kernel, f'assert data_head=="{archived_head}"',
                                 f'assert data_head=="{current_head}"')
    ns = {"_resolve_scale": legacy.resolve_scale, "_checkpoint_sha": legacy.CHECKPOINT_SHA}
    exec(compile(kernel, "<archived-kernel-one-approved-new-model>", "exec"), ns)
    primary_vectors = ns["total_vectors"]
    validation = legacy.autodiff(ns)
    vectors = ns["counter"]["vectors"]
    if vectors != primary_vectors + 2 * len(ns["H"]) or vectors > MAX_VECTORS:
        raise ValueError("Static model-input budget exceeded")
    if ns["model"].use_velocity_estimation:
        raise ValueError("Checkpoint unexpectedly has a velocity head")
    current = ns["results"][args.label]
    current.update(action_scale=ns["scale"].tolist(), use_velocity_estimation=False,
                   actor_input_dim=int(ns["model"].actor[0].in_features),
                   export_parity={"status": "not_requested"})
    prior_label = previous["new_model"]
    reused = {"DWAQ_old": historical["models"]["DWAQ_old"], prior_label: previous["models"][prior_label]}
    comparisons = {label: gain_comparison(current, value) for label, value in reused.items()}
    out = inputs["root"] / "evidence/analysis" / args.static_id
    out.mkdir(parents=True, exist_ok=False)
    samples = out / "samples-and-jacobians.npz"
    np.savez_compressed(samples, **ns["arrays"])
    import csv
    csv_path = out / "per-frame-comparison.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(ns["csvrows"][0]))
        writer.writeheader()
        writer.writerows(ns["csvrows"])
    labels = ["DWAQ_old", prior_label, args.label]
    short = {"DWAQ_old": "DWAQ Jun03", prior_label: "ROA Sep26 +velocity", args.label: "ROA Sep28 no velocity"}
    all_models = {**reused, args.label: current}
    with np.load(legacy.read_ref(historical["artifacts"]["samples_npz"]), allow_pickle=False) as old_arrays, np.load(legacy.read_ref(previous["artifacts"]["samples_npz"]), allow_pickle=False) as prior_arrays:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        turn = ns["masks"]["all_turn"]
        for label in labels:
            values = (ns["arrays"] if label == args.label else prior_arrays if label == prior_label else old_arrays)[label + ".baseline.targets"]
            for j in (0, 1):
                axes[0, j].plot(ns["t"][ns["output_rows"]][turn], values[turn, j], label=short[label])
        for j in (0, 1):
            axes[0, j].set_ylabel(("Right" if j == 0 else "Left") + " hip target (rad)")
            axes[0, j].set_xlabel("Recorded header time (s)")
            axes[0, j].legend(fontsize=8)
            axes[0, j].grid(alpha=.2)
        for j, kind in enumerate(("q", "dq")):
            for offset, stage in ((-.18, "onset_left"), (.18, "growth")):
                values = [all_models[label]["windows"][stage]["sensitivity"]["full_current_" + kind]["spectral_norm_median"] for label in labels]
                axes[1, j].bar(np.arange(3) + offset, values, .36, label=stage)
            axes[1, j].set_xticks(range(3), [short[k] for k in labels], rotation=12, ha="right", fontsize=9)
            axes[1, j].set_ylabel("Gq (rad/rad)" if kind == "q" else "Gdq (rad/(rad/s))")
            axes[1, j].legend()
            axes[1, j].grid(axis="y", alpha=.2)
        fig.suptitle("Sep24 frozen hardware inputs | only Sep28 policy newly computed")
        plot = out / "comparison.png"
        pdf = out / "comparison.pdf"
        fig.savefig(plot, dpi=150)
        fig.savefig(pdf)
        plt.close(fig)
    method = copy.deepcopy(historical["method"])
    method.pop("action_scale")
    method.update(new_action_scale=ns["scale"].tolist(), recorded_previous_action_scale=[.25] * 8 + [1., 1.],
                  previous_action_contract="Identical original recorded raw actions for all policies; no target-equivalent remapping or predicted feedback.",
                  finite_budget={"new_policies": 1, "maximum_model_input_vectors": MAX_VECTORS,
                                 "primary_model_input_vectors": primary_vectors, "actual_model_input_vectors": vectors,
                                 "one_CPU_thread": True, "wall_time_limit_s": 300})
    value = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
             "analysis_kind": "static_frozen_hardware_inputs", "new_model": args.label,
             "run_identity": inputs["contract"]["run_identity"], "effective_config": inputs["contract"]["effective_config"],
             "execution_input": inputs["contract_ref"], "analysis_argv": sys.argv,
             "models": {args.label: current}, "reused_models": reused,
             "historical_results": legacy.ref(legacy.OLD_STATIC), "previous_results": legacy.ref(inputs["previous_static"]),
             "historical_autograd": legacy.ref(old_auto_path), "historical_policy_inference_count": 0,
             "comparisons_vs_reused_models": comparisons, "data_refs": historical["data_refs"], "method": method,
             "hardware_data_repository": {"root": str(data_repo), "head": current_head,
                                          "archived_head": archived_head, "worktree_clean": True,
                                          "all_referenced_input_bytes_unchanged": True},
             "verification": {"autograd": validation, "new_weights_and_rng_unchanged": True,
                              "reconstructed_inputs_match_archived_tolerance": ns["archive_input_error"]},
             "source": source, "recovered_kernel_sha256": hashlib.sha256(kernel.encode()).hexdigest(),
             "artifacts": {"samples_npz": legacy.ref(samples), "per_frame_csv": legacy.ref(csv_path),
                           "plot": legacy.ref(plot), "pdf": legacy.ref(pdf)},
             "runtime": {"conda_environment": "isaacsim-5.1", "device": "CPU", "elapsed_s": time.monotonic() - started},
             "limitations": historical["limitations"] + [
                 "Both recent policies have hip scale .125; original recorded .25 raw-action history may be out of distribution.",
                 "The two ROA models were independently trained; disabling the head also changes actor dimension and auxiliary supervision."],
             "direct_parameter_change_supported": False,
             "policy_description_required": "仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。"}
    legacy.write_json(out / "metrics.json", value)
    with np.load(samples, allow_pickle=False) as persisted:
        if not all(np.array_equal(persisted[k], v) for k, v in ns["arrays"].items()):
            raise ValueError("Persisted samples differ")
    print(json.dumps({"metrics": legacy.ref(out / "metrics.json"), "model_input_vectors": vectors,
                      "gains": {stage: {k: current["windows"][stage]["sensitivity"]["full_current_" + k]["spectral_norm_median"] for k in ("q", "dq")} for stage in ("onset_left", "growth", "all_turn")}}, ensure_ascii=False), flush=True)


def closed_kernel(historical):
    kernel = historical["analysis_source"]["python"]
    if hashlib.sha256(kernel.encode()).hexdigest() != historical["analysis_source"]["sha256"]:
        raise ValueError("Historical closed analysis source changed")
    node = next(n for n in ast.parse(kernel).body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    source = ast.get_source_segment(kernel, node)
    source = legacy.replace_once(source, "def analyze(result_path):", "def analyze(result_path, action_scale):")
    source = legacy.replace_once(source, "target=a*np.array([.25]*8+[1]*2)", "target=a*np.asarray(action_scale)")
    source = legacy.replace_once(source, "'roa_student_velocity_b' in s[0]", "s[0].get('roa_student_velocity_b') is not None")
    source = legacy.replace_once(source, "  result['stages'][name]=xx", "  if len(ix)==hi-lo:\n   jp=arr('joint_position')[ix][:,h]\n   band=spectral_band_rms(jp,5)\n   xx['hip_position_band_5_25Hz_rms_rad']=float(np.sqrt(np.mean(band**2)))\n  result['stages'][name]=xx")
    return source


def closed_analysis(inputs, args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest_path = inputs["root"] / "evaluations" / inputs["contract"]["batch_id"] / "manifest.json"
    manifest = validate_batch(manifest_path)
    if any(c["status"] != "completed" for c in manifest["cases"]):
        raise ValueError("Closed comparison requires every approved case completed")
    historical = checked_json(legacy.OLD_CLOSED)
    previous = checked_json(inputs["previous_closed"])
    validate_batch(legacy.read_ref(previous["batch"]), expected_sha256=previous["batch"]["sha256"])
    kernel = closed_kernel(historical)
    ns = {"Path": Path, "json": json, "gzip": gzip, "np": np, "ref": legacy.ref,
          "old": historical, "joint_order": legacy.JOINT_ORDER}
    exec(compile(kernel, "<matching-archived-closed-analysis-physical-scales>", "exec"), ns)
    source = source_snapshot(inputs["root"])
    cases, comparisons = {}, {}
    for case in manifest["cases"]:
        scenario = case["scenario"]["scenario_id"]
        current = ns["analyze"](case["result"]["path"], inputs["scale"])
        baseline = previous["cases"][scenario]
        if current["scenario"] != baseline["scenario"]:
            raise ValueError("Previous and new scenario contracts differ")
        changes = {}
        for stage, new in current["stages"].items():
            old = baseline["stages"][stage]
            changes[stage] = {k: (v / old[k] - 1) * 100 for k, v in new.items()
                              if isinstance(v, (float, int)) and isinstance(old.get(k), (float, int)) and old[k] != 0}
        telemetry = json.loads(gzip.decompress(legacy.read_ref(current["telemetry"]).read_bytes()))
        actual = np.array([s["action"] for s in telemetry["samples"]])
        if all(s.get("roa_student_action") is not None for s in telemetry["samples"]):
            diagnostic = np.array([s["roa_student_action"] for s in telemetry["samples"]])
            current["student_action_max_error"] = float(np.max(np.abs(actual - diagnostic)))
            if current["student_action_max_error"] != 0:
                raise ValueError("Executed actions differ from the student branch")
        current["use_velocity_estimation"] = False
        cases[scenario] = current
        comparisons[scenario] = {"scenario_contract_equal": True, "change_percent": changes}
    out = inputs["root"] / "evidence/analysis" / inputs["contract"]["batch_id"]
    out.mkdir(parents=True, exist_ok=False)
    metrics = [("hip_target_step_rms_rad", "Hip target step RMS (rad)"),
               ("hip_velocity_band_5_25Hz_rms_rad_s", "Hip velocity 5-25 Hz RMS (rad/s)"),
               ("body_roll_pitch_angvel_rms_rad_s", "Body roll/pitch rate RMS (rad/s)"),
               ("tracking_yaw_rmse_rad_s", "Yaw tracking RMSE (rad/s)")]
    turn_ids = [c["scenario"]["scenario_id"] for c in manifest["cases"][:4]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, metrics):
        for j, (collection, label) in enumerate(((previous["cases"], "Sep26 +velocity"), (cases, "Sep28 no velocity"))):
            values = [collection[s]["stages"]["turn_steady_42_70s"].get(metric, np.nan) for s in turn_ids]
            axis.bar(np.arange(4) + (j - .5) * .36, values, .36, label=label)
        axis.set_xticks(range(4), [s.replace("-noise", "") for s in turn_ids], rotation=15)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Matching Native scenarios | steady turn 42-70 s | one env, seed 42")
    plot, pdf = out / "comparison.png", out / "comparison.pdf"
    fig.savefig(plot, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    stand_ids = [c["scenario"]["scenario_id"] for c in manifest["cases"][4:]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric, title in zip(axes, ("net_xy_displacement_m", "body_roll_pitch_angvel_rms_rad_s", "hip_target_step_rms_rad"), ("Net drift (m)", "Body roll/pitch RMS (rad/s)", "Hip target step RMS (rad)")):
        for j, (collection, label) in enumerate(((previous["cases"], "Sep26 +velocity"), (cases, "Sep28 no velocity"))):
            values = [collection[s]["stages"]["standing_40_120s"].get(metric) for s in stand_ids]
            axis.bar(np.arange(2) + (j - .5) * .36, [np.nan if v is None else v for v in values], .36, label=label)
        axis.set_xticks(range(2), ["No noise", "Training noise"])
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Zero-command standing | 40-120 s | net drift is unavailable across resets")
    stand_plot = out / "standing.png"
    fig.savefig(stand_plot, dpi=150)
    plt.close(fig)
    value = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
             "batch": legacy.ref(manifest_path), "historical_analysis": legacy.ref(inputs["previous_closed"]),
             "historical_kernel": legacy.ref(legacy.OLD_CLOSED), "historical_policy_inference_count": 0,
             "run_identity": inputs["contract"]["run_identity"], "effective_config": inputs["contract"]["effective_config"],
             "execution_input": inputs["contract_ref"], "analysis_argv": sys.argv,
             "method": {**copy.deepcopy(previous["method"]), "new_action_scale": inputs["scale"].tolist(),
                        "hip_target": "Every model uses its own physical action scales; both recent ROA policies use hip .125.",
                        "historical_metrics_recomputed": False},
             "cases": cases, "comparisons": comparisons, "source": source,
             "analysis_kernel_sha256": hashlib.sha256(kernel.encode()).hexdigest(),
             "static_analysis": legacy.ref(inputs["root"] / "evidence/analysis" / args.static_id / "metrics.json"),
             "artifacts": {"plot": legacy.ref(plot), "pdf": legacy.ref(pdf), "standing_plot": legacy.ref(stand_plot)},
             "assessment": {"convergence": "indeterminate", "reason": "No approved acceptance thresholds"},
             "limitations": ["One seed and one environment; simulation evidence only.",
                             "30 ms action delay exceeds the trained 0-15 ms range; sensor age is not reproduced.",
                             "Physical target metrics describe requests before actuator buffering.",
                             "Independently trained policies do not isolate the causal effect of deleting a velocity input."],
             "direct_parameter_change_supported": False,
             "policy_description_required": "仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。"}
    legacy.write_json(out / "metrics.json", value)
    index = inputs["root"] / "index.md"
    with index.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## 本轮对照分析\n\n- [新策略静态双髋敏感度](evidence/analysis/{args.static_id}/metrics.json)\n- [静态对照图](evidence/analysis/{args.static_id}/comparison.png)\n- [与上一轮 ROA 的闭环对照](evidence/analysis/{inputs['contract']['batch_id']}/metrics.json)\n- [站立保持对照](evidence/analysis/{inputs['contract']['batch_id']}/standing.png)\n- [训练源码与批准预算](evidence/source/launch-and-budget-20260929-001.json)\n")
    print(json.dumps({"metrics": legacy.ref(out / "metrics.json"), "completed_cases": len(cases),
                      "new_policy_steps": sum(c["scenario"]["duration_steps"] for c in manifest["cases"])}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("static", "closed"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--previous-run", default="2026-09-26_11-37-09")
    parser.add_argument("--static-id", default="static-rightturn-no-velocity-comparison-20260929-001")
    parser.add_argument("--label", default="ROA_0928_final_no_velocity_scale0125")
    args = parser.parse_args()
    if Path(sys.prefix).name != "isaacsim-5.1":
        parser.error("Use the isaacsim-5.1 conda environment")
    inputs = load_inputs(args)
    if args.mode == "static":
        def deadline(_signum, _frame):
            raise TimeoutError("Approved 300-second static budget exhausted")
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(300)
        try:
            static_analysis(inputs, args)
        finally:
            signal.alarm(0)
    else:
        closed_analysis(inputs, args)


if __name__ == "__main__":
    main()

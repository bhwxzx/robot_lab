#!/usr/bin/env python3
"""Bounded Sep26 ROA analysis using immutable Sep24 protocols and evidence.

Run in isaacsim-5.1. The static kernel is recovered from hash-verified historical
source, restricted to one new checkpoint, and adapted only for physical scales.
No historical policy inference, exports, simulator launch or training occurs here.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import gzip
import hashlib
import json
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK = "RobotLab-Isaac-Velocity-Flat-LW-wheel-Roa-v0"
RUN = "2026-09-26_11-37-09"
OLD_RUN = "2026-09-23_17-23-24"
BASE = ROOT / "learnings/policy_tuning" / TASK
OLD_STATIC = BASE / OLD_RUN / "evidence/analysis/static-rightturn-policy-comparison-20260926-001/metrics.json"
OLD_CLOSED = BASE / OLD_RUN / "evidence/analysis/flat-roa-final-assess-20260924-001/metrics.json"
STATIC_ID = "static-rightturn-scale-comparison-20260928-001"
BATCH = "flat-roa-scale-assess-20260928-001"
LABEL = "ROA_0926_final_scale0125"
CHECKPOINT_SHA = "cd50d786d48b8420159fc76145b84ab52ba72a9be39434a747017d2769b212bd"
STATIC_SHA = "46ee9bb22541c0b9829acdfdef186d6ff3a2b43549c78e274ec86a4406d00579"
MAX_VECTORS = 50000
JOINT_ORDER = [f"{side}_{joint}_joint" for joint in ("hip", "thigh", "shank", "foot", "wheel") for side in ("right", "left")]
sys.path.insert(0, str(ROOT / ".agents/skills/monitor-tune-isaaclab-training/scripts"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def read_ref(binding):
    path = Path(binding["path"])
    if path.is_symlink() or sha(path) != binding["sha256"]:
        raise ValueError(f"Changed evidence: {path}")
    return path


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def source_ref():
    return ref(Path(__file__))


def verify_historical_references(value):
    seen = set()
    def visit(item):
        if isinstance(item, dict):
            if "path" in item and "sha256" in item and item["path"] not in seen:
                read_ref(item)
                seen.add(item["path"])
            for key, child in item.items():
                if key not in {"source", "analysis_source"}:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    return len(seen)


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Historical kernel no longer matches: {old[:80]}")
    return source.replace(old, new, 1)


def resolve_scale(env):
    import numpy as np
    values = []
    for name in JOINT_ORDER:
        cfg = env["actions"]["joint_vel" if "wheel" in name else "joint_pos"]
        scale = cfg["scale"]
        if isinstance(scale, dict):
            matches = [float(v) for pattern, v in scale.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous action scale: {name}")
            values.append(matches[0])
        else:
            values.append(float(scale))
    result = np.array(values, dtype=np.float32)
    if not np.array_equal(result, np.array([.125, .125] + [.25] * 6 + [1., 1.], np.float32)):
        raise ValueError("Unexpected Sep26 physical action mapping")
    return result


def static_kernel(historical):
    source = historical["analysis_source"]["python"]
    if hashlib.sha256(source.encode()).hexdigest() != historical["analysis_source"]["sha256"]:
        raise ValueError("Historical analysis source hash mismatch")
    # Recover only computation: the old publication/comparison code is excluded.
    source = source[:source.index("assert total_vectors<=400000")]
    source = replace_once(source, f'runroot=repo/"learnings/policy_tuning/{TASK}/{OLD_RUN}"',
                          f'runroot=repo/"learnings/policy_tuning/{TASK}/{RUN}"')
    source = replace_once(source, 'out=runroot/"evidence/analysis/static-rightturn-policy-comparison-20260926-001"',
                          f'out=runroot/"evidence/analysis/{STATIC_ID}"')
    first, last = source.index("cases=[\n"), source.index("\nprevious=")
    source = source[:first] + f'cases=[("{LABEL}","LW_wheel_flat_roa","{RUN}","model_49999.pt")]' + source[last:]
    source = replace_once(source, ' assert float(env["actions"]["joint_pos"]["scale"])==.25',
                          ' scale=_resolve_scale(env)\n hip_scale_tensor=torch.from_numpy(scale[:2])')
    source = replace_once(source, ' checkpoint=ckroot/checkpoint_name;checkpoint_ref=ref(checkpoint)',
                          ' checkpoint=ckroot/checkpoint_name;checkpoint_ref=ref(checkpoint)\n assert checkpoint_ref["sha256"]==_checkpoint_sha')
    # recorded_raw was already reconstructed with the hardware's original .25 scale.
    for old, new in [
        ('((ap[:,:2]-am[:,:2])*.25/(2*eps*factor))', '((ap[:,:2]-am[:,:2])*hip_scale_tensor/(2*eps*factor))'),
        ('((infer(hp)[:,:2]-infer(hm)[:,:2])*.25/(2*eps))', '((infer(hp)[:,:2]-infer(hm)[:,:2])*hip_scale_tensor/(2*eps))'),
    ]:
        source = replace_once(source, old, new)
    first, last = source.index(" export_paths=[]"), source.index(" sample_indices=")
    source = source[:first] + " export_paths=[]\n jit_path=None\n" + source[last:]
    # Fail before executing if the kernel could instantiate an older model.
    cases = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "cases" for t in node.targets)]
    assert len(cases) == 1
    assert ast.literal_eval(cases[0].value) == [(LABEL, "LW_wheel_flat_roa", RUN, "model_49999.pt")]
    return source


def autodiff(ns):
    import numpy as np
    import torch
    model, hist = ns["model"], ns["H"]
    rng, weights = torch.get_rng_state().clone(), ns["weight_hash"](model)
    arrays, checks, forward = {}, {}, {}
    def graph(h, detach):
        latent, velocity = model.history_encoder(h.flatten(1))
        if detach:
            latent, velocity = latent.detach(), velocity.detach()
        inputs = [h[:, -1], velocity, latent] if model.use_velocity_estimation else [h[:, -1], latent]
        return model.actor(torch.cat(inputs, dim=-1))
    for detach in (False, True):
        h = hist.clone().requires_grad_(True)
        y = graph(h, detach)
        forward[str(detach)] = float((y.detach() - ns["baseline"]).abs().max())
        assert forward[str(detach)] < 1e-6
        ns["counter"]["vectors"] += len(h)
        ns["counter"]["calls"] += 1
        assert ns["counter"]["vectors"] <= MAX_VECTORS
        grads = [torch.autograd.grad(y[:, j].sum() * float(ns["scale"][j]), h,
                                    retain_graph=j == 0)[0].detach().numpy() for j in (0, 1)]
        g = np.stack(grads, axis=2)
        for kind, cols, inscale in (("q", [9, 10], 1.), ("dq", [19, 20], .05)):
            if detach:
                arrays["actor_direct_" + kind] = g[:, -1, :, :][:, :, cols] * inscale
            else:
                arrays["full_current_" + kind] = g[:, -1, :, :][:, :, cols] * inscale
                arrays["history_only_coherent_bias_" + kind] = g[:, :-1, :, :][:, :, :, cols].sum(axis=1) * inscale
                arrays["all_history_coherent_bias_" + kind] = g[:, :, :, :][:, :, :, cols].sum(axis=1) * inscale
    for mode, exact in arrays.items():
        checks[mode] = {}
        ns["arrays"][LABEL + "." + mode + ".autograd"] = exact
        for stage in ("pre_turn", "onset_left", "growth", "all_turn"):
            mask = ns["masks"][stage]
            gain = np.linalg.svd(exact[mask].astype(float), compute_uv=False)[:, 0]
            entry = {"autograd_gain_median": float(np.median(gain))}
            for epsilon, fd in enumerate(ns["jac"][mode], 1):
                fd_gain = np.linalg.svd(fd[mask], compute_uv=False)[:, 0]
                entry[f"eps{epsilon}_gain_median_relative_error"] = float(abs(np.median(fd_gain) - np.median(gain)) / max(np.median(gain), 1e-12))
                entry[f"eps{epsilon}_matrix_absolute_error_max"] = float(np.linalg.norm((fd-exact)[mask], axis=(1, 2)).max())
            if mode in ("full_current_q", "full_current_dq"):
                assert entry["eps1_gain_median_relative_error"] < .01
            checks[mode][stage] = entry
    assert torch.equal(rng, torch.get_rng_state())
    assert weights == ns["weight_hash"](model)
    assert all(p.grad is None for p in model.parameters())
    return {"forward_errors": forward, "jacobian_checks": checks,
            "additional_input_vectors": 2 * len(hist), "weights_rng_and_gradient_buffers_unchanged": True}


def static_analysis():
    import numpy as np
    started = time.monotonic()
    assert sha(OLD_STATIC) == STATIC_SHA
    historical = json.loads(OLD_STATIC.read_bytes())
    count = verify_historical_references(historical)
    old_auto_path = OLD_STATIC.with_name("autograd-verification.json")
    old_auto = json.loads(old_auto_path.read_bytes())
    count += verify_historical_references(old_auto)
    source = static_kernel(historical)
    ns = {"_resolve_scale": resolve_scale, "_checkpoint_sha": CHECKPOINT_SHA}
    exec(compile(source, "<verified-Sep24-kernel-restricted-to-Sep26>", "exec"), ns)
    primary_vectors = ns["total_vectors"]
    validation = autodiff(ns)
    vectors = ns["counter"]["vectors"]
    assert vectors == primary_vectors + validation["additional_input_vectors"] <= MAX_VECTORS
    current = ns["results"][LABEL]
    current.update(action_scale=ns["scale"].tolist(), export_parity={"status": "not_requested"})
    comparisons = {}
    for label, previous in historical["models"].items():
        comparisons[label] = {}
        for stage, window in current["windows"].items():
            old = previous["windows"][stage]
            ratios = {}
            for kind in ("q", "dq"):
                key = "full_current_" + kind
                ratios[kind + "_gain_ratio"] = window["sensitivity"][key]["spectral_norm_median"] / old["sensitivity"][key]["spectral_norm_median"]
            for key in ("hip_target_abs_peak_rad", "hip_target_first_difference_rms_rad"):
                ratios[key + "_ratio"] = window["outputs"]["baseline"][key] / old["outputs"]["baseline"][key]
            if stage in ("onset_left", "growth", "all_turn"):
                for kind in ("q", "dq"):
                    aligned = [ratios[kind + "_gain_ratio"]]
                    for alignment in current["alternate_alignments"]:
                        aligned.append(current["alternate_alignments"][alignment][stage]["full_current_"+kind]["spectral_norm_median"] / previous["alternate_alignments"][alignment][stage]["full_current_"+kind]["spectral_norm_median"])
                    ratios[kind + "_gain_ratio_across_alignments_min_max"] = [min(aligned), max(aligned)]
            comparisons[label][stage] = ratios
    out = BASE / RUN / "evidence/analysis" / STATIC_ID
    out.mkdir(parents=True, exist_ok=False)
    npz_path = out / "samples-and-jacobians.npz"
    np.savez_compressed(npz_path, **ns["arrays"])
    csv_path = out / "per-frame-comparison.csv"
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(ns["csvrows"][0]))
        writer.writeheader()
        writer.writerows(ns["csvrows"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(read_ref(historical["artifacts"]["samples_npz"]), allow_pickle=False) as old_samples:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
        turn = ns["masks"]["all_turn"]
        for label in ("DWAQ_old", "ROA_0923_final", LABEL):
            samples = ns["arrays"] if label == LABEL else old_samples
            values = samples[label + ".baseline.targets"]
            for j in (0, 1):
                axes[0, j].plot(ns["t"][ns["output_rows"]][turn], values[turn, j], label=label)
        for j in (0, 1):
            axes[0, j].set_ylabel(("Right" if j == 0 else "Left") + " hip target (rad)")
            axes[0, j].set_xlabel("Recorded header time (s)")
            axes[0, j].legend(fontsize=8)
        labels = list(historical["models"]) + [LABEL]
        for j, kind in enumerate(("q", "dq")):
            for offset, stage in ((-.18, "onset_left"), (.18, "growth")):
                values = [(current if label == LABEL else historical["models"][label])["windows"][stage]["sensitivity"]["full_current_" + kind]["spectral_norm_median"] for label in labels]
                axes[1, j].bar(np.arange(len(labels)) + offset, values, .36, label=stage)
            axes[1, j].set_xticks(range(len(labels)), labels, rotation=35, ha="right", fontsize=8)
            axes[1, j].set_ylabel("Gq (rad/rad)" if kind == "q" else "Gdq (rad/(rad/s))")
            axes[1, j].legend()
        fig.suptitle("Sep24 frozen inputs | old results reused; only Sep26 newly measured")
        plot = out / "comparison.png"
        fig.savefig(plot, dpi=150)
        plt.close(fig)
    method = copy.deepcopy(historical["method"])
    method.pop("action_scale")
    method.update(recorded_previous_action_scale=[.25]*8+[1., 1.], new_action_scale=ns["scale"].tolist(),
                  previous_action_contract="Identical original recorded raw actions; no remapping and no predicted feedback. Sep26 physical targets differ for identical hip raw values.",
                  finite_budget={"new_policies": 1, "maximum_model_input_vectors": MAX_VECTORS,
                                 "primary_model_input_vectors": primary_vectors, "actual_model_input_vectors": vectors,
                                 "one_CPU_thread": True, "wall_time_limit_s": 300})
    result = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "analysis_kind": "static_frozen_hardware_inputs",
              "new_model": LABEL, "models": {LABEL: current}, "method": method,
              "historical_results": ref(OLD_STATIC), "historical_autograd": ref(old_auto_path),
              "reused_model_labels": list(historical["models"]), "historical_policy_inference_count": 0,
              "comparisons_vs_reused_models": comparisons, "data_refs": historical["data_refs"],
              "verification": {"historical_references_checked": count, "autograd": validation,
                               "reconstructed_inputs_match_archived_tolerance": ns["archive_input_error"],
                               "new_weights_and_rng_unchanged": True},
              "source": source_ref(), "recovered_kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "historical_kernel_sha256": historical["analysis_source"]["sha256"],
              "artifacts": {"samples_npz": ref(npz_path), "per_frame_csv": ref(csv_path), "plot": ref(plot)},
              "runtime": {"conda_environment": "isaacsim-5.1", "device": "CPU", "elapsed_s": time.monotonic()-started},
              "limitations": historical["limitations"] + ["Shared recorded .25 hip-action history may be out of distribution for the retrained .125 policy; target-equivalent remapping was not performed."],
              "direct_parameter_change_supported": False,
              "policy_description_required": "仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。"}
    write_json(out / "metrics.json", result)
    with np.load(npz_path, allow_pickle=False) as persisted:
        assert all(np.array_equal(persisted[k], v) for k, v in ns["arrays"].items())
    assert sha(OLD_STATIC) == STATIC_SHA
    print(json.dumps({"metrics": ref(out / "metrics.json"), "model_input_vectors": vectors,
                      "onset": current["windows"]["onset_left"]["sensitivity"]["full_current_q"],
                      "growth": current["windows"]["growth"]["sensitivity"]["full_current_q"]}, ensure_ascii=False))


def closed_analysis():
    import numpy as np
    import yaml
    from summarize_evaluation_batch import validate_batch
    from policy_evaluation_evidence import validate_evaluation_bundle
    manifest_path = BASE / RUN / "evaluations" / BATCH / "manifest.json"
    manifest = validate_batch(manifest_path)
    assert len(manifest["cases"]) == 6 and all(c["status"] == "completed" for c in manifest["cases"])
    historical = json.loads(OLD_CLOSED.read_bytes())
    tracked_bytes = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:" + str(OLD_CLOSED.relative_to(ROOT))])
    assert tracked_bytes == OLD_CLOSED.read_bytes()
    kernel = historical["analysis_source"]["python"]
    assert hashlib.sha256(kernel.encode()).hexdigest() == historical["analysis_source"]["sha256"]
    tree = ast.parse(kernel)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    source = ast.get_source_segment(kernel, node)
    source = replace_once(source, "def analyze(result_path):", "def analyze(result_path, action_scale):")
    source = replace_once(source, "target=a*np.array([.25]*8+[1]*2)", "target=a*np.asarray(action_scale)")
    source = replace_once(source, "  result['stages'][name]=xx", "  if len(ix)==hi-lo:\n   jp=arr('joint_position')[ix][:,h]\n   band=spectral_band_rms(jp,5)\n   xx['hip_position_band_5_25Hz_rms_rad']=float(np.sqrt(np.mean(band**2)))\n  result['stages'][name]=xx")
    ns = {"Path": Path, "json": json, "gzip": gzip, "np": np, "ref": ref,
          "old": historical, "joint_order": JOINT_ORDER}
    exec(compile(source, "<verified-Sep24-closed-analysis-with-model-scales>", "exec"), ns)
    env = yaml.load((ROOT / "logs/rsl_rl/LW_wheel_flat_roa" / RUN / "params/env.yaml").read_text(), Loader=yaml.BaseLoader)
    action_scale = resolve_scale(env)
    old_manifests = [historical["batch"]] + historical["historical_batches"]
    for binding in old_manifests:
        validate_batch(read_ref(binding), expected_sha256=binding["sha256"])
    comparisons, cases, supplemental = {}, {}, {}
    for case in manifest["cases"]:
        scenario = case["scenario"]["scenario_id"]
        current = ns["analyze"](case["result"]["path"], action_scale)
        cases[scenario] = current
        for key, previous in historical["cases"].items():
            if key.split("/")[-1] != scenario:
                continue
            assert previous["scenario"] == current["scenario"]
            validate_evaluation_bundle(read_ref(previous["result"]))
            telemetry = json.loads(gzip.decompress(read_ref(previous["telemetry"]).read_bytes()))
            hip_ids = [telemetry["joint_names"].index(j) for j in JOINT_ORDER[:2]]
            old_positions = np.array([s["joint_position"] for s in telemetry["samples"]])[:, hip_ids]
            supplemental[key] = {}
            changes = {}
            for stage, new in current["stages"].items():
                old = previous["stages"][stage]
                changes[stage] = {metric: (val/old[metric]-1)*100 for metric, val in new.items()
                                  if isinstance(val, (float, int)) and isinstance(old.get(metric), (float, int)) and old[metric] != 0}
                if new.get("hip_position_band_5_25Hz_rms_rad") is not None and not previous["done_steps"]:
                    lo, hi = historical["method"]["turn_stages_half_open_steps"][stage] if len(telemetry["samples"]) == 4500 else {
                        "startup_0_1s": [0, 50], "standing_1_40s": [50, 2000], "standing_40_120s": [2000, 6000]}[stage]
                    x = old_positions[lo:hi]
                    z = np.fft.rfft(x-x.mean(axis=0), axis=0)
                    freq = np.fft.rfftfreq(len(x), .02)
                    weights = np.full(len(freq), 2.)
                    weights[0] = 1.
                    if len(x) % 2 == 0:
                        weights[-1] = 1.
                    band = np.sqrt(np.sum(abs(z[freq >= 5])**2 * weights[freq >= 5, None], axis=0) / len(x)**2)
                    value = float(np.sqrt(np.mean(band**2)))
                    supplemental[key][stage] = {"hip_position_band_5_25Hz_rms_rad": value,
                                               "source": previous["telemetry"]}
                    changes[stage]["hip_position_band_5_25Hz_rms_rad"] = (new["hip_position_band_5_25Hz_rms_rad"]/value-1)*100 if value else None
            comparisons[scenario + " vs " + key] = {"scenario_contract_equal": True, "change_percent": changes}
    out = BASE / RUN / "evidence/analysis" / BATCH
    out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    scenarios = [c["scenario"]["scenario_id"] for c in manifest["cases"][:4]]
    metrics = [("hip_target_step_rms_rad", "Hip target step RMS (rad)"),
               ("hip_velocity_band_5_25Hz_rms_rad_s", "Hip velocity 5-25 Hz RMS (rad/s)"),
               ("body_roll_pitch_angvel_rms_rad_s", "Body roll/pitch rate RMS (rad/s)"),
               ("tracking_yaw_rmse_rad_s", "Yaw tracking RMSE (rad/s)")]
    for axis, (metric, title) in zip(axes.flat, metrics):
        for j, rid in enumerate(("2026-09-19_11-41-48", "2026-09-21_20-20-10", OLD_RUN, RUN)):
            values = [(cases[scenario] if rid == RUN else historical["cases"][rid+"/"+scenario])["stages"]["turn_steady_42_70s"][metric] for scenario in scenarios]
            axis.bar(np.arange(4)+(j-1.5)*.2, values, .2, label=rid[:10])
        axis.set_xticks(range(4), [s.replace("-noise", "") for s in scenarios], rotation=15)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    fig.suptitle("Matching Native scenarios | turn 42-70 s | only Sep26 newly evaluated")
    plot = out / "comparison.png"
    fig.savefig(plot, dpi=150)
    plt.close(fig)
    method = copy.deepcopy(historical["method"])
    method.update(hip_target="Raw actions converted with each model's effective physical scale; Sep26 hips .125, historical hips .25",
                  new_action_scale=action_scale.tolist(), historical_metrics_recomputed=False,
                  supplemental_historical_metric="Hip position band RMS derived from existing validated telemetry; no policy inference")
    result = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "batch": ref(manifest_path),
              "historical_analysis": ref(OLD_CLOSED), "historical_batches": old_manifests,
              "method": method, "cases": cases, "comparisons": comparisons,
              "supplemental_historical_position_metrics": supplemental, "plot": ref(plot),
              "source": source_ref(), "analysis_kernel_sha256": hashlib.sha256(source.encode()).hexdigest(),
              "static_analysis": ref(BASE / RUN / "evidence/analysis" / STATIC_ID / "metrics.json"),
              "assessment": {"convergence": "indeterminate", "reason": "No approved acceptance thresholds"},
              "direct_parameter_change_supported": False,
              "limitations": ["One seed, one environment; simulation evidence only.", "30 ms exceeds the trained 0-15 ms actuator delay range.", "Physical target metrics describe requested targets before actuator buffering, not measured actuator outputs."],
              "policy_description_required": "仅可进入受监督实物测试；未经实物验证，不代表 hardware-ready。"}
    write_json(out / "metrics.json", result)
    index = BASE / RUN / "index.md"
    with index.open("a", encoding="utf-8") as stream:
        stream.write(f"\n## 分析证据\n\n- [闭环比较](evidence/analysis/{BATCH}/metrics.json)\n- [静态敏感度](evidence/analysis/{STATIC_ID}/metrics.json)\n- [准确启动参数与预算](evidence/source/launch-and-budget-20260928-001.json)\n")
    print(json.dumps({"metrics": ref(out / "metrics.json"), "new_policy_steps": 30000,
                      "completed_cases": len(cases)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("static", "closed"))
    args = parser.parse_args()
    if Path(sys.prefix).name != "isaacsim-5.1":
        parser.error("Use the isaacsim-5.1 conda environment")
    if args.mode == "static":
        def deadline(_signum, _frame):
            raise TimeoutError("Approved 300-second static analysis budget exhausted")
        signal.signal(signal.SIGALRM, deadline)
        signal.alarm(300)
        try:
            static_analysis()
        finally:
            signal.alarm(0)
    else:
        closed_analysis()


if __name__ == "__main__":
    main()

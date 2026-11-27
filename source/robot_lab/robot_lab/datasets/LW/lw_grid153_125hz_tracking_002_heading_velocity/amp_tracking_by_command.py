"""Measure command tracking from the final 28-D LW AMP expert Frames."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory


TABLE = "amp_tracking_by_command.csv"
WINDOW = 10
# (command component, unit, expert frame column), in Command order.
AXES = (("vx", "mps", 0), ("vy", "mps", 1), ("wz", "rad_s", 5))


def first_frame_at_or_after(seconds, dt):
    index = seconds / dt
    nearest = round(index)
    # A phase boundary assembled from several stride periods can be a few ULPs off its tick.
    return nearest if abs(index - nearest) <= 8 * math.ulp(index) else math.ceil(index)


def percentile(ordered, fraction):
    rank = (len(ordered) - 1) * fraction
    lo, hi = math.floor(rank), math.ceil(rank)
    return ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])


def velocity_statistics(values, target):
    mean = fmean(values)
    windows = [fmean(values[i:i + WINDOW]) for i in range(len(values) - WINDOW + 1)]
    window_mean = fmean(windows) if windows else None
    ordered = sorted(windows)
    return {
        "frame_mean": mean, "frame_signed_error": mean - target, "frame_abs_error": abs(mean - target),
        "window10_mean": window_mean,
        "window10_p10": percentile(ordered, .10) if ordered else None,
        "window10_p90": percentile(ordered, .90) if ordered else None,
        "window10_signed_error": window_mean - target if windows else None,
        "window10_abs_error": abs(window_mean - target) if windows else None,
    }


def measure_clip(path):
    motion = json.loads(path.read_bytes())
    frames, cmd, dt = motion["Frames"], motion["Command"], motion["FrameDuration"]
    if (motion.get("RootVelocityFrame") != "heading" or motion.get("RootVelocityPoint") != "base_link"
            or motion.get("FrameLayout", [])[:2] != ["base_linear_velocity_heading", "base_angular_velocity_heading"]):
        raise ValueError(f"{path.name}: expected heading-frame base_link expert velocities")
    if (not frames or any(len(f) != 28 or not all(map(math.isfinite, f)) for f in frames)
            or len(cmd) != 3 or not all(map(math.isfinite, cmd)) or not math.isfinite(dt) or dt <= 0):
        raise ValueError(f"{path.name}: invalid expert frames, command or frame clock")

    if motion["ClipType"] == "episode":
        phase = "cruise"
        start, stop = motion["CruiseStart"], motion["BrakeStart"]
        if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
            raise ValueError(f"{path.name}: invalid cruise boundaries")
        first, end = first_frame_at_or_after(start, dt), first_frame_at_or_after(stop, dt)
    elif motion["ClipType"] == "stand":
        phase, start, stop = "stand", None, None
        first, end = 0, len(frames)
    else:
        raise ValueError(f"{path.name}: unsupported ClipType {motion['ClipType']!r}")
    if not 0 <= first < end <= len(frames):
        raise ValueError(f"{path.name}: empty or out-of-range measurement interval [{first}, {end})")

    row = {f"{axis}_command_{unit}": value for (axis, unit, _), value in zip(AXES, cmd)}
    row.update(motion_file=path.name, phase=phase, frame_duration_s=dt, cruise_start_s=start, brake_start_s=stop,
               first_frame=first, end_frame_exclusive=end, frame_count=end - first,
               window10_count=max(0, end - first - WINDOW + 1))
    selected = frames[first:end]
    for (axis, unit, column), target in zip(AXES, cmd):
        stats = velocity_statistics([frame[column] for frame in selected], target)
        row.update({f"{key}_{axis}_{unit}": value for key, value in stats.items()})
    return row


def command(row):
    return tuple(row[f"{axis}_command_{unit}"] for axis, unit, _ in AXES)


def cruise_error_summary(rows, prefix):
    threshold = .01
    eligible = [r for r in rows if r[f"{prefix}_abs_error_vx_mps"] is not None]
    summary = {"commands_measured": len(eligible)}
    over = set()
    for axis, unit in (("vx", "mps"), ("wz", "rad_s")):
        key = f"{prefix}_abs_error_{axis}_{unit}"
        above = {command(r) for r in eligible if r[key] > threshold}
        over.update(above)
        worst = max(eligible, key=lambda r: r[key], default=None)
        summary[axis] = {f"abs_error_threshold_{unit}": threshold,
                         "commands_above_threshold": len(above),
                         f"max_abs_error_{unit}": worst[key] if worst else None,
                         "command_at_max": list(command(worst)) if worst else None}
    summary["commands_above_either_threshold"] = len(over)
    summary["all_measured_commands_within_thresholds"] = not over if eligible else None
    return summary


def strict_frame_tracking(rows):
    """Check each command independently, including the full stand clip."""
    failures = set()
    summary = {"commands_measured": len(rows), "comparison": "absolute error < threshold"}
    for axis, unit in (("vx", "mps"), ("wz", "rad_s")):
        key = f"frame_abs_error_{axis}_{unit}"
        worst = max(rows, key=lambda row: row[key])
        outside = [command(row) for row in rows if row[key] >= .01]
        failures.update(outside)
        summary[axis] = {f"threshold_{unit}": .01, "commands_at_or_above_threshold": len(outside),
                         f"max_abs_error_{unit}": worst[key], "command_at_max": list(command(worst))}
    summary["commands_failing_either_axis"] = len(failures)
    summary["passed"] = not failures
    return summary


def write_tracking_table(root):
    """Write the CSV, refresh an existing release summary, and return the tracking summary."""
    rows = sorted((measure_clip(p) for p in (root / "motion_amp_expert").glob("motion_*.txt")), key=command)
    if not rows or len({command(r) for r in rows}) != len(rows):
        raise ValueError(f"{root}: expected one expert clip per command and at least one command")
    cruise = [r for r in rows if r["phase"] == "cruise"]
    stand = [r for r in rows if r["phase"] == "stand"]
    summary = {"table": TABLE, "source": "motion_amp_expert/motion_*.txt", "source_field": "Frames",
               "frame_columns": {axis: column for axis, _, column in AXES},
               "cruise_interval": "[CruiseStart, BrakeStart)", "window_frames": WINDOW, "window_stride_frames": 1,
               "window_percentiles": "linear quantiles of within-phase window means",
               "commands": len(rows), "cruise_commands": len(cruise), "stand_commands": len(stand),
               "cruise_frames": sum(r["frame_count"] for r in cruise),
               "cruise_windows": sum(r["window10_count"] for r in cruise),
               "stand_frames": sum(r["frame_count"] for r in stand),
               "stand_windows": sum(r["window10_count"] for r in stand),
               "strict_frame_tracking": strict_frame_tracking(rows),
               "cruise_frame_mean": cruise_error_summary(cruise, "frame"),
               "cruise_window10_mean": cruise_error_summary(cruise, "window10")}

    verification = root / "release_verification.json"
    with TemporaryDirectory(dir=root) as temp:
        staged = Path(temp)
        table_path = staged / TABLE
        with table_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        summary["table_sha256"] = hashlib.sha256(table_path.read_bytes()).hexdigest()
        if verification.is_file():
            record = json.loads(verification.read_bytes())
            record["amp_tracking"] = summary
            (staged / verification.name).write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False)
                                                    + "\n", encoding="utf-8")
        table_path.replace(root / TABLE)
        if (staged / verification.name).exists():
            (staged / verification.name).replace(verification)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Release or run directory containing motion_amp_expert")
    args = parser.parse_args()
    print(json.dumps(write_tracking_table(args.root), indent=2))

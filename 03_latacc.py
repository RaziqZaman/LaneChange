#!/usr/bin/env python3
"""Plot sampled SV lateral-acceleration traces around lane changes.

The script reads ``outputs/01_traces.csv``, samples 24 lane-change events,
plots the SV lateral-acceleration traces, and highlights the point at the
``lane_change_timestep`` for each trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRACES = Path("outputs/01_traces.csv")
DEFAULT_OUTPUT = Path("outputs/03_latacc.png")
DEFAULT_JERK_OUTPUT = Path("outputs/03_latjerk.png")
DEFAULT_VEL_OUTPUT = Path("outputs/03_latvel.png")
DEFAULT_OUTPUT_SMOOTH = Path("outputs/03_latacc-smooth.png")
DEFAULT_JERK_OUTPUT_SMOOTH = Path("outputs/03_latjerk-smooth.png")
DEFAULT_VEL_OUTPUT_SMOOTH = Path("outputs/03_latvel-smooth.png")
NUM_STEPS = 91  # t00 .. t90 inclusive, each 0.1 s
DT = 0.1
ROLLING_WINDOW = 21  # 10 steps before/after + current (±1 s)
ZERO_THRESH = 0.07
ZERO_WINDOW_STEPS = max(1, int(round(0.5 / DT)))  # 0.5 s worth of samples


def make_time_axis() -> np.ndarray:
    # 0.0, 0.1, ..., 9.0 (91 steps)
    return np.arange(NUM_STEPS) * DT


def make_jerk_time_axis() -> np.ndarray:
    # Differences between consecutive samples -> 90 points
    return np.arange(NUM_STEPS - 1) * DT


def get_lateral_columns() -> List[str]:
    return [f"t{idx:02d}_SV_lateral_accel" for idx in range(NUM_STEPS)]


def smooth_series(values: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """Centered rolling mean handling edges and missing values."""
    mask = np.isfinite(values)
    filled = np.where(mask, values, 0.0)
    kernel = np.ones(window, dtype=float)
    summed = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(mask.astype(float), kernel, mode="same")
    return np.divide(summed, counts, out=np.zeros_like(summed), where=counts > 0)


def find_markers(lat_acc: np.ndarray, switch_idx: Optional[int]) -> tuple[Optional[float], Optional[float]]:
    """Find maneuver start/end markers based on near-zero acceleration windows."""
    if switch_idx is None:
        return None, None
    n = len(lat_acc)
    idx = int(switch_idx)
    idx = max(0, min(idx, n - 1))

    start_marker: Optional[float] = None
    end_marker: Optional[float] = None

    # Backward scan: look for 1 s window of near-zero before switch
    for end in range(min(idx, n - 1), ZERO_WINDOW_STEPS - 1, -1):
        start = end - ZERO_WINDOW_STEPS + 1
        window = lat_acc[start : end + 1]
        if np.all(np.abs(window) <= ZERO_THRESH):
            for pos in range(end + 1, idx + 1):
                if abs(lat_acc[pos]) > ZERO_THRESH:
                    start_marker = pos * DT
                    break
            break

    # Forward scan: look for 1 s window of near-zero after switch
    for start in range(idx, n - ZERO_WINDOW_STEPS + 1):
        end = start + ZERO_WINDOW_STEPS - 1
        window = lat_acc[start : end + 1]
        if np.all(np.abs(window) <= ZERO_THRESH):
            for pos in range(start - 1, idx - 1, -1):
                if abs(lat_acc[pos]) > ZERO_THRESH:
                    end_marker = pos * DT
                    break
            break

    return start_marker, end_marker


def plot_sample(
    df: pd.DataFrame,
    output_path: Path,
    jerk_output_path: Path,
    vel_output_path: Path,
    output_path_smooth: Path,
    jerk_output_path_smooth: Path,
    vel_output_path_smooth: Path,
    sample_size: int,
    seed: Optional[int],
) -> None:
    sample_size = min(sample_size, len(df))
    sampled = df.sample(n=sample_size, random_state=seed)

    time_axis = make_time_axis()
    jerk_time_axis = make_jerk_time_axis()
    lat_cols = get_lateral_columns()

    ncols = 6
    nrows = int(np.ceil(sample_size / ncols))
    fig_acc, axes_acc = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    fig_jerk, axes_jerk = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    fig_vel, axes_vel = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    fig_acc_s, axes_acc_s = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    fig_jerk_s, axes_jerk_s = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    fig_vel_s, axes_vel_s = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 2.5 * nrows),
        sharex=True,
        sharey=True,
    )
    axes_acc_flat = axes_acc.flatten()
    axes_jerk_flat = axes_jerk.flatten()
    axes_vel_flat = axes_vel.flatten()
    axes_acc_s_flat = axes_acc_s.flatten()
    axes_jerk_s_flat = axes_jerk_s.flatten()
    axes_vel_s_flat = axes_vel_s.flatten()

    highlight_kwargs = {"color": "#e15759", "zorder": 5, "s": 35}
    line_kwargs = {"color": "#4c78a8", "linewidth": 1.1}
    jerk_line_kwargs = {"color": "#f28e2b", "linewidth": 1.1}
    vel_line_kwargs = {"color": "#59a14f", "linewidth": 1.1}

    for (
        ax_acc,
        ax_jerk,
        ax_vel,
        ax_acc_s,
        ax_jerk_s,
        ax_vel_s,
        (_, row),
    ) in zip(
        axes_acc_flat,
        axes_jerk_flat,
        axes_vel_flat,
        axes_acc_s_flat,
        axes_jerk_s_flat,
        axes_vel_s_flat,
        sampled.iterrows(),
    ):
        lat_acc = row[lat_cols].to_numpy(dtype=float)
        # Replace non-finite values to avoid NaNs wiping out curves
        lat_acc = np.where(np.isfinite(lat_acc), lat_acc, 0.0)
        # Finite difference for jerk; pad length NUM_STEPS-1
        lat_jerk = np.diff(lat_acc) / DT
        # Integrate acceleration to velocity (baseline at zero)
        lat_vel = np.cumsum(lat_acc) * DT
        lat_acc_s = smooth_series(lat_acc)
        lat_jerk_s = smooth_series(lat_jerk)
        lat_vel_s = smooth_series(lat_vel)
        start_line: Optional[float] = None
        end_line: Optional[float] = None

        ax_acc.plot(time_axis, lat_acc, **line_kwargs)
        ax_jerk.plot(jerk_time_axis, lat_jerk, **jerk_line_kwargs)
        ax_vel.plot(time_axis, lat_vel, **vel_line_kwargs)
        ax_acc_s.plot(time_axis, lat_acc_s, **line_kwargs)
        ax_jerk_s.plot(jerk_time_axis, lat_jerk_s, **jerk_line_kwargs)
        ax_vel_s.plot(time_axis, lat_vel_s, **vel_line_kwargs)

        step = row["lane_change_timestep"]
        if pd.notna(step):
            idx = int(step)
            if 0 <= idx < NUM_STEPS:
                ax_acc.scatter(
                    time_axis[idx],
                    lat_acc[idx],
                    **highlight_kwargs,
                )
                ax_vel.scatter(
                    time_axis[idx],
                    lat_vel[idx],
                    **highlight_kwargs,
                )
                ax_acc_s.scatter(
                    time_axis[idx],
                    lat_acc_s[idx],
                    **highlight_kwargs,
                )
                ax_vel_s.scatter(
                    time_axis[idx],
                    lat_vel_s[idx],
                    **highlight_kwargs,
                )
                start_line, end_line = find_markers(lat_acc, idx)
            jerk_idx = min(max(idx, 0), NUM_STEPS - 2)
            if 0 <= jerk_idx < len(lat_jerk):
                ax_jerk.scatter(
                    jerk_time_axis[jerk_idx],
                    lat_jerk[jerk_idx],
                    **highlight_kwargs,
                )
                ax_jerk_s.scatter(
                    jerk_time_axis[jerk_idx],
                    lat_jerk_s[jerk_idx],
                    **highlight_kwargs,
                )

        for t_cross in (start_line, end_line):
            if t_cross is None:
                continue
            for ax in (
                ax_acc,
                ax_jerk,
                ax_vel,
                ax_acc_s,
                ax_jerk_s,
                ax_vel_s,
            ):
                ax.axvline(t_cross, color="#8c8c8c", linestyle=":", linewidth=2.0, alpha=0.8)

        scenario = row.get("scenario_id", "")
        ax_acc.set_title(f"{scenario}", fontsize=8)
        ax_acc.grid(True, linewidth=0.4, alpha=0.6)
        ax_jerk.set_title(f"{scenario}", fontsize=8)
        ax_jerk.grid(True, linewidth=0.4, alpha=0.6)
        ax_vel.set_title(f"{scenario}", fontsize=8)
        ax_vel.grid(True, linewidth=0.4, alpha=0.6)
        ax_acc_s.set_title(f"{scenario}", fontsize=8)
        ax_acc_s.grid(True, linewidth=0.4, alpha=0.6)
        ax_jerk_s.set_title(f"{scenario}", fontsize=8)
        ax_jerk_s.grid(True, linewidth=0.4, alpha=0.6)
        ax_vel_s.set_title(f"{scenario}", fontsize=8)
        ax_vel_s.grid(True, linewidth=0.4, alpha=0.6)

    # Hide unused axes if any
    for ax in axes_acc_flat[sample_size:]:
        ax.axis("off")
    for ax in axes_jerk_flat[sample_size:]:
        ax.axis("off")
    for ax in axes_vel_flat[sample_size:]:
        ax.axis("off")
    for ax in axes_acc_s_flat[sample_size:]:
        ax.axis("off")
    for ax in axes_jerk_s_flat[sample_size:]:
        ax.axis("off")
    for ax in axes_vel_s_flat[sample_size:]:
        ax.axis("off")

    fig_acc.suptitle("SV lateral acceleration traces (highlight = lane_change_timestep)", fontsize=14)
    fig_acc.supxlabel("Time (s)")
    fig_acc.supylabel("Lateral acceleration")
    fig_acc.tight_layout()
    fig_acc.subplots_adjust(top=0.9)

    fig_jerk.suptitle("SV lateral jerk traces (highlight = lane_change_timestep)", fontsize=14)
    fig_jerk.supxlabel("Time (s)")
    fig_jerk.supylabel("Lateral jerk")
    fig_jerk.tight_layout()
    fig_jerk.subplots_adjust(top=0.9)

    fig_vel.suptitle("SV lateral velocity traces (highlight = lane_change_timestep)", fontsize=14)
    fig_vel.supxlabel("Time (s)")
    fig_vel.supylabel("Lateral velocity (integrated from accel)")
    fig_vel.tight_layout()
    fig_vel.subplots_adjust(top=0.9)

    fig_acc_s.suptitle(
        "SV lateral acceleration traces (2 s centered rolling mean, highlight = lane_change_timestep)",
        fontsize=14,
    )
    fig_acc_s.supxlabel("Time (s)")
    fig_acc_s.supylabel("Lateral acceleration (rolling mean)")
    fig_acc_s.tight_layout()
    fig_acc_s.subplots_adjust(top=0.9)

    fig_jerk_s.suptitle(
        "SV lateral jerk traces (2 s centered rolling mean, highlight = lane_change_timestep)",
        fontsize=14,
    )
    fig_jerk_s.supxlabel("Time (s)")
    fig_jerk_s.supylabel("Lateral jerk (rolling mean)")
    fig_jerk_s.tight_layout()
    fig_jerk_s.subplots_adjust(top=0.9)

    fig_vel_s.suptitle(
        "SV lateral velocity traces (2 s centered rolling mean, highlight = lane_change_timestep)",
        fontsize=14,
    )
    fig_vel_s.supxlabel("Time (s)")
    fig_vel_s.supylabel("Lateral velocity (rolling mean)")
    fig_vel_s.tight_layout()
    fig_vel_s.subplots_adjust(top=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_acc.savefig(output_path, dpi=200)
    fig_jerk.savefig(jerk_output_path, dpi=200)
    fig_vel.savefig(vel_output_path, dpi=200)
    fig_acc_s.savefig(output_path_smooth, dpi=200)
    fig_jerk_s.savefig(jerk_output_path_smooth, dpi=200)
    fig_vel_s.savefig(vel_output_path_smooth, dpi=200)
    print(
        "Saved plots to "
        f"{output_path}, {jerk_output_path}, {vel_output_path}, "
        f"{output_path_smooth}, {jerk_output_path_smooth}, and {vel_output_path_smooth}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot sampled SV lateral-acceleration traces around lane changes.",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=DEFAULT_TRACES,
        help="Path to outputs/01_traces.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the lateral-acceleration figure.",
    )
    parser.add_argument(
        "--jerk-output",
        type=Path,
        default=DEFAULT_JERK_OUTPUT,
        help="Output path for the lateral-jerk figure.",
    )
    parser.add_argument(
        "--vel-output",
        type=Path,
        default=DEFAULT_VEL_OUTPUT,
        help="Output path for the lateral-velocity figure.",
    )
    parser.add_argument(
        "--output-smooth",
        type=Path,
        default=DEFAULT_OUTPUT_SMOOTH,
        help="Output path for the smoothed lateral-acceleration figure.",
    )
    parser.add_argument(
        "--jerk-output-smooth",
        type=Path,
        default=DEFAULT_JERK_OUTPUT_SMOOTH,
        help="Output path for the smoothed lateral-jerk figure.",
    )
    parser.add_argument(
        "--vel-output-smooth",
        type=Path,
        default=DEFAULT_VEL_OUTPUT_SMOOTH,
        help="Output path for the smoothed lateral-velocity figure.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=24,
        help="Number of lane-change traces to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for sampling (omit for different sample each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lat_cols = get_lateral_columns()
    usecols = ["scenario_id", "lane_change_timestep", *lat_cols]

    df = pd.read_csv(args.traces, usecols=usecols)
    if df.empty:
        raise SystemExit(f"No rows found in {args.traces}")

    plot_sample(
        df,
        output_path=args.output,
        jerk_output_path=args.jerk_output,
        vel_output_path=args.vel_output,
        output_path_smooth=args.output_smooth,
        jerk_output_path_smooth=args.jerk_output_smooth,
        vel_output_path_smooth=args.vel_output_smooth,
        sample_size=args.sample_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

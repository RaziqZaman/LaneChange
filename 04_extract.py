#!/usr/bin/env python3
"""Extract lane-change metrics with initiation/midpoint/completion annotations.

Reads ``outputs/03_full-traces.csv`` and writes ``outputs/04_metrics.csv`` with:
    - scenario_id, initiation_timestep, midpoint_timestep, completion_timestep
    - role state (HD/AV/NP) for LC/RC/LT/RT
    - durations (total, approach, settle)
    - SV init/mid/comp/peak long/lat speed/acc/jerk
    - LC/RC/LT/RT init/mid/comp/min/max long_gap/rel_speed/ttc/headway
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


INPUT_PATH = Path("outputs/03_full-traces.csv")
OUTPUT_PATH = Path("outputs/04_metrics.csv")
DT = 0.1
NUM_STEPS = 91  # t00..t90
ROLES = ["LC", "RC", "LT", "RT"]
PHASES_SV = ["init", "mid", "comp", "peak"]
AXES = ["long", "lat"]
METRICS_SV = ["speed", "acc", "jerk"]
PHASES_ROLE = ["init", "mid", "comp", "min", "max"]
ROLE_METRICS = ["long_gap", "rel_speed", "ttc", "headway"]


def lateral_cols(prefix: str) -> List[str]:
    return [f"t{idx:02d}_{prefix}" for idx in range(NUM_STEPS)]


def extract_series(row: pd.Series, prefix: str) -> np.ndarray:
    cols = lateral_cols(prefix)
    series = row.get(cols, pd.Series([np.nan] * len(cols)))
    return pd.Series(series).to_numpy(dtype=float)


def jerks_from(acc: np.ndarray) -> np.ndarray:
    jerk = np.diff(acc) / DT
    return np.concatenate([jerk, [np.nan]])  # pad to length NUM_STEPS


def state_from_flag(flag: float) -> str:
    if pd.isna(flag):
        return ""
    return "AV" if int(flag) == 1 else "HD"


def safe_slice(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(start, 0)
    end = min(end, len(arr) - 1)
    if start > end:
        return np.array([], dtype=float)
    return arr[start : end + 1]


def pick_value(arr: np.ndarray, idx: int) -> float:
    return arr[idx] if 0 <= idx < len(arr) else np.nan


def closing_speed(rel_speed: np.ndarray) -> np.ndarray:
    # Assume positive relative speed means SV is closing the gap
    return np.where(rel_speed > 0, rel_speed, 0.0)


def ttc_and_headway(gap: np.ndarray, rel_speed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    close = closing_speed(rel_speed)
    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(close > 0, gap / close, np.inf)
        headway = np.where(close > 0, gap / close, np.inf)
    ttc[~np.isfinite(ttc)] = np.inf
    headway[~np.isfinite(headway)] = np.inf
    return ttc, headway


def build_output_columns() -> List[str]:
    cols: List[str] = [
        "scenario_id",
        "initiation_timestep",
        "midpoint_timestep",
        "completion_timestep",
    ]
    cols += [f"{role}_state" for role in ROLES]
    cols += [
        "lane_change_duration",
        "approach_duration",
        "settle_duration",
    ]
    for phase in PHASES_SV:
        for axis in AXES:
            for metric in METRICS_SV:
                cols.append(f"SV_{phase}_{axis}_{metric}")
    for role in ROLES:
        for phase in PHASES_ROLE:
            for metric in ROLE_METRICS:
                cols.append(f"{role}_{phase}_{metric}")
    return cols


def process_row(row: pd.Series) -> Dict[str, float]:
    mid_idx = int(row["midpoint_timestep"])
    init_idx = int(row["initiation_timestep"])
    comp_idx = int(row["completion_timestep"])

    sv_speed = extract_series(row, "SV_speed")
    sv_long_acc = extract_series(row, "SV_longitudinal_accel")
    sv_lat_acc = extract_series(row, "SV_lateral_accel")
    sv_long_jerk = jerks_from(sv_long_acc)
    sv_lat_jerk = jerks_from(sv_lat_acc)

    sv_axes = {
        "long": {"speed": sv_speed, "acc": sv_long_acc, "jerk": sv_long_jerk},
        "lat": {"speed": sv_speed, "acc": sv_lat_acc, "jerk": sv_lat_jerk},
    }

    data: Dict[str, float] = {
        "scenario_id": row["scenario_id"],
        "initiation_timestep": init_idx,
        "midpoint_timestep": mid_idx,
        "completion_timestep": comp_idx,
    }

    for role in ROLES:
        flag = row.get(f"{role.lower()}_is_SDC", np.nan)
        data[f"{role}_state"] = state_from_flag(flag)

    data["lane_change_duration"] = row["duration"]
    data["approach_duration"] = (mid_idx - init_idx) * DT
    data["settle_duration"] = (comp_idx - mid_idx) * DT

    window_start, window_end = init_idx, comp_idx
    for phase, idx in {"init": init_idx, "mid": mid_idx, "comp": comp_idx}.items():
        for axis in AXES:
            for metric in METRICS_SV:
                arr = sv_axes[axis][metric]
                data[f"SV_{phase}_{axis}_{metric}"] = pick_value(arr, idx)
    for axis in AXES:
        for metric in METRICS_SV:
            arr = sv_axes[axis][metric]
            window = safe_slice(arr, window_start, window_end)
            data[f"SV_peak_{axis}_{metric}"] = np.nanmax(window) if window.size else np.nan

    for role in ROLES:
        gap = extract_series(row, f"{role}_longitudinal_gap")
        rel = extract_series(row, f"{role}_relative_speed")
        ttc, headway = ttc_and_headway(gap, rel)

        series_map = {
            "long_gap": gap,
            "rel_speed": rel,
            "ttc": ttc,
            "headway": headway,
        }
        for phase, idx in {"init": init_idx, "mid": mid_idx, "comp": comp_idx}.items():
            for metric, arr in series_map.items():
                data[f"{role}_{phase}_{metric}"] = pick_value(arr, idx)
        for metric, arr in series_map.items():
            window = safe_slice(arr, window_start, window_end)
            data[f"{role}_min_{metric}"] = np.nanmin(window) if window.size else np.nan
            data[f"{role}_max_{metric}"] = np.nanmax(window) if window.size else np.nan

    return data


def process(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    if df.empty:
        raise SystemExit(f"No rows found in {input_path}")

    rows = []
    for _, row in df.iterrows():
        rows.append(process_row(row))

    out_cols = build_output_columns()
    out_df = pd.DataFrame(rows, columns=out_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote {len(out_df)} rows to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract lane-change metrics from 03_full-traces.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Path to 03_full-traces.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write 04_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()

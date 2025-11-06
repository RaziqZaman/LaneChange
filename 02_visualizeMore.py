#!/usr/bin/env python3
"""Generate per-timestep histograms contrasting human-only vs human-AV events.

For each lane-change trace in ``outputs/01_traces.csv`` this script isolates the
columns for timestep ``tXX`` (where ``XX`` is the lane-change switch index),
filters out subject-vehicle SDC events, and splits the rows into:

  * Human-only lane changes – all neighbour roles (LC/RC/LT/RT) are human drivers
    or not present.
  * Human-AV lane changes – exactly one neighbour role is the autonomous vehicle.

The script then creates overlaid histograms (red vs blue) comparing the two
groups for the following metrics:

  * Subject speed, longitudinal acceleration, lateral acceleration.
  * Neighbour longitudinal gap and relative speed for LC, RC, LT, RT.

Each histogram is saved to ``outputs/`` as ``02_more_<metric>.png``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("outputs/01_traces.csv")
OUTPUT_DIR = Path("outputs")

SUBJECT_METRICS = [
    "SV_speed",
    "SV_longitudinal_accel",
    "SV_lateral_accel",
]

NEIGHBOR_METRICS = [
    ("LC", "longitudinal_gap"),
    ("LC", "relative_speed"),
    ("RC", "longitudinal_gap"),
    ("RC", "relative_speed"),
    ("LT", "longitudinal_gap"),
    ("LT", "relative_speed"),
    ("RT", "longitudinal_gap"),
    ("RT", "relative_speed"),
]


def parse_flag(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        flag = int(float(raw))
    except ValueError:
        return None
    if flag not in (0, 1):
        return None
    return flag


def load_records(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            yield row


def build_column_name(timestep: int, prefix: str) -> str:
    return f"t{timestep:02d}_{prefix}"


def collect_values(
    row: Dict[str, str],
    column_name: str,
) -> Optional[float]:
    raw = row.get(column_name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def make_histogram(
    values_hh: List[float],
    values_hav: List[float],
    title: str,
    output_path: Path,
) -> None:
    if not values_hh and not values_hav:
        return

    combined = values_hh + values_hav
    if not combined:
        return

    min_val = min(combined)
    max_val = max(combined)
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5

    bins = np.linspace(min_val, max_val, 20)

    plt.figure(figsize=(7, 5))
    plt.hist(
        values_hh,
        bins=bins,
        color="#f1948a",
        alpha=0.6,
        label="Human-Human",
        density=True,
    )
    plt.hist(
        values_hav,
        bins=bins,
        color="#85c1e9",
        alpha=0.6,
        label="Human-AV",
        density=True,
    )
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Proportion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise per-timestep metrics for human-only vs human-AV lane changes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to 01_traces.csv.")
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to store the generated PNGs."
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    collected: Dict[str, Tuple[List[float], List[float]]] = {}
    for metric in SUBJECT_METRICS:
        collected[metric] = ([], [])
    for role, suffix in NEIGHBOR_METRICS:
        collected[f"{role}_{suffix}"] = ([], [])

    for row in load_records(args.input):
        sv_flag = parse_flag(row.get("sv_is_SDC"))
        if sv_flag != 0:
            continue

        lc_flag = parse_flag(row.get("lc_is_SDC"))
        rc_flag = parse_flag(row.get("rc_is_SDC"))
        lt_flag = parse_flag(row.get("lt_is_SDC"))
        rt_flag = parse_flag(row.get("rt_is_SDC"))

        flags = [flag for flag in (lc_flag, rc_flag, lt_flag, rt_flag) if flag is not None]
        if any(flag not in (0, 1) for flag in flags):
            continue
        if sum(flag == 1 for flag in flags) > 1:
            continue

        timestep_raw = row.get("lane_change_timestep")
        if timestep_raw is None or not timestep_raw.strip():
            continue
        try:
            timestep = int(float(timestep_raw))
        except ValueError:
            continue
        if timestep < 0 or timestep > 90:
            continue

        has_av_neighbor = any(flag == 1 for flag in flags)
        group_index = 1 if has_av_neighbor else 0

        # Subject metrics
        for metric in SUBJECT_METRICS:
            column = build_column_name(timestep, metric)
            value = collect_values(row, column)
            if value is not None:
                collected[metric][group_index].append(value)

        # Neighbour metrics
        for role, suffix in NEIGHBOR_METRICS:
            flag = {
                "LC": lc_flag,
                "RC": rc_flag,
                "LT": lt_flag,
                "RT": rt_flag,
            }[role]

            if flag is None or flag == 1:
                if has_av_neighbor and flag == 1:
                    column = build_column_name(timestep, f"{role}_{suffix}")
                    value = collect_values(row, column)
                    if value is not None:
                        collected[f"{role}_{suffix}"][group_index].append(value)
                elif not has_av_neighbor and flag is None:
                    column = build_column_name(timestep, f"{role}_{suffix}")
                    value = collect_values(row, column)
                    if value is not None:
                        collected[f"{role}_{suffix}"][group_index].append(value)
            else:
                column = build_column_name(timestep, f"{role}_{suffix}")
                value = collect_values(row, column)
                if value is not None:
                    collected[f"{role}_{suffix}"][group_index].append(value)

    for metric in SUBJECT_METRICS:
        hh_values, hav_values = collected[metric]
        output_path = args.output_dir / f"02_more_{metric}.png"
        make_histogram(hh_values, hav_values, f"{metric}", output_path)

    for key, (hh_values, hav_values) in collected.items():
        if key in SUBJECT_METRICS:
            continue
        if key in ("RT_longitudinal_gap", "RC_longitudinal_gap"):
            hh_values = [val for val in hh_values if val <= 0.0]
            hav_values = [val for val in hav_values if val <= 0.0]
        if key in ("LC_longitudinal_gap", "LT_longitudinal_gap"):
            hh_values = [val for val in hh_values if val >= 0.0]
            hav_values = [val for val in hav_values if val >= 0.0]
        output_path = args.output_dir / f"02_more_{key}.png"
        title = key.replace("_", " ")
        make_histogram(hh_values, hav_values, title, output_path)


if __name__ == "__main__":
    main()

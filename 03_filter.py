#!/usr/bin/env python3
"""Filter lane-change traces and annotate initiation/completion timesteps.

Reads ``outputs/01_traces.csv`` and writes ``outputs/03_full-traces.csv`` with:
    - ``scenario_id``
    - ``initiation_timestep`` (first >0.07 m/s^2 after 0.5 s near-zero before mid)
    - ``midpoint_timestep`` (renamed from ``lane_change_timestep``)
    - ``completion_timestep`` (last >0.07 m/s^2 before 0.5 s near-zero after mid)
    - all remaining original columns (dropping ``lane_change_timestamp``)

Only rows where both initiation and completion are found inside the trace are kept.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


INPUT_PATH = Path("outputs/01_traces.csv")
OUTPUT_PATH = Path("outputs/03_full-traces.csv")
DT = 0.1
WIN_SEC = 0.5
WIN_STEPS = max(1, int(round(WIN_SEC / DT)))  # 0.5 s window => 5 steps at 0.1 s
THRESH = 0.07
NUM_STEPS = 91  # t00..t90


def get_lateral_columns() -> List[str]:
    return [f"t{idx:02d}_SV_lateral_accel" for idx in range(NUM_STEPS)]


def find_markers(lat_acc: np.ndarray, mid_idx: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (initiation_idx, completion_idx) or (None, None) if not found."""
    n = len(lat_acc)
    if not (0 <= mid_idx < n):
        return None, None

    # Backward scan for near-zero window before midpoint
    init_idx: Optional[int] = None
    for end in range(min(mid_idx, n - 1), WIN_STEPS - 1, -1):
        start = end - WIN_STEPS + 1
        window = lat_acc[start : end + 1]
        if np.all(np.abs(window) <= THRESH):
            for pos in range(end + 1, mid_idx + 1):
                if abs(lat_acc[pos]) > THRESH:
                    init_idx = pos
                    break
            break

    # Forward scan for near-zero window after midpoint
    comp_idx: Optional[int] = None
    for start in range(mid_idx, n - WIN_STEPS + 1):
        end = start + WIN_STEPS - 1
        window = lat_acc[start : end + 1]
        if np.all(np.abs(window) <= THRESH):
            candidate = start - 1
            comp_idx = candidate if candidate >= 0 else None
            break

    return init_idx, comp_idx


def build_output_columns(orig_cols: List[str]) -> List[str]:
    rest = [
        c
        for c in orig_cols
        if c not in {"scenario_id", "lane_change_timestep", "lane_change_timestamp"}
    ]
    return [
        "scenario_id",
        "initiation_timestep",
        "midpoint_timestep",
        "completion_timestep",
        "duration",
        *rest,
    ]


def process(traces_path: Path, output_path: Path) -> None:
    lat_cols = get_lateral_columns()
    df = pd.read_csv(traces_path)
    if df.empty:
        raise SystemExit(f"No rows found in {traces_path}")

    records = []
    for _, row in df.iterrows():
        mid = row.get("lane_change_timestep")
        if pd.isna(mid):
            continue
        mid_idx = int(mid)
        lat_acc = row[lat_cols].to_numpy(dtype=float)
        lat_acc = np.where(np.isfinite(lat_acc), lat_acc, 0.0)

        init_idx, comp_idx = find_markers(lat_acc, mid_idx)
        if (
            init_idx is None
            or comp_idx is None
            or not (0 <= init_idx <= mid_idx <= comp_idx < NUM_STEPS)
        ):
            continue

        rec = row.to_dict()
        rec.pop("lane_change_timestamp", None)
        rec["midpoint_timestep"] = rec.pop("lane_change_timestep")
        rec["initiation_timestep"] = init_idx
        rec["completion_timestep"] = comp_idx
        rec["duration"] = (comp_idx - init_idx) / 10.0
        records.append(rec)

    if not records:
        raise SystemExit("No lane changes met the initiation/completion criteria.")

    out_cols = build_output_columns(list(df.columns))
    out_df = pd.DataFrame(records, columns=out_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    mean_duration = out_df["duration"].mean()
    print(f"Mean lane-change duration (s): {mean_duration:.3f}")
    print(f"Wrote {len(out_df)} filtered lane changes to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter lane changes and add initiation/completion timesteps.",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=INPUT_PATH,
        help="Path to 01_traces.csv input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path for filtered output CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(args.traces, args.output)


if __name__ == "__main__":
    main()

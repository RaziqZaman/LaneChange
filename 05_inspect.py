#!/usr/bin/env python3
"""Count lane-change rows by LC/RC/LT/RT state combinations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("outputs/04_metrics.csv")
OUTPUT_PATH = Path("outputs/05_inspection.csv")
STATE_COLS = ["LC_state", "RC_state", "LT_state", "RT_state"]


def process(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path, keep_default_na=False)
    if df.empty:
        raise SystemExit(f"No rows found in {input_path}")

    missing = [col for col in STATE_COLS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected state columns: {missing}")

    df[STATE_COLS] = df[STATE_COLS].fillna("")

    grouped = df.groupby(STATE_COLS, dropna=False).size().reset_index(name="count")
    grouped["has_AV"] = grouped[STATE_COLS].eq("AV").any(axis=1)
    grouped = grouped.sort_values(
        by=["has_AV", *STATE_COLS],
        ascending=[True, True, True, True, True],
    ).drop(columns="has_AV")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)
    print(f"Wrote {len(grouped)} combinations to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize counts by LC/RC/LT/RT state combinations.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PATH,
        help="Path to 04_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write 05_inspection.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(args.input, args.output)


if __name__ == "__main__":
    main()

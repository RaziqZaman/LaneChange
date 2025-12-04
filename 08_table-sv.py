#!/usr/bin/env python3
"""Build SV metric comparison tables (CSV + PNG) for each lane-change phase.

Reads ``outputs/06_analysis_stats_base.csv`` (unless overridden) and produces
four tables – initiation, midpoint, completion, and peak subject-vehicle (SV)
metrics – saved under ``outputs/tables/`` as both CSV and PNG.
Each table contains 30 rows (5 comparison pairs × 6 SV metrics) sorted by the
absolute value of Cliff's Delta.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("outputs/06_analysis_stats_base.csv")
OUTPUT_DIR = Path("outputs/tables")

STAGES: Dict[str, str] = {
    "init": "Initiation",
    "mid": "Midpoint",
    "comp": "Completion",
    "peak": "Peak",
}

EXPECT_COMPARISONS = 5
EXPECT_METRICS_PER_STAGE = 6


def fmt_ci(value: float, lower: float, upper: float, digits: int = 3) -> str:
    if any(np.isnan([value, lower, upper])):
        return ""
    return f"{value:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def fmt_p(value: float) -> str:
    if np.isnan(value):
        return ""
    return f"{value:.3e}"


def build_table_rows(stage_df: pd.DataFrame) -> Tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    row_colors: List[str] = []

    def color_for_delta(delta_abs: float) -> str:
        if delta_abs > 0.474:
            return "#f9d6d5"  # light red
        if delta_abs > 0.33:
            return "#fff3cd"  # light yellow
        if delta_abs >= 0.147:
            return "#d4efdf"  # light green
        return "#ffffff"  # white/default

    for _, row in stage_df.iterrows():
        rows.append(
            {
                "Metric": row["metric"],
                "Group A (n)": f"{row['group1']} ({int(row['n1'])})" if pd.notna(row.get("n1")) else row["group1"],
                "Group B (n)": f"{row['group2']} ({int(row['n2'])})" if pd.notna(row.get("n2")) else row["group2"],
                "Mean_A - Mean_B (95% CI)": fmt_ci(
                    row["diff_mean"], row["diff_mean_ci_lower"], row["diff_mean_ci_upper"]
                ),
                "StDev A - StDev B (95% CI)": fmt_ci(
                    row["diff_std"], row["diff_std_ci_lower"], row["diff_std_ci_upper"]
                ),
                "MWU p-value": fmt_p(row["mwu_p_value"]),
                "KS p-value": fmt_p(row["ks_p_value"]),
                "Cliff's Delta (95% CI)": fmt_ci(
                    row["cliffs_delta"], row["cliffs_delta_ci_lower"], row["cliffs_delta_ci_upper"]
                ),
            }
        )
        row_colors.append(color_for_delta(abs(row["cliffs_delta"])))
    return rows, row_colors


def render_png(table_df: pd.DataFrame, title: str, output_path: Path, row_colors: List[str]) -> None:
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.axis("off")
    ax.set_title(title, fontsize=14, pad=18, fontweight="bold")  # ~0.25" padding

    cell_colors = [[row_colors[i]] * table_df.shape[1] for i in range(table_df.shape[0])]

    # Set compact but readable column widths.
    default_width = 0.04  # compact columns to keep ~0.1" padding
    col_widths: List[float] = [default_width] * table_df.shape[1]
    if table_df.shape[1] == 8:
        col_widths = [0.09, 0.055, 0.055, 0.12, 0.12, 0.055, 0.055, 0.09]

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
        cellColours=cell_colors,
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.3)

    # Standard thin borders only.
    for cell in table.get_celld().values():
        cell.set_linewidth(0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.1)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def process_stage(df: pd.DataFrame, stage: str, pretty_name: str, output_dir: Path) -> None:
    prefix = f"SV_{stage}_"
    stage_df = df[df["metric"].str.startswith(prefix)].copy()
    if stage_df.empty:
        print(f"No rows found for stage '{stage}' ({pretty_name}); skipping.")
        return

    # Sort by |Cliff's Delta| descending.
    stage_df["abs_cliffs"] = stage_df["cliffs_delta"].abs()
    stage_df = stage_df.sort_values(by="abs_cliffs", ascending=False)

    # Expect exactly 5 comparisons × 6 metrics = 30 rows.
    if len(stage_df) != EXPECT_COMPARISONS * EXPECT_METRICS_PER_STAGE:
        print(
            f"Warning: stage '{stage}' has {len(stage_df)} rows; expected "
            f"{EXPECT_COMPARISONS * EXPECT_METRICS_PER_STAGE}."
        )

    rows, row_colors = build_table_rows(stage_df)
    table_df = pd.DataFrame(rows)

    csv_path = output_dir / f"08_sv_{stage}.csv"
    png_path = output_dir / f"08_sv_{stage}.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    table_df.to_csv(csv_path, index=False)
    render_png(table_df, f"SV {pretty_name} Metrics", png_path, row_colors)
    print(f"Wrote {len(table_df)} rows to {csv_path} and {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SV metric comparison tables (CSV + PNG) for each phase."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to 06_analysis_stats_base.csv.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to store generated tables (default: outputs/tables).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Unused; reserved for future reproducibility options.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    if df.empty:
        raise SystemExit(f"No rows found in {args.input}")

    for stage, pretty in STAGES.items():
        process_stage(df, stage, pretty, args.output_dir)


if __name__ == "__main__":
    main()

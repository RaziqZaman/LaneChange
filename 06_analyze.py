#!/usr/bin/env python3
"""Analyze lane-change metrics with AV/Human group comparisons.

This script reads outputs/04_metrics.csv and runs non-parametric tests
across AV-related cohorts:
    1) Any AV present in the lane-change vs no AVs present.
    2) Per-role (LC/RC/LT/RT) AV vs HD occupants.
    3) Proximity-stratified versions of the above, where AV rows are
       bucketed by initial longitudinal gap in fixed 15 m increments.

For every numeric metric, it produces Mann-Whitney U, Kolmogorov-Smirnov,
and Cliff's Delta results, together with distribution plots organized by
comparison type (base vs proximity) and metric category:
    - base/proximity/SV/{duration,long_speed,long_acc,long_jerk,lat_speed,lat_acc,lat_jerk}
    - base/proximity/{LC,RC,LT,RT}/{long_gap,rel_speed,ttc,headway}

Outputs (prefix 06_analysis_*):
    outputs/06_analysis_stats.csv   – tabular test results.
    outputs/06_analysis_overview.txt – group counts and bucket details.
    outputs/06_analysis_plots/06_analysis_<comparison>_<metric>.png – histograms.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

# Point matplotlib cache at a writable location inside the workspace.
_cache_root = Path("outputs/.cache")
_cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))
(_cache_root / "fontconfig").mkdir(parents=True, exist_ok=True)

_mpl_dir = Path("outputs/mpl_config")
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_dir))
_mpl_dir.mkdir(parents=True, exist_ok=True)

import matplotlib

# Ensure headless plotting for batch runs.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


INPUT_PATH = Path("outputs/04_metrics.csv")
STATS_PATH = Path("outputs/06_analysis_stats.csv")
OVERVIEW_PATH = Path("outputs/06_analysis_overview.txt")
PLOTS_DIR = Path("outputs/06_analysis_plots")
PROXIMITY_BUCKET_WIDTH = 15.0  # meters
PROXIMITY_OVERLAY_BUCKET_WIDTH = 45.0  # meters for overlay plots

STATE_COLS = ["LC_state", "RC_state", "LT_state", "RT_state"]
ROLES = ["LC", "RC", "LT", "RT"]
ROLE_INIT_GAP = {
    "LC": "LC_init_long_gap",
    "RC": "RC_init_long_gap",
    "LT": "LT_init_long_gap",
    "RT": "RT_init_long_gap",
}


@dataclass
class Comparison:
    name: str
    group1_label: str
    group2_label: str
    mask1: pd.Series
    mask2: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run AV vs HD statistical tests on 04_metrics.csv",
    )
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Path to 04_metrics.csv")
    parser.add_argument("--stats", type=Path, default=STATS_PATH, help="Output CSV for test results")
    parser.add_argument("--overview", type=Path, default=OVERVIEW_PATH, help="Output TXT for summary info")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PLOTS_DIR,
        help="Directory for per-metric distribution plots",
    )
    return parser.parse_args()


def cliffs_delta(sample1: Sequence[float], sample2: Sequence[float]) -> float:
    """Compute Cliff's delta efficiently using binary searches."""
    n1 = len(sample1)
    n2 = len(sample2)
    if n1 == 0 or n2 == 0:
        return math.nan

    b_sorted = np.sort(np.asarray(sample2))
    more = 0
    less = 0
    for val in sample1:
        # Count how many b values are less/greater than val (ties ignored).
        more += np.searchsorted(b_sorted, val, side="left")
        less += n2 - np.searchsorted(b_sorted, val, side="right")
    return (more - less) / float(n1 * n2)


def metric_columns(df: pd.DataFrame) -> List[str]:
    exclude = set(STATE_COLS + ["scenario_id"])
    cols: List[str] = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def metric_stage(metric: str) -> str:
    lower = metric.lower()
    if "init_" in lower:
        return "init"
    if "mid_" in lower:
        return "mid"
    if "comp_" in lower:
        return "comp"
    if "peak_" in lower or lower.startswith("min_") or lower.startswith("max_"):
        return "peak"
    return "other"


def metric_subpath(metric: str) -> Path:
    """Determine subfolder path for a metric."""
    lower = metric.lower()

    # Subject vehicle metrics.
    if metric.startswith("SV_") or "duration" in lower:
        sv_root = Path("SV")
        if "duration" in lower:
            return Path(metric_stage(metric)) / sv_root / "duration"
        if "long_speed" in lower:
            return Path(metric_stage(metric)) / sv_root / "long_speed"
        if "long_acc" in lower:
            return Path(metric_stage(metric)) / sv_root / "long_acc"
        if "long_jerk" in lower:
            return Path(metric_stage(metric)) / sv_root / "long_jerk"
        if "lat_speed" in lower:
            return Path(metric_stage(metric)) / sv_root / "lat_speed"
        if "lat_acc" in lower:
            return Path(metric_stage(metric)) / sv_root / "lat_acc"
        if "lat_jerk" in lower:
            return Path(metric_stage(metric)) / sv_root / "lat_jerk"
        return Path(metric_stage(metric)) / sv_root / "other"

    # Neighbor roles.
    for role in ROLES:
        if metric.startswith(f"{role}_"):
            role_root = Path(role)
            if "long_gap" in lower:
                return Path(metric_stage(metric)) / role_root / "long_gap"
            if "rel_speed" in lower:
                return Path(metric_stage(metric)) / role_root / "rel_speed"
            if "ttc" in lower:
                return Path(metric_stage(metric)) / role_root / "ttc"
            if "headway" in lower:
                return Path(metric_stage(metric)) / role_root / "headway"
            return Path(metric_stage(metric)) / role_root / "other"

    return Path(metric_stage(metric)) / "other"


def clean_numeric(series: pd.Series) -> pd.Series:
    """Coerce to numeric, drop inf/nan."""
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.dropna()


def describe(series: pd.Series) -> Dict[str, float]:
    """Return summary statistics for a numeric series."""
    if series.empty:
        return {
            "mean": math.nan,
            "min": math.nan,
            "q1": math.nan,
            "median": math.nan,
            "q3": math.nan,
            "std": math.nan,
            "max": math.nan,
        }
    values = series.to_numpy()
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "q1": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "q3": float(np.percentile(values, 75)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "max": float(np.max(values)),
    }


def _stat_values(arr: np.ndarray) -> Dict[str, float]:
    n = len(arr)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if n > 1 else 0.0,
        "min": float(np.min(arr)),
        "q1": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "q3": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
    }


def bootstrap_intervals(
    series1: pd.Series,
    series2: pd.Series,
    n_boot: int = 5,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Bootstrap CIs for Cliff's delta and stat differences."""
    if len(series1) == 0 or len(series2) == 0:
        keys = [
            "cliffs_delta_ci_lower",
            "cliffs_delta_ci_upper",
            "diff_mean_ci_lower",
            "diff_mean_ci_upper",
            "diff_std_ci_lower",
            "diff_std_ci_upper",
            "diff_min_ci_lower",
            "diff_min_ci_upper",
            "diff_q1_ci_lower",
            "diff_q1_ci_upper",
            "diff_median_ci_lower",
            "diff_median_ci_upper",
            "diff_q3_ci_lower",
            "diff_q3_ci_upper",
            "diff_max_ci_lower",
            "diff_max_ci_upper",
        ]
        return {k: math.nan for k in keys}

    a = series1.to_numpy()
    b = series2.to_numpy()
    rng = np.random.default_rng()

    deltas: List[float] = []
    diff_mean: List[float] = []
    diff_std: List[float] = []
    diff_min: List[float] = []
    diff_q1: List[float] = []
    diff_median: List[float] = []
    diff_q3: List[float] = []
    diff_max: List[float] = []

    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        stats_a = _stat_values(sa)
        stats_b = _stat_values(sb)

        deltas.append(cliffs_delta(sa, sb))
        diff_mean.append(stats_a["mean"] - stats_b["mean"])
        diff_std.append(stats_a["std"] - stats_b["std"])
        diff_min.append(stats_a["min"] - stats_b["min"])
        diff_q1.append(stats_a["q1"] - stats_b["q1"])
        diff_median.append(stats_a["median"] - stats_b["median"])
        diff_q3.append(stats_a["q3"] - stats_b["q3"])
        diff_max.append(stats_a["max"] - stats_b["max"])

    def ci(values: List[float]) -> tuple[float, float]:
        lower, upper = np.percentile(values, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        return float(lower), float(upper)

    delta_ci = ci(deltas)
    return {
        "cliffs_delta_ci_lower": delta_ci[0],
        "cliffs_delta_ci_upper": delta_ci[1],
        "diff_mean_ci_lower": ci(diff_mean)[0],
        "diff_mean_ci_upper": ci(diff_mean)[1],
        "diff_std_ci_lower": ci(diff_std)[0],
        "diff_std_ci_upper": ci(diff_std)[1],
        "diff_min_ci_lower": ci(diff_min)[0],
        "diff_min_ci_upper": ci(diff_min)[1],
        "diff_q1_ci_lower": ci(diff_q1)[0],
        "diff_q1_ci_upper": ci(diff_q1)[1],
        "diff_median_ci_lower": ci(diff_median)[0],
        "diff_median_ci_upper": ci(diff_median)[1],
        "diff_q3_ci_lower": ci(diff_q3)[0],
        "diff_q3_ci_upper": ci(diff_q3)[1],
        "diff_max_ci_lower": ci(diff_max)[0],
        "diff_max_ci_upper": ci(diff_max)[1],
    }


def make_buckets(series: pd.Series, width: float) -> pd.Series:
    """Assign bucket labels with step `width`; returns object series."""
    if not np.isfinite(width) or width <= 0:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    clean = series.dropna()
    if clean.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    min_val = clean.min()
    max_val = clean.max()
    start = math.floor(min_val / width) * width
    end = math.ceil(max_val / width) * width
    bins = np.arange(start, end + width, width)
    if len(bins) < 2:
        bins = np.array([start, start + width])
    labels = [f"{bins[i]:.2f}_to_{bins[i+1]:.2f}" for i in range(len(bins) - 1)]
    cut = pd.cut(series, bins=bins, labels=labels, include_lowest=True)
    return cut.astype("object")


def compute_plot_bins(df: pd.DataFrame, metrics: List[str]) -> Dict[str, np.ndarray]:
    bins_map: Dict[str, np.ndarray] = {}
    for metric in metrics:
        values = clean_numeric(df[metric])
        if values.empty:
            continue
        try:
            bins = np.histogram_bin_edges(values, bins=30)
            bins_map[metric] = bins
        except Exception:
            continue
    return bins_map


def compute_ymax_map(
    df: pd.DataFrame,
    metrics: List[str],
    comparisons: List[Comparison],
    bins_map: Dict[str, np.ndarray],
) -> Dict[str, float]:
    y_max: Dict[str, float] = {m: 0.0 for m in metrics}

    def update(metric: str, series: pd.Series) -> None:
        if series.empty:
            return
        bins = bins_map.get(metric)
        if bins is None:
            return
        hist, _ = np.histogram(series, bins=bins, density=True)
        if len(hist):
            y_max[metric] = max(y_max[metric], float(hist.max()))

    # Base and proximity comparisons (group1/group2).
    for comp in comparisons:
        df1 = df.loc[comp.mask1]
        df2 = df.loc[comp.mask2]
        for metric in metrics:
            update(metric, clean_numeric(df1[metric]))
            update(metric, clean_numeric(df2[metric]))

    # Overlay bucket groupings using 45m buckets.
    has_av = df[STATE_COLS].eq("AV").any(axis=1)
    any_overlay = make_buckets(df["closest_av_init_gap"], PROXIMITY_OVERLAY_BUCKET_WIDTH)
    role_overlay: Dict[str, pd.Series] = {}
    for role in ROLES:
        state_col = f"{role}_state"
        gap_col = ROLE_INIT_GAP[role]
        mask = df[state_col] == "AV"
        bucket_series = pd.Series(pd.NA, index=df.index, dtype="object")
        if mask.any():
            bucket_series.loc[mask] = make_buckets(df.loc[mask, gap_col], PROXIMITY_OVERLAY_BUCKET_WIDTH)
        role_overlay[role] = bucket_series

    for metric in metrics:
        update(metric, clean_numeric(df.loc[~has_av, metric]))
        for bucket in sorted(any_overlay.dropna().unique()):
            update(metric, clean_numeric(df.loc[any_overlay == bucket, metric]))

    for role in ROLES:
        state_col = f"{role}_state"
        bucket_series = role_overlay[role]
        for metric in metrics:
            update(metric, clean_numeric(df.loc[df[state_col] == "HD", metric]))
            for bucket in sorted(bucket_series.dropna().unique()):
                update(metric, clean_numeric(df.loc[bucket_series == bucket, metric]))

    return y_max


def compute_closest_av_gap(row: pd.Series) -> float:
    """Return the smallest init gap for any AV in the row, or NaN."""
    gaps: List[float] = []
    for role in ROLES:
        if row[f"{role}_state"] == "AV":
            gap_col = ROLE_INIT_GAP.get(role)
            if gap_col:
                value = row.get(gap_col)
                if value is None or value == "" or pd.isna(value):
                    continue
                try:
                    gaps.append(float(value))
                except (TypeError, ValueError):
                    continue
    return min(gaps) if gaps else math.nan


def plot_metric(
    metric: str,
    series_map: Dict[str, Sequence[float]],
    output_path: Path,
    bins: Optional[np.ndarray] = None,
    y_max: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in series_map.items():
        if len(values) == 0:
            continue
        ax.hist(
            values,
            bins=bins if bins is not None else 30,
            alpha=0.5,
            density=True,
            label=f"{label} (n={len(values)})",
        )
    ax.set_title(metric)
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")
    if bins is not None and len(bins) >= 2:
        ax.set_xlim(bins[0], bins[-1])
    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0, y_max * 1.05)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_tests_for_pair(
    comparison: Comparison,
    metric: str,
    series1: pd.Series,
    series2: pd.Series,
    results: List[Dict[str, object]],
    plots_dir: Path,
    bins_map: Dict[str, np.ndarray],
    y_max_map: Dict[str, float],
) -> None:
    n1 = len(series1)
    n2 = len(series2)
    desc1 = describe(series1)
    desc2 = describe(series2)
    boot = bootstrap_intervals(series1, series2)
    base_fields = {
        "comparison": comparison.name,
        "group1": comparison.group1_label,
        "group2": comparison.group2_label,
        "metric": metric,
        "n1": n1,
        "n2": n2,
        "g1_mean": desc1["mean"],
        "g1_std": desc1["std"],
        "g1_min": desc1["min"],
        "g1_q1": desc1["q1"],
        "g1_median": desc1["median"],
        "g1_q3": desc1["q3"],
        "g1_max": desc1["max"],
        "g2_mean": desc2["mean"],
        "g2_std": desc2["std"],
        "g2_min": desc2["min"],
        "g2_q1": desc2["q1"],
        "g2_median": desc2["median"],
        "g2_q3": desc2["q3"],
        "g2_max": desc2["max"],
        "diff_mean": desc1["mean"] - desc2["mean"],
        "diff_std": desc1["std"] - desc2["std"],
        "diff_min": desc1["min"] - desc2["min"],
        "diff_q1": desc1["q1"] - desc2["q1"],
        "diff_median": desc1["median"] - desc2["median"],
        "diff_q3": desc1["q3"] - desc2["q3"],
        "diff_max": desc1["max"] - desc2["max"],
        **boot,
    }

    if n1 == 0 or n2 == 0:
        results.append(
            {
                **base_fields,
                "test": "skipped",
                "statistic": math.nan,
                "p_value": math.nan,
                "cliffs_delta": math.nan,
                "notes": "insufficient data",
            }
        )
        return

    try:
        mw = mannwhitneyu(series1, series2, alternative="two-sided")
        ks = ks_2samp(series1, series2, alternative="two-sided", mode="asymp")
        delta = cliffs_delta(series1, series2)
        mw_norm = float(mw.statistic) / float(n1 * n2) if n1 and n2 else math.nan
        results.extend(
            [
                {
                    **base_fields,
                    "test": "mannwhitney_u",
                    "statistic": mw_norm,
                    "p_value": float(mw.pvalue),
                    "cliffs_delta": delta,
                    "notes": "",
                },
                {
                    **base_fields,
                    "test": "ks_2samp",
                    "statistic": float(ks.statistic),
                    "p_value": float(ks.pvalue),
                    "cliffs_delta": delta,
                    "notes": "",
                },
            ]
        )
    except Exception as exc:  # pragma: no cover - defensive
        results.append(
            {
                **base_fields,
                "test": "error",
                "statistic": math.nan,
                "p_value": math.nan,
                "cliffs_delta": math.nan,
                "notes": f"failed: {exc}",
            }
        )
        return

    plot_metric(
        metric,
        {
            comparison.group1_label: series1,
            comparison.group2_label: series2,
        },
        plots_dir
        / ("proximity" if "bucket" in comparison.name else "base")
        / metric_subpath(metric)
        / f"06_analysis_{sanitize_name(comparison.name)}_{sanitize_name(metric)}.png",
        bins=bins_map.get(metric),
        y_max=y_max_map.get(metric),
    )


def plot_proximity_overlays(
    df: pd.DataFrame,
    metrics: List[str],
    plots_dir: Path,
    bins_map: Dict[str, np.ndarray],
    y_max_map: Dict[str, float],
) -> None:
    """Plot all proximity buckets for each metric on shared histograms."""
    has_av = df[STATE_COLS].eq("AV").any(axis=1)
    overlay_width = PROXIMITY_OVERLAY_BUCKET_WIDTH  # meters for overlay plots only

    # Build overlay bucket labels (does not affect statistical comparisons).
    any_overlay = make_buckets(df["closest_av_init_gap"], overlay_width)
    role_overlay: Dict[str, pd.Series] = {}
    for role in ROLES:
        state_col = f"{role}_state"
        gap_col = ROLE_INIT_GAP[role]
        mask = df[state_col] == "AV"
        bucket_series = pd.Series(pd.NA, index=df.index, dtype="object")
        if mask.any():
            bucket_series.loc[mask] = make_buckets(df.loc[mask, gap_col], overlay_width)
        role_overlay[role] = bucket_series

    def plot_overlay(series_map: Dict[str, pd.Series], name: str, metric: str) -> None:
        non_empty = {k: v for k, v in series_map.items() if len(v) > 0}
        if len(non_empty) < 2:
            return
        output_path = (
            plots_dir
            / "proximity"
            / metric_subpath(metric)
            / f"06_analysis_{sanitize_name(name)}_{sanitize_name(metric)}.png"
        )
        plot_metric(
            metric,
            non_empty,
            output_path,
            bins=bins_map.get(metric),
            y_max=y_max_map.get(metric),
        )

    # Any-AV buckets vs no-AV baseline.
    for metric in metrics:
        series_map: Dict[str, pd.Series] = {}
        no_av_series = clean_numeric(df.loc[~has_av, metric])
        if len(no_av_series) > 0:
            series_map["no_AV"] = no_av_series
        for bucket in sorted(any_overlay.dropna().unique()):
            series_map[f"bucket_{bucket}"] = clean_numeric(df.loc[any_overlay == bucket, metric])
        plot_overlay(series_map, "any_av_buckets", metric)

    # Role-specific buckets vs HD baseline.
    for role in ROLES:
        state_col = f"{role}_state"
        bucket_series = role_overlay[role]
        for metric in metrics:
            series_map: Dict[str, pd.Series] = {}
            hd_series = clean_numeric(df.loc[df[state_col] == "HD", metric])
            if len(hd_series) > 0:
                series_map[f"{role}_HD"] = hd_series
            for bucket in sorted(bucket_series.dropna().unique()):
                series_map[f"{role}_bucket_{bucket}"] = clean_numeric(df.loc[bucket_series == bucket, metric])
            plot_overlay(series_map, f"{role}_av_buckets", metric)


def build_base_comparisons(df: pd.DataFrame) -> List[Comparison]:
    has_av = df[STATE_COLS].eq("AV").any(axis=1)
    comps = [
        Comparison(
            name="any_av_vs_no_av",
            group1_label="any_AV",
            group2_label="no_AV",
            mask1=has_av,
            mask2=~has_av,
        )
    ]
    for role in ROLES:
        state_col = f"{role}_state"
        comps.append(
            Comparison(
                name=f"{role}_AV_vs_{role}_HD",
                group1_label=f"{role}_AV",
                group2_label=f"{role}_HD",
                mask1=df[state_col] == "AV",
                mask2=df[state_col] == "HD",
            )
        )
    return comps


def build_proximity_comparisons(df: pd.DataFrame) -> List[Comparison]:
    comps: List[Comparison] = []

    # Any-AV proximity buckets vs no-AV.
    bucket_col = "any_av_bucket"
    has_av = df[STATE_COLS].eq("AV").any(axis=1)
    for bucket in sorted(df.loc[has_av, bucket_col].dropna().unique()):
        label = str(bucket)
        comps.append(
            Comparison(
                name=f"any_av_bucket_{label}_vs_no_av",
                group1_label=f"any_AV_{label}",
                group2_label="no_AV",
                mask1=(df[bucket_col] == bucket),
                mask2=~has_av,
            )
        )

    # Role-specific AV buckets vs HD.
    for role in ROLES:
        bucket_col = f"{role}_av_bucket"
        state_col = f"{role}_state"
        hd_mask = df[state_col] == "HD"
        for bucket in sorted(df.loc[df[state_col] == "AV", bucket_col].dropna().unique()):
            label = str(bucket)
            comps.append(
                Comparison(
                    name=f"{role}_av_bucket_{label}_vs_{role}_HD",
                    group1_label=f"{role}_AV_{label}",
                    group2_label=f"{role}_HD",
                    mask1=(df[bucket_col] == bucket),
                    mask2=hd_mask,
                )
            )
    return comps


def add_bucket_columns(df: pd.DataFrame) -> Dict[str, float]:
    """Add bucket columns; returns dict of bucket widths used."""
    bucket_widths: Dict[str, float] = {}

    width = PROXIMITY_BUCKET_WIDTH

    # Closest AV gap for any-AV comparison.
    if tqdm is not None:
        try:
            tqdm.pandas(desc="closest_av_gap")
            closest_raw = df.progress_apply(compute_closest_av_gap, axis=1)
        except Exception:
            closest_raw = df.apply(compute_closest_av_gap, axis=1)
    else:
        closest_raw = df.apply(compute_closest_av_gap, axis=1)

    df["closest_av_init_gap"] = closest_raw
    closest_series = clean_numeric(closest_raw)
    bucket_widths["any_av"] = width
    df["any_av_bucket"] = make_buckets(closest_series, width)

    # Role-specific buckets (only set where the role is an AV).
    for role in ROLES:
        state_col = f"{role}_state"
        gap_col = ROLE_INIT_GAP[role]
        mask = df[state_col] == "AV"
        gap_values = clean_numeric(df.loc[mask, gap_col])
        bucket_widths[role] = width
        bucket_series = pd.Series(pd.NA, index=df.index, dtype="object")
        if mask.any():
            bucket_series.loc[gap_values.index] = make_buckets(gap_values, width)
        df[f"{role}_av_bucket"] = bucket_series
    return bucket_widths


def write_overview(
    output_path: Path,
    df: pd.DataFrame,
    bucket_widths: Dict[str, float],
    comparisons: Iterable[Comparison],
) -> None:
    lines: List[str] = []
    lines.append("Group sizes:")
    for comp in comparisons:
        lines.append(
            f"  {comp.name}: {comp.group1_label}={int(comp.mask1.sum())}, "
            f"{comp.group2_label}={int(comp.mask2.sum())}"
        )
    lines.append("")
    lines.append("Bucket widths (fixed meters):")
    for key, width in bucket_widths.items():
        lines.append(f"  {key}: {width if np.isfinite(width) else 'nan'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, keep_default_na=False)
    df.replace("", np.nan, inplace=True)
    for col in df.columns:
        if col not in STATE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if df.empty:
        raise SystemExit(f"No rows found in {args.input}")

    # Normalize state columns for grouping.
    missing_states = [col for col in STATE_COLS if col not in df.columns]
    if missing_states:
        raise SystemExit(f"Missing state columns: {missing_states}")
    df[STATE_COLS] = df[STATE_COLS].fillna("")

    metrics = metric_columns(df)
    if not metrics:
        raise SystemExit("No numeric metrics found to analyze.")

    bucket_widths = {k: float(v) if np.isfinite(v) else math.nan for k, v in add_bucket_columns(df).items()}

    base_comparisons = build_base_comparisons(df)
    proximity_comparisons = build_proximity_comparisons(df)

    plot_bins = compute_plot_bins(df, metrics)
    y_max_map = compute_ymax_map(df, metrics, [*base_comparisons, *proximity_comparisons], plot_bins)

    all_comparisons = [*base_comparisons, *proximity_comparisons]
    total_steps = len(all_comparisons) * len(metrics)
    bar = tqdm(total=total_steps, desc="Comparisons x metrics", leave=False) if tqdm else None

    results: List[Dict[str, object]] = []
    for comp in all_comparisons:
        df1 = df.loc[comp.mask1]
        df2 = df.loc[comp.mask2]
        for metric in metrics:
            series1 = clean_numeric(df1[metric])
            series2 = clean_numeric(df2[metric])
            run_tests_for_pair(
                comp,
                metric,
                series1,
                series2,
                results,
                args.plots_dir,
                plot_bins,
                y_max_map,
            )
            if bar is not None:
                bar.update(1)

    if bar is not None:
        bar.close()

    stats_df = pd.DataFrame(results)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.stats, index=False)

    # Subset outputs
    subset_outputs = {
        "base": stats_df[~stats_df["comparison"].str.contains("bucket", na=False)],
        "buckets": stats_df[stats_df["comparison"].str.contains("bucket", na=False)],
        "mannwhitney": stats_df[stats_df["test"] == "mannwhitney_u"],
        "ks": stats_df[stats_df["test"] == "ks_2samp"],
        "sv_metrics": stats_df[stats_df["metric"].str.startswith("SV_", na=False)],
        "role_metrics": stats_df[
            stats_df["metric"].str.startswith(("LC_", "RC_", "LT_", "RT_"), na=False)
        ],
    }
    for name, sdf in subset_outputs.items():
        out_path = args.stats.parent / f"06_analysis_stats_{name}.csv"
        sdf.to_csv(out_path, index=False)

    plot_proximity_overlays(df, metrics, args.plots_dir, plot_bins, y_max_map)

    write_overview(
        args.overview,
        df,
        bucket_widths,
        [*base_comparisons, *proximity_comparisons],
    )
    print(f"Wrote {len(stats_df)} test rows to {args.stats}")
    print(f"Plots saved under {args.plots_dir}")
    print(f"Overview written to {args.overview}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualise a random Waymo Motion scenario with agent trajectories and map data.

The script selects a random TFRecord from ``data/`` (unless overridden) and
draws:

* ``outputs/motion.png`` – vehicle trajectories across all 91 timesteps with
  current speeds and heading angles annotated.
* ``outputs/roads.png`` – lane centre polylines (solid blue) and lane boundary
  polylines coloured/styled by their road-graph type.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
except Exception as exc:  # pragma: no cover - visual output requires matplotlib
    raise SystemExit("matplotlib is required to run this script.") from exc

import numpy as np

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required to parse TFRecord files. "
        "Install it in your virtual environment before running this script."
    ) from exc


PAST_STEPS = 10
CURRENT_STEPS = 1
FUTURE_STEPS = 80
TOTAL_STEPS = PAST_STEPS + CURRENT_STEPS + FUTURE_STEPS

LANE_CENTER_TYPES = {1, 2}
BOUNDARY_STYLES: Dict[int, Tuple[str, str, float]] = {
    6: ("#bbbbbb", (0, (6, 6)), 1.2),   # Broken single white
    7: ("#bbbbbb", "-", 1.5),           # Solid single white
    8: ("#bbbbbb", "-", 2.2),           # Solid double white
    9: ("#f7d16e", (0, (6, 6)), 1.2),   # Broken single yellow
    10: ("#f7d16e", (0, (6, 6)), 2.0),  # Broken double yellow
    11: ("#f7d16e", "-", 1.5),          # Solid single yellow
    12: ("#f7d16e", "-", 2.2),          # Solid double yellow
    13: ("#f7d16e", "-.", 2.0),         # Passing double yellow
}


def get_float_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[float]:
    feat = feature_map.get(key)
    return feat.float_list.value if feat is not None else ()


def get_int_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[int]:
    feat = feature_map.get(key)
    return feat.int64_list.value if feat is not None else ()


def get_bytes_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[bytes]:
    feat = feature_map.get(key)
    return feat.bytes_list.value if feat is not None else ()


def slice_segment(values: Sequence[float], track_idx: int, steps: int) -> List[Optional[float]]:
    if not values:
        return [None] * steps
    base = track_idx * steps
    out: List[Optional[float]] = []
    for offset in range(steps):
        idx = base + offset
        out.append(float(values[idx]) if idx < len(values) else None)
    return out


def is_padding(x: Optional[float], y: Optional[float], z: Optional[float]) -> bool:
    if x is None or y is None or z is None:
        return True
    return (
        math.isclose(x, -1.0, abs_tol=1e-3)
        and math.isclose(y, -1.0, abs_tol=1e-3)
        and math.isclose(z, -1.0, abs_tol=1e-3)
    )


def decode_track_ids(feature_map: Dict[str, tf.train.Feature], num_tracks: int) -> List[str]:
    byte_ids = get_bytes_list(feature_map, "state/id")
    float_ids = get_float_list(feature_map, "state/id")
    int_ids = get_int_list(feature_map, "state/id")

    decoded: List[str] = []
    for idx in range(num_tracks):
        value: Optional[str] = None

        if idx < len(byte_ids):
            raw = byte_ids[idx]
            try:
                value = raw.decode("utf-8")
            except Exception:
                try:
                    tensor = tf.io.parse_tensor(raw, out_type=tf.string)  # type: ignore[arg-type]
                    arr = tensor.numpy()
                    if arr.size:
                        value = arr.flatten()[0].decode("utf-8")
                except Exception:
                    value = None

        if value is None and idx < len(float_ids):
            fid = float_ids[idx]
            if math.isfinite(fid):
                value = str(int(round(fid)))

        if value is None and idx < len(int_ids):
            value = str(int(int_ids[idx]))

        decoded.append(value or f"track_{idx}")
    return decoded


def select_random_example(data_dir: Path, rng: random.Random) -> Tuple[Path, int, tf.train.Example]:
    files = sorted(data_dir.glob("*.tfrecord*"))
    if not files:
        raise SystemExit(f"No TFRecord files found under {data_dir}")
    file_path = rng.choice(files)

    dataset = tf.data.TFRecordDataset(str(file_path))
    chosen_raw: Optional[bytes] = None
    chosen_index = -1
    for index, raw in enumerate(dataset):
        if rng.random() < 1.0 / (index + 1):
            chosen_raw = raw.numpy()
            chosen_index = index
    if chosen_raw is None:
        raise SystemExit(f"Failed to sample record from {file_path}")

    example = tf.train.Example()
    example.ParseFromString(chosen_raw)
    return file_path, chosen_index, example


def collect_vehicle_traces(feature_map: Dict[str, tf.train.Feature]) -> List[Dict[str, object]]:
    state_type = get_float_list(feature_map, "state/type")
    num_tracks = len(state_type)
    if num_tracks == 0:
        return []

    track_ids = decode_track_ids(feature_map, num_tracks)
    is_sdc_list = get_float_list(feature_map, "state/is_sdc")

    vehicle_indices = [idx for idx, value in enumerate(state_type) if int(round(value)) == 1]
    if not vehicle_indices:
        return []

    past_x = get_float_list(feature_map, "state/past/x")
    past_y = get_float_list(feature_map, "state/past/y")
    past_z = get_float_list(feature_map, "state/past/z")
    past_valid = get_float_list(feature_map, "state/past/valid")

    current_x = get_float_list(feature_map, "state/current/x")
    current_y = get_float_list(feature_map, "state/current/y")
    current_z = get_float_list(feature_map, "state/current/z")
    current_valid = get_float_list(feature_map, "state/current/valid")

    future_x = get_float_list(feature_map, "state/future/x")
    future_y = get_float_list(feature_map, "state/future/y")
    future_z = get_float_list(feature_map, "state/future/z")
    future_valid = get_float_list(feature_map, "state/future/valid")

    current_speed = get_float_list(feature_map, "state/current/speed")
    current_heading = get_float_list(feature_map, "state/current/bbox_yaw")

    traces: List[Dict[str, object]] = []

    for track_idx in vehicle_indices:
        positions: List[Tuple[float, float]] = []

        segments = [
            ("state/past", past_x, past_y, past_z, past_valid, PAST_STEPS),
            ("state/current", current_x, current_y, current_z, current_valid, CURRENT_STEPS),
            ("state/future", future_x, future_y, future_z, future_valid, FUTURE_STEPS),
        ]

        offset = 0
        current_position: Optional[Tuple[float, float]] = None

        for _, xs, ys, zs, valids, steps in segments:
            slice_x = slice_segment(xs, track_idx, steps)
            slice_y = slice_segment(ys, track_idx, steps)
            slice_z = slice_segment(zs, track_idx, steps)
            slice_valid = slice_segment(valids, track_idx, steps) if valids else [None] * steps

            for step in range(steps):
                x = slice_x[step]
                y = slice_y[step]
                z = slice_z[step]
                valid = slice_valid[step]

                if x is None or y is None or z is None:
                    continue
                if valid is not None and valid <= 0.5:
                    continue
                if is_padding(x, y, z):
                    continue

                positions.append((x, y))
                if offset == PAST_STEPS and step == 0:
                    current_position = (x, y)
            offset += steps

        speed = float(slice_segment(current_speed, track_idx, 1)[0] or 0.0)
        heading = float(slice_segment(current_heading, track_idx, 1)[0] or 0.0)
        is_sdc = int(round(is_sdc_list[track_idx])) if track_idx < len(is_sdc_list) else 0

        traces.append(
            {
                "track_id": track_ids[track_idx],
                "positions": positions,
                "current_pos": current_position,
                "speed": speed,
                "heading": heading,
                "is_sdc": bool(is_sdc),
            }
        )

    return traces


def collect_road_points(feature_map: Dict[str, tf.train.Feature]) -> Tuple[
    Dict[int, List[Tuple[int, float, float]]],
    Dict[Tuple[int, int], List[Tuple[int, float, float]]],
]:
    types = get_int_list(feature_map, "roadgraph_samples/type")
    ids = get_int_list(feature_map, "roadgraph_samples/id")
    xyz = get_float_list(feature_map, "roadgraph_samples/xyz")

    lane_centers: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)
    boundaries: Dict[Tuple[int, int], List[Tuple[int, float, float]]] = defaultdict(list)

    for sample_idx, lane_type in enumerate(types):
        if sample_idx >= len(ids):
            continue
        base = 3 * sample_idx
        if base + 2 >= len(xyz):
            continue
        lane_id = int(ids[sample_idx])
        x = float(xyz[base])
        y = float(xyz[base + 1])

        if lane_type in LANE_CENTER_TYPES:
            lane_centers[lane_id].append((sample_idx, x, y))
        elif lane_type in BOUNDARY_STYLES:
            boundaries[(lane_type, lane_id)].append((sample_idx, x, y))

    return lane_centers, boundaries


def compute_extent(
    traces: List[Dict[str, object]],
    lane_centers: Dict[int, List[Tuple[int, float, float]]],
    boundaries: Dict[Tuple[int, int], List[Tuple[int, float, float]]],
) -> Optional[Tuple[float, float, float, float]]:
    xs: List[float] = []
    ys: List[float] = []

    for trace in traces:
        positions = trace.get("positions")
        if not positions:
            continue
        xs.extend(pt[0] for pt in positions)
        ys.extend(pt[1] for pt in positions)

    for points in lane_centers.values():
        xs.extend(pt[1] for pt in points)
        ys.extend(pt[2] for pt in points)

    for points in boundaries.values():
        xs.extend(pt[1] for pt in points)
        ys.extend(pt[2] for pt in points)

    if not xs or not ys:
        return None

    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    span_x = xmax - xmin
    span_y = ymax - ymin
    padding = max(span_x, span_y) * 0.05
    if padding <= 0.0:
        padding = 5.0
    return (xmin - padding, xmax + padding, ymin - padding, ymax + padding)


def plot_motion(
    traces: List[Dict[str, object]],
    output_path: Path,
    extent: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    if not traces:
        print("No vehicle traces available for motion plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    av_color = "#1f77b4"
    default_cycle = plt.rcParams.get("axes.prop_cycle")
    base_colors = default_cycle.by_key().get("color", []) if default_cycle else []
    human_palette = [c for c in base_colors if c.lower() != av_color]
    if not human_palette:
        av_rgba = mcolors.to_rgba(av_color)
        human_palette = [
            plt.cm.tab20(i / 20.0) for i in range(20)
            if not np.allclose(plt.cm.tab20(i / 20.0), av_rgba)
        ]
    if not human_palette:
        human_palette = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    human_color_cycle = cycle(human_palette)

    for trace in traces:
        positions = trace["positions"]  # type: ignore[assignment]
        if not positions:
            continue
        xs = [pt[0] for pt in positions]
        ys = [pt[1] for pt in positions]
        if trace.get("is_sdc"):
            color = av_color
        else:
            color = next(human_color_cycle)
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.8)
        ax.scatter(xs, ys, s=5, color=color, alpha=0.8)

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Vehicle Trajectories")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_roads(
    lane_centers: Dict[int, List[Tuple[int, float, float]]],
    boundaries: Dict[Tuple[int, int], List[Tuple[int, float, float]]],
    output_path: Path,
    extent: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    if not lane_centers and not boundaries:
        print("No road data available for roads plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    # Lane centres in solid blue.
    for points in lane_centers.values():
        if len(points) < 2:
            continue
        sorted_points = sorted(points, key=lambda item: item[0])
        xs = [pt[1] for pt in sorted_points]
        ys = [pt[2] for pt in sorted_points]
        ax.plot(xs, ys, color="#1f77b4", linewidth=1.5, linestyle="-", alpha=0.9)

    # Lane boundaries coloured by type.
    for (lane_type, _), points in boundaries.items():
        if len(points) < 2:
            continue
        style = BOUNDARY_STYLES.get(lane_type, ("#bbbbbb", "-", 1.0))
        sorted_points = sorted(points, key=lambda item: item[0])
        xs = [pt[1] for pt in sorted_points]
        ys = [pt[2] for pt in sorted_points]
        ax.plot(xs, ys, color=style[0], linestyle=style[1], linewidth=style[2], alpha=0.9)

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Lane Centres and Lane Boundaries")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise a random scenario from the Waymo Motion dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing TFRecord files.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for generated images.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    file_path, record_index, example = select_random_example(args.data_dir, rng)
    feature_map = example.features.feature

    scenario_bytes = get_bytes_list(feature_map, "scenario/id")
    scenario_id = scenario_bytes[0].decode("utf-8", errors="ignore") if scenario_bytes else ""
    print(f"Selected scenario: {scenario_id or '[unknown id]'} from {file_path.name} (record {record_index})")

    traces = collect_vehicle_traces(feature_map)
    lane_centers, boundaries = collect_road_points(feature_map)
    extent = compute_extent(traces, lane_centers, boundaries)

    motion_path = args.output_dir / "motion.png"
    roads_path = args.output_dir / "roads.png"

    plot_motion(traces, motion_path, extent)
    plot_roads(lane_centers, boundaries, roads_path, extent)
    print(f"Saved visualisations to {motion_path} and {roads_path}")


if __name__ == "__main__":
    main()

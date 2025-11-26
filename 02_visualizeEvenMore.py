#!/usr/bin/env python3
"""Visualise a random Waymo Motion scenario with agent trajectories and map data.

The script selects a random TFRecord from ``data/`` (unless overridden) and
draws:

* ``outputs/motion.png`` – vehicle trajectories coloured by agent role.
* ``outputs/roads.png`` – lane centre polylines (solid blue) and lane boundary
  polylines coloured/styled by their road-graph type.
* ``outputs/closestLane.png`` – vehicle trajectories with their nearest lane
  centre lines.
* ``outputs/laneChange.png`` – same as ``closestLane`` but highlighting vehicles
  that change lanes.
* ``outputs/velocityComponents.png`` – trajectories with projected velocity
  components (parallel/perpendicular to the nearest lane centre).
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
    from matplotlib import patches as mpatches
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

LANE_CELL_SIZE = 10.0
LANE_SEARCH_RADIUS = 6


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

    past_vx = get_float_list(feature_map, "state/past/velocity_x")
    past_vy = get_float_list(feature_map, "state/past/velocity_y")
    current_vx = get_float_list(feature_map, "state/current/velocity_x")
    current_vy = get_float_list(feature_map, "state/current/velocity_y")
    future_vx = get_float_list(feature_map, "state/future/velocity_x")
    future_vy = get_float_list(feature_map, "state/future/velocity_y")

    current_speed = get_float_list(feature_map, "state/current/speed")
    current_heading = get_float_list(feature_map, "state/current/bbox_yaw")

    traces: List[Dict[str, object]] = []

    for track_idx in vehicle_indices:
        positions: List[Tuple[float, float]] = []

        segments = [
            (past_x, past_y, past_z, past_valid, past_vx, past_vy, PAST_STEPS),
            (current_x, current_y, current_z, current_valid, current_vx, current_vy, CURRENT_STEPS),
            (future_x, future_y, future_z, future_valid, future_vx, future_vy, FUTURE_STEPS),
        ]

        offset = 0
        current_position: Optional[Tuple[float, float]] = None
        current_index: Optional[int] = None
        velocities: List[Tuple[float, float]] = []

        for xs, ys, zs, valids, vxs, vys, steps in segments:
            slice_x = slice_segment(xs, track_idx, steps)
            slice_y = slice_segment(ys, track_idx, steps)
            slice_z = slice_segment(zs, track_idx, steps)
            slice_valid = slice_segment(valids, track_idx, steps) if valids else [None] * steps
            slice_vx = slice_segment(vxs, track_idx, steps) if vxs else [None] * steps
            slice_vy = slice_segment(vys, track_idx, steps) if vys else [None] * steps

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
                vx_val = slice_vx[step] if step < len(slice_vx) else None
                vy_val = slice_vy[step] if step < len(slice_vy) else None
                vx = float(vx_val) if vx_val is not None else 0.0
                vy = float(vy_val) if vy_val is not None else 0.0
                velocities.append((vx, vy))
                if offset == PAST_STEPS and step == 0:
                    current_position = (x, y)
                    current_index = len(positions) - 1
            offset += steps

        speed = float(slice_segment(current_speed, track_idx, 1)[0] or 0.0)
        heading = float(slice_segment(current_heading, track_idx, 1)[0] or 0.0)
        is_sdc = int(round(is_sdc_list[track_idx])) if track_idx < len(is_sdc_list) else 0

        traces.append(
            {
                "track_id": track_ids[track_idx],
                "positions": positions,
                "current_pos": current_position,
                "current_index": current_index,
                "velocities": velocities,
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


def build_lane_polylines(lane_centers: Dict[int, List[Tuple[int, float, float]]]) -> Dict[int, List[Tuple[float, float]]]:
    polylines: Dict[int, List[Tuple[float, float]]] = {}
    for lane_id, points in lane_centers.items():
        if len(points) < 2:
            continue
        sorted_points = sorted(points, key=lambda item: item[0])
        polyline = [(float(x), float(y)) for _, x, y in sorted_points]
        polylines[lane_id] = polyline
    return polylines


def compute_lane_directions(polylines: Dict[int, List[Tuple[float, float]]]) -> Dict[int, Tuple[float, float]]:
    dir_map: Dict[int, Tuple[float, float]] = {}
    for lane_id, points in polylines.items():
        if len(points) < 2:
            continue
        sx = sy = 0.0
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            dx = x1 - x0
            dy = y1 - y0
            norm = math.hypot(dx, dy)
            if norm < 1e-6:
                continue
            sx += dx / norm
            sy += dy / norm
        norm_total = math.hypot(sx, sy)
        if norm_total < 1e-6:
            continue
        dir_map[lane_id] = (sx / norm_total, sy / norm_total)
    return dir_map


def build_lane_grid(polylines: Dict[int, List[Tuple[float, float]]]) -> Dict[Tuple[int, int], List[Tuple[float, float, int]]]:
    grid: Dict[Tuple[int, int], List[Tuple[float, float, int]]] = {}
    for lane_id, points in polylines.items():
        for x, y in points:
            cell = (int(math.floor(x / LANE_CELL_SIZE)), int(math.floor(y / LANE_CELL_SIZE)))
            grid.setdefault(cell, []).append((x, y, lane_id))
    return grid


def nearest_lane_id(grid: Dict[Tuple[int, int], List[Tuple[float, float, int]]], x: float, y: float) -> Optional[int]:
    if not grid:
        return None
    cell_x = int(math.floor(x / LANE_CELL_SIZE))
    cell_y = int(math.floor(y / LANE_CELL_SIZE))
    best_lane: Optional[int] = None
    best_dist2 = float("inf")
    for radius in range(LANE_SEARCH_RADIUS + 1):
        found = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                bucket = (cell_x + dx, cell_y + dy)
                for px, py, lane_id in grid.get(bucket, ()):
                    found = True
                    dist2 = (px - x) ** 2 + (py - y) ** 2
                    if dist2 < best_dist2:
                        best_dist2 = dist2
                        best_lane = lane_id
        if found and best_lane is not None:
            break
    return best_lane


def detect_lane_changes(lane_ids: List[Optional[int]]) -> List[int]:
    prev: Optional[int] = None
    change_indices: List[int] = []
    for idx, lane_id in enumerate(lane_ids):
        if lane_id is None:
            continue
        if prev is None:
            prev = lane_id
            continue
        if lane_id != prev:
            change_indices.append(idx)
            prev = lane_id
    return change_indices


def assign_lane_information(
    traces: List[Dict[str, object]],
    lane_grid: Dict[Tuple[int, int], List[Tuple[float, float, int]]],
) -> None:
    for trace in traces:
        positions: List[Tuple[float, float]] = trace.get("positions", [])  # type: ignore[assignment]
        lane_ids: List[Optional[int]] = []
        for x, y in positions:
            lane_ids.append(nearest_lane_id(lane_grid, x, y))
        trace["lane_ids"] = lane_ids
        change_indices = detect_lane_changes(lane_ids)
        trace["lane_change"] = bool(change_indices)
        trace["lane_change_indices"] = change_indices
        current_index: Optional[int] = trace.get("current_index")  # type: ignore[assignment]
        current_lane: Optional[int] = None
        if current_index is not None and 0 <= current_index < len(lane_ids):
            current_lane = lane_ids[current_index]
        if current_lane is None:
            for lane_id in lane_ids:
                if lane_id is not None:
                    current_lane = lane_id
                    break
        trace["current_lane_id"] = current_lane

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


def assign_trace_colors(traces: List[Dict[str, object]]) -> List[str]:
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

    colors: List[str] = []
    for trace in traces:
        if trace.get("is_sdc"):
            colors.append(av_color)
        else:
            colors.append(mcolors.to_hex(next(human_color_cycle)))
    return colors


def plot_motion(
    traces: List[Dict[str, object]],
    colors: List[str],
    output_path: Path,
    extent: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    if not traces:
        print("No vehicle traces available for motion plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))

    for trace, color in zip(traces, colors):
        positions = trace["positions"]  # type: ignore[assignment]
        if not positions:
            continue
        xs = [pt[0] for pt in positions]
        ys = [pt[1] for pt in positions]
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


def draw_traces_with_lanes(
    ax: plt.Axes,
    traces: List[Dict[str, object]],
    colors: List[str],
    lane_polylines: Dict[int, List[Tuple[float, float]]],
    extent: Optional[Tuple[float, float, float, float]] = None,
    highlight_changes: bool = False,
    emphasize_lane: bool = False,
    draw_boxes: bool = False,
    change_markers: bool = False,
) -> None:
    lane_color_map: Dict[int, str] = {}

    for trace, color in zip(traces, colors):
        lane_id = trace.get("current_lane_id")
        if lane_id is not None and lane_id in lane_polylines:
            lane_color = lane_color_map.setdefault(lane_id, color)
            polyline = lane_polylines[lane_id]
            xs_lane = [pt[0] for pt in polyline]
            ys_lane = [pt[1] for pt in polyline]
            lw = 2.5 if emphasize_lane else 1.2
            ax.plot(xs_lane, ys_lane, linestyle="--", linewidth=lw, color=lane_color, alpha=0.8 if emphasize_lane else 0.6)
            if emphasize_lane:
                ax.scatter(xs_lane, ys_lane, s=18, color=lane_color, alpha=0.4, linewidths=0)

        positions = trace.get("positions")  # type: ignore[assignment]
        if not positions:
            continue
        xs = [pt[0] for pt in positions]
        ys = [pt[1] for pt in positions]
        lw = 2.4 if highlight_changes and trace.get("lane_change") else 1.6
        alpha = 0.95 if highlight_changes and trace.get("lane_change") else 0.8
        ax.plot(xs, ys, "-", color=color, linewidth=lw, alpha=alpha)
        ax.scatter(xs, ys, s=5, color=color, alpha=0.6)

        if draw_boxes:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            width = max(xmax - xmin, 1.0)
            height = max(ymax - ymin, 1.0)
            rect = mpatches.Rectangle(
                (xmin - 0.5, ymin - 0.5),
                width + 1.0,
                height + 1.0,
                linewidth=1.2,
                edgecolor=color,
                facecolor="none",
                alpha=0.75,
                linestyle="--",
            )
            ax.add_patch(rect)

        if change_markers and trace.get("lane_change_indices"):
            change_indices: List[int] = trace["lane_change_indices"]  # type: ignore[assignment]
            points = [positions[idx] for idx in change_indices if 0 <= idx < len(positions)]
            if points:
                cx = [pt[0] for pt in points]
                cy = [pt[1] for pt in points]
                ax.scatter(cx, cy, s=90, color=color, edgecolor="black", linewidths=1.2, alpha=0.95, marker="o")

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


def plot_closest_lane(
    traces: List[Dict[str, object]],
    colors: List[str],
    lane_polylines: Dict[int, List[Tuple[float, float]]],
    output_path: Path,
    extent: Optional[Tuple[float, float, float, float]],
) -> None:
    if not traces:
        print("No vehicle traces available for closest-lane plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_traces_with_lanes(
        ax,
        traces,
        colors,
        lane_polylines,
        extent,
        highlight_changes=False,
        emphasize_lane=True,
        draw_boxes=True,
    )
    ax.set_title("Trajectories with Nearest Lane Centres")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lane_change_highlight(
    traces: List[Dict[str, object]],
    colors: List[str],
    lane_polylines: Dict[int, List[Tuple[float, float]]],
    output_path: Path,
    extent: Optional[Tuple[float, float, float, float]],
) -> None:
    if not traces:
        print("No vehicle traces available for lane-change plot.")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_traces_with_lanes(
        ax,
        traces,
        colors,
        lane_polylines,
        extent,
        highlight_changes=True,
        emphasize_lane=True,
        change_markers=True,
    )
    ax.set_title("Lane-Change Trajectories Highlighted")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_velocity_components(
    traces: List[Dict[str, object]],
    colors: List[str],
    lane_polylines: Dict[int, List[Tuple[float, float]]],
    lane_dir_map: Dict[int, Tuple[float, float]],
    output_path: Path,
    extent: Optional[Tuple[float, float, float, float]],
) -> None:
    if not traces:
        print("No vehicle traces available for velocity-component plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_traces_with_lanes(
        ax,
        traces,
        colors,
        lane_polylines,
        extent,
        highlight_changes=False,
        emphasize_lane=True,
    )

    for trace, color in zip(traces, colors):
        current_index: Optional[int] = trace.get("current_index")  # type: ignore[assignment]
        lane_id: Optional[int] = trace.get("current_lane_id")  # type: ignore[assignment]
        positions: List[Tuple[float, float]] = trace.get("positions", [])  # type: ignore[assignment]
        velocities: List[Tuple[float, float]] = trace.get("velocities", [])  # type: ignore[assignment]

        if (
            current_index is None
            or lane_id is None
            or current_index >= len(positions)
            or current_index >= len(velocities)
        ):
            continue

        lane_dir = lane_dir_map.get(lane_id)
        if lane_dir is None:
            continue

        pos_x, pos_y = positions[current_index]
        vx, vy = velocities[current_index]
        dir_x, dir_y = lane_dir
        perp_x, perp_y = -dir_y, dir_x

        parallel_mag = vx * dir_x + vy * dir_y
        perpendicular_mag = vx * perp_x + vy * perp_y

        parallel_vec = (parallel_mag * dir_x, parallel_mag * dir_y)
        perpendicular_vec = (perpendicular_mag * perp_x, perpendicular_mag * perp_y)

        scale = 12.0
        if abs(parallel_mag) > 1e-3:
            ax.arrow(
                pos_x,
                pos_y,
                parallel_vec[0] * scale,
                parallel_vec[1] * scale,
                color=color,
                width=0.45,
                length_includes_head=True,
                head_width=1.6,
                alpha=0.9,
            )
        if abs(perpendicular_mag) > 1e-3:
            ax.arrow(
                pos_x,
                pos_y,
                perpendicular_vec[0] * scale,
                perpendicular_vec[1] * scale,
                color="#2c3e50",
                width=0.38,
                length_includes_head=True,
                head_width=1.4,
                alpha=0.85,
            )

    if extent is not None:
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Velocity Components Parallel vs Perpendicular to Lane")
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
    lane_polylines = build_lane_polylines(lane_centers)
    lane_dir_map = compute_lane_directions(lane_polylines)
    lane_grid = build_lane_grid(lane_polylines)
    assign_lane_information(traces, lane_grid)
    colors = assign_trace_colors(traces)
    extent = compute_extent(traces, lane_centers, boundaries)

    motion_path = args.output_dir / "motion.png"
    roads_path = args.output_dir / "roads.png"
    closest_lane_path = args.output_dir / "closestLane.png"
    lane_change_path = args.output_dir / "laneChange.png"
    velocity_components_path = args.output_dir / "velocityComponents.png"

    plot_motion(traces, colors, motion_path, extent)
    plot_roads(lane_centers, boundaries, roads_path, extent)
    plot_closest_lane(traces, colors, lane_polylines, closest_lane_path, extent)
    plot_lane_change_highlight(traces, colors, lane_polylines, lane_change_path, extent)
    plot_velocity_components(traces, colors, lane_polylines, lane_dir_map, velocity_components_path, extent)
    print(
        "Saved visualisations to "
        f"{motion_path}, {roads_path}, {closest_lane_path}, {lane_change_path}, and {velocity_components_path}"
    )


if __name__ == "__main__":
    main()

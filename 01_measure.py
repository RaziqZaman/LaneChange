#!/usr/bin/env python3
"""Measure lane-change interactions with detailed per-timestep metrics.

This script scans Waymo Open Motion TFRecord scenarios, identifies every lane
change performed by vehicle-type agents, and records contextual information
about surrounding vehicles together with detailed time-series measurements.

Outputs:
    outputs/01_events.csv  – summary rows per lane-change event.
    outputs/01_traces.csv  – per-event traces with per-timestep metrics.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "TensorFlow is required to parse TFRecord files. "
        "Install it in your virtual environment before running this script."
    ) from exc

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# Waymo Motion constants.
PAST_STEPS = 10
CURRENT_STEPS = 1
FUTURE_STEPS = 80
TOTAL_STEPS = PAST_STEPS + CURRENT_STEPS + FUTURE_STEPS

LANE_CENTER_TYPES = {1, 2}
ROAD_LINE_TYPES = {6, 7, 8, 9, 10, 11, 12, 13}

LANE_CELL_SIZE = 10.0
LANE_SEARCH_RADIUS = 6
LATERAL_DISTANCE_THRESHOLD = 3.5
PARALLEL_DOT_THRESHOLD = math.cos(math.radians(20))
DEFAULT_DT_SECONDS = 0.1
PRE_WINDOW_STEPS = max(1, int(round(0.5 / DEFAULT_DT_SECONDS)))
LANE_CENTER_SEPARATION_THRESHOLD = 0.75

STEP_LABELS = [f"t{idx:02d}" for idx in range(TOTAL_STEPS)]

OUTPUT_DIR = Path("outputs")
EVENTS_CSV = OUTPUT_DIR / "01_events.csv"
TRACES_CSV = OUTPUT_DIR / "01_traces.csv"


# ---------------------------------------------------------------------------
# Small helpers for TF feature extraction

def get_float_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[float]:
    feat = feature_map.get(key)
    return feat.float_list.value if feat is not None else ()


def get_int_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[int]:
    feat = feature_map.get(key)
    return feat.int64_list.value if feat is not None else ()


def get_bytes_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[bytes]:
    feat = feature_map.get(key)
    return feat.bytes_list.value if feat is not None else ()


def slice_segment(values: Sequence[Any], track_idx: int, steps: int) -> List[Any]:
    if not values:
        return [None] * steps
    base = track_idx * steps
    out: List[Any] = []
    for offset in range(steps):
        idx = base + offset
        out.append(values[idx] if idx < len(values) else None)
    return out


def decode_track_ids(feature_map: Dict[str, tf.train.Feature], num_tracks: int) -> List[str]:
    """Best-effort decoding of track identifiers."""
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

        decoded.append(value or "")
    return decoded


def is_padding(x: Optional[float], y: Optional[float], z: Optional[float]) -> bool:
    if x is None or y is None or z is None:
        return True
    return (
        math.isclose(x, -1.0, abs_tol=1e-3)
        and math.isclose(y, -1.0, abs_tol=1e-3)
        and math.isclose(z, -1.0, abs_tol=1e-3)
    )


@dataclass
class TrackSeries:
    track_id: str
    is_sdc: int
    agent_type: int
    positions: List[Optional[Tuple[float, float, float]]]
    timestamps: List[Optional[int]]
    speeds: List[Optional[float]]
    velocities: List[Tuple[Optional[float], Optional[float]]]
    valid: List[bool]
    lane_ids: List[Optional[int]]


def assemble_track_series(feature_map: Dict[str, tf.train.Feature], track_idx: int, num_tracks: int) -> TrackSeries:
    x_series: List[Optional[float]] = [None] * TOTAL_STEPS
    y_series: List[Optional[float]] = [None] * TOTAL_STEPS
    z_series: List[Optional[float]] = [None] * TOTAL_STEPS
    timestamps: List[Optional[int]] = [None] * TOTAL_STEPS
    speeds: List[Optional[float]] = [None] * TOTAL_STEPS
    velocities: List[Tuple[Optional[float], Optional[float]]] = [(None, None)] * TOTAL_STEPS
    valid_flags: List[bool] = [False] * TOTAL_STEPS

    segments = [
        ("state/past", PAST_STEPS, 0),
        ("state/current", CURRENT_STEPS, PAST_STEPS),
        ("state/future", FUTURE_STEPS, PAST_STEPS + CURRENT_STEPS),
    ]

    for prefix, steps, offset in segments:
        xs = slice_segment(get_float_list(feature_map, f"{prefix}/x"), track_idx, steps)
        ys = slice_segment(get_float_list(feature_map, f"{prefix}/y"), track_idx, steps)
        zs = slice_segment(get_float_list(feature_map, f"{prefix}/z"), track_idx, steps)
        ts_int = slice_segment(get_int_list(feature_map, f"{prefix}/timestamp_micros"), track_idx, steps)
        ts_float = slice_segment(get_float_list(feature_map, f"{prefix}/timestamp_micros"), track_idx, steps)
        speed_seg = slice_segment(get_float_list(feature_map, f"{prefix}/speed"), track_idx, steps)
        vx_seg = slice_segment(get_float_list(feature_map, f"{prefix}/velocity_x"), track_idx, steps)
        vy_seg = slice_segment(get_float_list(feature_map, f"{prefix}/velocity_y"), track_idx, steps)
        valid_seg = slice_segment(get_float_list(feature_map, f"{prefix}/valid"), track_idx, steps)

        for step in range(steps):
            idx = offset + step
            x_val = xs[step]
            y_val = ys[step]
            z_val = zs[step]

            valid = False
            if valid_seg and valid_seg[step] is not None:
                valid = bool(valid_seg[step] > 0.5)
            else:
                valid = not is_padding(x_val, y_val, z_val)

            if valid:
                x_series[idx] = float(x_val) if x_val is not None else None
                y_series[idx] = float(y_val) if y_val is not None else None
                z_series[idx] = float(z_val) if z_val is not None else None
            else:
                x_series[idx] = y_series[idx] = z_series[idx] = None

            if ts_int[step] is not None:
                timestamps[idx] = int(ts_int[step])
            elif ts_float[step] is not None:
                timestamps[idx] = int(ts_float[step])

            speeds[idx] = float(speed_seg[step]) if speed_seg[step] is not None else None

            vx_val = float(vx_seg[step]) if vx_seg[step] is not None else None
            vy_val = float(vy_seg[step]) if vy_seg[step] is not None else None
            velocities[idx] = (vx_val, vy_val)

            valid_flags[idx] = valid

    positions: List[Optional[Tuple[float, float, float]]] = []
    for x_val, y_val, z_val, valid in zip(x_series, y_series, z_series, valid_flags):
        if valid and x_val is not None and y_val is not None:
            positions.append((x_val, y_val, z_val if z_val is not None else 0.0))
        else:
            positions.append(None)

    track_ids = get_float_list(feature_map, "state/id")
    is_sdc_list = get_int_list(feature_map, "state/is_sdc")
    type_list = get_float_list(feature_map, "state/type")

    track_id_str = ""
    if track_idx < len(track_ids):
        track_id_str = str(int(round(track_ids[track_idx])))
    is_sdc_val = int(is_sdc_list[track_idx]) if track_idx < len(is_sdc_list) else 0
    agent_type = int(type_list[track_idx]) if track_idx < len(type_list) else 0

    return TrackSeries(
        track_id=track_id_str,
        is_sdc=is_sdc_val,
        agent_type=agent_type,
        positions=positions,
        timestamps=timestamps,
        speeds=speeds,
        velocities=velocities,
        valid=valid_flags,
        lane_ids=[None] * TOTAL_STEPS,
    )


# ---------------------------------------------------------------------------
# Map helpers

def collect_lane_points(feature_map: Dict[str, tf.train.Feature]) -> Tuple[
    Dict[Tuple[int, int], List[Tuple[float, float, float, int]]],
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[float, float]],
]:
    types = get_int_list(feature_map, "roadgraph_samples/type")
    ids = get_int_list(feature_map, "roadgraph_samples/id")
    xyz = get_float_list(feature_map, "roadgraph_samples/xyz")
    directions = get_float_list(feature_map, "roadgraph_samples/dir")

    lane_points: List[Tuple[float, float, float, int]] = []
    dir_map: Dict[int, List[Tuple[float, float]]] = {}

    center_sums: Dict[int, Tuple[float, float, int]] = {}

    for idx, t in enumerate(types):
        if t not in LANE_CENTER_TYPES:
            continue
        base_xyz = 3 * idx
        base_dir = 3 * idx
        if base_xyz + 2 >= len(xyz) or idx >= len(ids):
            continue
        x = float(xyz[base_xyz])
        y = float(xyz[base_xyz + 1])
        z = float(xyz[base_xyz + 2])
        lane_id = int(ids[idx])
        lane_points.append((x, y, z, lane_id))

        if base_dir + 1 < len(directions):
            dx = float(directions[base_dir])
            dy = float(directions[base_dir + 1])
            norm = math.hypot(dx, dy)
            if norm > 1e-6:
                dir_map.setdefault(lane_id, []).append((dx / norm, dy / norm))

        sum_x, sum_y, count = center_sums.get(lane_id, (0.0, 0.0, 0))
        center_sums[lane_id] = (sum_x + x, sum_y + y, count + 1)

    lane_dir_avg: Dict[int, Tuple[float, float]] = {}
    for lane_id, vectors in dir_map.items():
        sx = sum(v[0] for v in vectors)
        sy = sum(v[1] for v in vectors)
        norm = math.hypot(sx, sy)
        if norm > 1e-6:
            lane_dir_avg[lane_id] = (sx / norm, sy / norm)

    lane_grid: Dict[Tuple[int, int], List[Tuple[float, float, float, int]]] = {}
    for x, y, z, lane_id in lane_points:
        cell = (int(math.floor(x / LANE_CELL_SIZE)), int(math.floor(y / LANE_CELL_SIZE)))
        lane_grid.setdefault(cell, []).append((x, y, z, lane_id))

    lane_centers: Dict[int, Tuple[float, float]] = {}
    for lane_id, (sx, sy, count) in center_sums.items():
        if count > 0:
            lane_centers[lane_id] = (sx / count, sy / count)

    return lane_grid, lane_dir_avg, lane_centers


def build_road_line_segments(feature_map: Dict[str, tf.train.Feature]) -> List[Tuple[float, float, float, float]]:
    types = get_int_list(feature_map, "roadgraph_samples/type")
    ids = get_int_list(feature_map, "roadgraph_samples/id")
    xyz = get_float_list(feature_map, "roadgraph_samples/xyz")

    point_map: Dict[int, List[Tuple[int, float, float]]] = {}
    for idx, lane_type in enumerate(types):
        if lane_type not in ROAD_LINE_TYPES:
            continue
        if idx >= len(ids):
            continue
        base = 3 * idx
        if base + 1 >= len(xyz):
            continue
        lane_id = int(ids[idx])
        x = float(xyz[base])
        y = float(xyz[base + 1])
        point_map.setdefault(lane_id, []).append((idx, x, y))

    segments: List[Tuple[float, float, float, float]] = []
    for lane_id, points in point_map.items():
        if len(points) < 2:
            continue
        points.sort(key=lambda item: item[0])
        for i in range(len(points) - 1):
            _, x0, y0 = points[i]
            _, x1, y1 = points[i + 1]
            if x0 == x1 and y0 == y1:
                continue
            segments.append((x0, y0, x1, y1))
    return segments


def nearest_lane_id(
    grid: Dict[Tuple[int, int], List[Tuple[float, float, float, int]]],
    x: float,
    y: float,
    z: float,
) -> Optional[int]:
    if not grid:
        return None

    cell_x = int(math.floor(x / LANE_CELL_SIZE))
    cell_y = int(math.floor(y / LANE_CELL_SIZE))
    best_dist2 = float("inf")
    best_lane: Optional[int] = None

    for radius in range(LANE_SEARCH_RADIUS + 1):
        found = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                bucket = (cell_x + dx, cell_y + dy)
                for px, py, pz, lane_id in grid.get(bucket, ()):
                    found = True
                    dist2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                    if dist2 < best_dist2:
                        best_dist2 = dist2
                        best_lane = lane_id
        if found and best_lane is not None:
            break
    return best_lane


def _orientation(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _on_segment(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> bool:
    return min(ax, cx) - 1e-6 <= bx <= max(ax, cx) + 1e-6 and min(ay, cy) - 1e-6 <= by <= max(ay, cy) + 1e-6


def segments_intersect(p0: Tuple[float, float], p1: Tuple[float, float], q0: Tuple[float, float], q1: Tuple[float, float]) -> bool:
    ax, ay = p0
    bx, by = p1
    cx, cy = q0
    dx, dy = q1

    o1 = _orientation(ax, ay, bx, by, cx, cy)
    o2 = _orientation(ax, ay, bx, by, dx, dy)
    o3 = _orientation(cx, cy, dx, dy, ax, ay)
    o4 = _orientation(cx, cy, dx, dy, bx, by)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if abs(o1) < 1e-6 and _on_segment(ax, ay, cx, cy, bx, by):
        return True
    if abs(o2) < 1e-6 and _on_segment(ax, ay, dx, dy, bx, by):
        return True
    if abs(o3) < 1e-6 and _on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if abs(o4) < 1e-6 and _on_segment(cx, cy, bx, by, dx, dy):
        return True
    return False


def crosses_road_line_between(
    positions: List[Optional[Tuple[float, float]]],
    start_idx: int,
    end_idx: int,
    road_line_segments: List[Tuple[float, float, float, float]],
) -> bool:
    if not road_line_segments:
        return False
    lower = max(0, start_idx)
    upper = min(len(positions) - 1, end_idx)
    if upper <= lower:
        return False
    for idx in range(lower, upper):
        p0 = positions[idx]
        p1 = positions[idx + 1]
        if p0 is None or p1 is None:
            continue
        x0, y0 = p0
        x1, y1 = p1
        if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
            continue
        for q0x, q0y, q1x, q1y in road_line_segments:
            if segments_intersect((x0, y0), (x1, y1), (q0x, q0y), (q1x, q1y)):
                return True
    return False


def lanes_are_parallel(lane_before: Optional[int], lane_after: Optional[int], lane_dir_map: Dict[int, Tuple[float, float]]) -> bool:
    if lane_before is None or lane_after is None:
        return False
    dir_before = lane_dir_map.get(int(lane_before))
    dir_after = lane_dir_map.get(int(lane_after))
    if dir_before is None or dir_after is None:
        return False
    dot = abs(dir_before[0] * dir_after[0] + dir_before[1] * dir_after[1])
    return dot >= PARALLEL_DOT_THRESHOLD


def extract_lane_change_events(
    lane_ids: List[Optional[int]],
    positions: List[Optional[Tuple[float, float]]],
    road_line_segments: List[Tuple[float, float, float, float]],
    lane_dir_map: Dict[int, Tuple[float, float]],
) -> List[Tuple[int, int, int]]:
    events: List[Tuple[int, int, int]] = []
    current_lane: Optional[int] = None
    last_idx: Optional[int] = None

    for idx, lane in enumerate(lane_ids):
        if lane is None:
            continue
        if current_lane is None:
            current_lane = lane
            last_idx = idx
            continue
        if lane == current_lane:
            last_idx = idx
            continue

        start_idx = last_idx if last_idx is not None else idx - 1
        if start_idx is None or start_idx < 0:
            current_lane = lane
            last_idx = idx
            continue

        if not crosses_road_line_between(positions, start_idx, idx, road_line_segments):
            current_lane = lane
            last_idx = idx
            continue

        if not lanes_are_parallel(current_lane, lane, lane_dir_map):
            current_lane = lane
            last_idx = idx
            continue

        events.append((int(current_lane), int(lane), idx))
        current_lane = lane
        last_idx = idx

    return events


def compute_lane_dirs_for_steps(lane_ids: List[Optional[int]], lane_dir_map: Dict[int, Tuple[float, float]]) -> List[Tuple[float, float]]:
    dirs: List[Tuple[float, float]] = []
    prev_dir = (1.0, 0.0)
    for lane_id in lane_ids:
        vector = lane_dir_map.get(int(lane_id)) if lane_id is not None else None
        if vector is not None:
            prev_dir = vector
        dirs.append(prev_dir)
    return dirs


def compute_subject_metrics(
    track: TrackSeries,
    lane_dirs: List[Tuple[float, float]],
) -> Dict[str, List[Optional[float]]]:
    long_vel: List[Optional[float]] = [None] * TOTAL_STEPS
    lat_vel: List[Optional[float]] = [None] * TOTAL_STEPS
    long_accel: List[Optional[float]] = [None] * TOTAL_STEPS
    lat_accel: List[Optional[float]] = [None] * TOTAL_STEPS

    for idx in range(TOTAL_STEPS):
        vx, vy = track.velocities[idx]
        dir_x, dir_y = lane_dirs[idx]
        if vx is None or vy is None:
            long_vel[idx] = None
            lat_vel[idx] = None
        else:
            long_vel[idx] = vx * dir_x + vy * dir_y
            perp_x, perp_y = -dir_y, dir_x
            lat_vel[idx] = vx * perp_x + vy * perp_y

    for idx in range(1, TOTAL_STEPS):
        dt = None
        if track.timestamps[idx] is not None and track.timestamps[idx - 1] is not None:
            dt_val = (track.timestamps[idx] - track.timestamps[idx - 1]) / 1_000_000.0
            if dt_val > 1e-6:
                dt = dt_val
        if dt is None:
            dt = DEFAULT_DT_SECONDS

        if long_vel[idx] is not None and long_vel[idx - 1] is not None:
            long_accel[idx] = (long_vel[idx] - long_vel[idx - 1]) / dt
        else:
            long_accel[idx] = None

        if lat_vel[idx] is not None and lat_vel[idx - 1] is not None:
            lat_accel[idx] = (lat_vel[idx] - lat_vel[idx - 1]) / dt
        else:
            lat_accel[idx] = None

    return {
        "speed": track.speeds,
        "long_vel": long_vel,
        "lat_vel": lat_vel,
        "long_accel": long_accel,
        "lat_accel": lat_accel,
    }


def find_sustained_neighbors(
    track_data: List[TrackSeries],
    subject_idx: int,
    lane_id: Optional[int],
    lane_dir_map: Dict[int, Tuple[float, float]],
    start_idx: int,
    end_idx: int,
    min_steps: int,
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    start_idx = max(0, start_idx)
    end_idx = min(TOTAL_STEPS - 1, end_idx)
    if end_idx < start_idx:
        return None, None, None

    window = list(range(start_idx, end_idx + 1))
    if len(window) < min_steps:
        return None, None, None

    subject = track_data[subject_idx]
    subject_positions: List[Optional[Tuple[float, float, float]]] = [subject.positions[t] for t in window]
    if any(pos is None for pos in subject_positions):
        return None, None, None
    if any(not subject.valid[t] for t in window):
        return None, None, None

    lane_dir = lane_dir_map.get(int(lane_id)) if lane_id is not None else None
    if lane_dir is None:
        lane_dir = (1.0, 0.0)

    lead_idx: Optional[int] = None
    rear_idx: Optional[int] = None
    sdc_idx: Optional[int] = None
    best_lead = float("inf")
    best_rear = -float("inf")
    best_sdc = float("inf")

    for idx, other in enumerate(track_data):
        if idx == subject_idx or other.agent_type != 1:
            continue

        is_sdc = other.is_sdc == 1
        lon_values: List[float] = []
        total_distance = 0.0
        valid = True

        for subj_pos, t in zip(subject_positions, window):
            if subj_pos is None:
                valid = False
                break
            if not other.valid[t]:
                valid = False
                break
            other_pos = other.positions[t]
            if other_pos is None:
                valid = False
                break

            if not is_sdc:
                other_lane = other.lane_ids[t]
                if lane_id is not None and other_lane != lane_id:
                    valid = False
                    break

            delta_x = other_pos[0] - subj_pos[0]
            delta_y = other_pos[1] - subj_pos[1]
            lon = delta_x * lane_dir[0] + delta_y * lane_dir[1]
            lat = delta_x * (-lane_dir[1]) + delta_y * lane_dir[0]

            if not is_sdc and abs(lat) > LATERAL_DISTANCE_THRESHOLD:
                valid = False
                break

            lon_values.append(lon)
            total_distance += math.hypot(delta_x, delta_y)

        if not valid or len(lon_values) < min_steps:
            continue

        if all(lon > 0 for lon in lon_values):
            avg_lon = sum(lon_values) / len(lon_values)
            if avg_lon < best_lead:
                best_lead = avg_lon
                lead_idx = idx
        elif all(lon < 0 for lon in lon_values):
            avg_lon = sum(lon_values) / len(lon_values)
            if avg_lon > best_rear:
                best_rear = avg_lon
                rear_idx = idx

        if is_sdc:
            avg_dist = total_distance / len(lon_values)
            if avg_dist < best_sdc:
                best_sdc = avg_dist
                sdc_idx = idx

    return lead_idx, rear_idx, sdc_idx


def find_lateral_settle_index(lat_accel: List[Optional[float]], start_idx: int) -> int:
    prev = None
    for idx in range(start_idx, TOTAL_STEPS):
        val = lat_accel[idx]
        if val is None:
            continue
        if abs(val) < 1e-6:
            continue
        if prev is None:
            prev = val
            continue
        if prev * val <= 0:
            return idx
        prev = val
    return min(TOTAL_STEPS - 1, start_idx)


def compute_neighbor_metrics(
    subject: TrackSeries,
    neighbor: Optional[TrackSeries],
    lane_dirs: List[Tuple[float, float]],
    subject_long_vel: List[Optional[float]],
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    if neighbor is None:
        return [None] * TOTAL_STEPS, [None] * TOTAL_STEPS

    gaps: List[Optional[float]] = [None] * TOTAL_STEPS
    rel_speeds: List[Optional[float]] = [None] * TOTAL_STEPS

    for idx in range(TOTAL_STEPS):
        dir_x, dir_y = lane_dirs[idx]
        subject_pos = subject.positions[idx]
        neighbor_pos = neighbor.positions[idx]
        if subject_pos is None or neighbor_pos is None:
            gaps[idx] = None
            rel_speeds[idx] = None
            continue

        delta_x = neighbor_pos[0] - subject_pos[0]
        delta_y = neighbor_pos[1] - subject_pos[1]
        gaps[idx] = delta_x * dir_x + delta_y * dir_y

        vx_n, vy_n = neighbor.velocities[idx]
        long_vel_subject = subject_long_vel[idx]
        if vx_n is None or vy_n is None or long_vel_subject is None:
            rel_speeds[idx] = None
        else:
            rel_speeds[idx] = vx_n * dir_x + vy_n * dir_y - long_vel_subject

    return gaps, rel_speeds


def format_series(series: List[Optional[float]]) -> List[str]:
    formatted: List[str] = []
    for value in series:
        if value is None or not math.isfinite(value):
            formatted.append("")
        else:
            formatted.append(f"{value:.6f}")
    return formatted


def format_position_series(track: Optional[TrackSeries]) -> List[str]:
    if track is None:
        return ["" for _ in range(TOTAL_STEPS)]
    formatted: List[str] = []
    for pos in track.positions:
        if pos is None:
            formatted.append("")
        else:
            x, y, z = pos
            formatted.append(f"{x:.6f},{y:.6f},{z:.6f}")
    return formatted


def build_trace_header(base_header: List[str]) -> List[str]:
    per_step_columns = [
        "SV_speed",
        "SV_longitudinal_accel",
        "SV_lateral_accel",
        "SV_position",
        "LC_longitudinal_gap",
        "LC_relative_speed",
        "LC_position",
        "RC_longitudinal_gap",
        "RC_relative_speed",
        "RC_position",
        "LT_longitudinal_gap",
        "LT_relative_speed",
        "LT_position",
        "RT_longitudinal_gap",
        "RT_relative_speed",
        "RT_position",
        "SDC_longitudinal_gap",
        "SDC_relative_speed",
        "SDC_position",
    ]
    header = list(base_header)
    for label in STEP_LABELS:
        for col in per_step_columns:
            header.append(f"{label}_{col}")
    return header


# ---------------------------------------------------------------------------
# Main processing

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure lane-change events with detailed metrics.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing TFRecord files.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit for processed records.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events_header = [
        "scenario_id",
        "lane_change_timestep",
        "lane_change_timestamp",
        "current_lane_id",
        "target_lane_id",
        "sv_id",
        "sv_is_SDC",
        "sdc_id",
        "sdc_is_SDC",
        "lc_id",
        "lc_is_SDC",
        "rc_id",
        "rc_is_SDC",
        "lt_id",
        "lt_is_SDC",
        "rt_id",
        "rt_is_SDC",
    ]
    traces_header = build_trace_header(events_header)

    with EVENTS_CSV.open("w", newline="", encoding="utf-8") as events_fp, TRACES_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as traces_fp:
        events_writer = csv.writer(events_fp)
        traces_writer = csv.writer(traces_fp)

        events_writer.writerow(events_header)
        traces_writer.writerow(traces_header)

        total_scenarios = 0
        total_events = 0

        for file_path, record_index, example in iterate_examples(args.data_dir, args.max_records):
            total_scenarios += 1

            feature_map = example.features.feature

            lane_index, lane_dir_map, lane_centers = collect_lane_points(feature_map)
            road_line_segments = build_road_line_segments(feature_map)

            state_type = get_float_list(feature_map, "state/type")
            num_tracks = len(state_type)
            if num_tracks == 0:
                continue

            track_ids = decode_track_ids(feature_map, num_tracks)
            track_data: List[TrackSeries] = []

            for track_idx in range(num_tracks):
                track = assemble_track_series(feature_map, track_idx, num_tracks)
                track_data.append(track)

            sdc_track_idx: Optional[int] = None
            for idx, track in enumerate(track_data):
                if track.is_sdc:
                    sdc_track_idx = idx
                    break

            # Populate lane IDs for each track.
            for track in track_data:
                lane_ids: List[Optional[int]] = []
                for pos in track.positions:
                    if pos is None:
                        lane_ids.append(None)
                        continue
                    x, y, z = pos
                    lane_ids.append(nearest_lane_id(lane_index, x, y, z))
                track.lane_ids = lane_ids

            scenario_bytes = get_bytes_list(feature_map, "scenario/id")
            scenario_id = (
                scenario_bytes[0].decode("utf-8", errors="ignore")
                if scenario_bytes
                else f"{file_path.name}:{record_index}"
            )

            for track_idx, track in enumerate(track_data):
                if track.agent_type != 1:
                    continue  # vehicles only

                lane_ids = track.lane_ids
                positions_xy: List[Optional[Tuple[float, float]]] = []
                for pos in track.positions:
                    if pos is None:
                        positions_xy.append(None)
                    else:
                        positions_xy.append((pos[0], pos[1]))

                events = extract_lane_change_events(
                    lane_ids, positions_xy, road_line_segments, lane_dir_map
                )
                if not events:
                    continue

                lane_dirs = compute_lane_dirs_for_steps(lane_ids, lane_dir_map)
                subject_metrics = compute_subject_metrics(track, lane_dirs)

                for current_lane_id, target_lane_id, switch_index in events:
                    current_center = lane_centers.get(current_lane_id)
                    target_center = lane_centers.get(target_lane_id)
                    if current_center is None or target_center is None:
                        continue
                    if math.hypot(
                        current_center[0] - target_center[0],
                        current_center[1] - target_center[1],
                    ) < LANE_CENTER_SEPARATION_THRESHOLD:
                        continue

                    lane_change_timestamp = (
                        track.timestamps[switch_index] if track.timestamps[switch_index] is not None else ""
                    )

                    pre_start = switch_index - PRE_WINDOW_STEPS
                    pre_end = switch_index - 1
                    lc_idx = rc_idx = lt_idx = rt_idx = None
                    sdc_pre = sdc_post = None
                    if pre_end >= pre_start:
                        lc_idx, rc_idx, sdc_pre = find_sustained_neighbors(
                            track_data,
                            track_idx,
                            current_lane_id,
                            lane_dir_map,
                            pre_start,
                            pre_end,
                            PRE_WINDOW_STEPS,
                        )

                    settle_idx = find_lateral_settle_index(subject_metrics["lat_accel"], switch_index)
                    post_start = max(switch_index, settle_idx)
                    post_end = post_start + PRE_WINDOW_STEPS - 1
                    lt_idx, rt_idx, sdc_post = find_sustained_neighbors(
                        track_data,
                        track_idx,
                        target_lane_id,
                        lane_dir_map,
                        post_start,
                        post_end,
                        PRE_WINDOW_STEPS,
                    )

                    sdc_idx = sdc_post if sdc_post is not None else sdc_pre
                    if sdc_idx is None and sdc_track_idx is not None and sdc_track_idx != track_idx:
                        sdc_idx = sdc_track_idx

                    def neighbor_info(idx: Optional[int]) -> Tuple[str, str]:
                        if idx is None:
                            return "", ""
                        return track_data[idx].track_id, str(track_data[idx].is_sdc)

                    total_events += 1

                    lc_id, lc_sdc = neighbor_info(lc_idx)
                    rc_id, rc_sdc = neighbor_info(rc_idx)
                    lt_id, lt_sdc = neighbor_info(lt_idx)
                    rt_id, rt_sdc = neighbor_info(rt_idx)
                    sdc_id, sdc_sdc = neighbor_info(sdc_idx)

                    events_writer.writerow([
                        scenario_id,
                        switch_index,
                        lane_change_timestamp,
                        current_lane_id,
                        target_lane_id,
                        track.track_id,
                        track.is_sdc,
                        sdc_id,
                        sdc_sdc,
                        lc_id,
                        lc_sdc,
                        rc_id,
                        rc_sdc,
                        lt_id,
                        lt_sdc,
                        rt_id,
                        rt_sdc,
                    ])

                    subject_long_vel = subject_metrics["long_vel"]
                    sv_speed_series = format_series(subject_metrics["speed"])
                    sv_long_accel_series = format_series(subject_metrics["long_accel"])
                    sv_lat_accel_series = format_series(subject_metrics["lat_accel"])
                    sv_pos_series = format_position_series(track)

                    def neighbor_series(idx: Optional[int]) -> Tuple[List[str], List[str], List[str]]:
                        neighbor_track = track_data[idx] if idx is not None else None
                        gaps, rel = compute_neighbor_metrics(
                            track, neighbor_track, lane_dirs, subject_long_vel
                        )
                        return (
                            format_series(gaps),
                            format_series(rel),
                            format_position_series(neighbor_track),
                        )

                    lc_gap_series, lc_rel_series, lc_pos_series = neighbor_series(lc_idx)
                    rc_gap_series, rc_rel_series, rc_pos_series = neighbor_series(rc_idx)
                    lt_gap_series, lt_rel_series, lt_pos_series = neighbor_series(lt_idx)
                    rt_gap_series, rt_rel_series, rt_pos_series = neighbor_series(rt_idx)
                    sdc_gap_series, sdc_rel_series, sdc_pos_series = neighbor_series(sdc_idx)

                    trace_row: List[Any] = [
                        scenario_id,
                        switch_index,
                        lane_change_timestamp,
                        current_lane_id,
                        target_lane_id,
                        track.track_id,
                        track.is_sdc,
                        sdc_id,
                        sdc_sdc,
                        lc_id,
                        lc_sdc,
                        rc_id,
                        rc_sdc,
                        lt_id,
                        lt_sdc,
                        rt_id,
                        rt_sdc,
                    ]

                    for idx in range(TOTAL_STEPS):
                        trace_row.extend([
                            sv_speed_series[idx],
                            sv_long_accel_series[idx],
                            sv_lat_accel_series[idx],
                            sv_pos_series[idx],
                            lc_gap_series[idx],
                            lc_rel_series[idx],
                            lc_pos_series[idx],
                            rc_gap_series[idx],
                            rc_rel_series[idx],
                            rc_pos_series[idx],
                            lt_gap_series[idx],
                            lt_rel_series[idx],
                            lt_pos_series[idx],
                            rt_gap_series[idx],
                            rt_rel_series[idx],
                            rt_pos_series[idx],
                            sdc_gap_series[idx],
                            sdc_rel_series[idx],
                            sdc_pos_series[idx],
                        ])

                    traces_writer.writerow(trace_row)

        print(f"Processed {total_scenarios} scenarios. Detected {total_events} lane-change events.")


def iterate_examples(data_dir: Path, max_records: Optional[int]):
    file_paths = sorted(data_dir.glob("*.tfrecord*"))
    file_iter: Iterable[Path] = file_paths
    if tqdm is not None:
        file_iter = tqdm(file_paths, total=len(file_paths), desc="Files", unit="file")

    processed = 0
    for tf_path in file_iter:
        dataset = tf.data.TFRecordDataset(str(tf_path))
        for record_index, raw in enumerate(dataset):
            if max_records is not None and processed >= max_records:
                return
            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            yield tf_path, record_index, example
            processed += 1


if __name__ == "__main__":
    main()

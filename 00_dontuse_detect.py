#!/usr/bin/env python3
"""Detect lane-change events for vehicles in Waymo Motion TFRecord scenarios.

For every TFRecord stored under ``data/`` the script parses each scenario,
collects the 91-step XYZ trajectory (10 past, 1 current, 80 future) for every
agent marked as a vehicle (``state/type == 1``), and identifies the closest
lane-centre ID at each timestep using the road graph samples where
``roadgraph_samples/type`` is 1 or 2.

If a vehicle's nearest-lane assignment changes over its trace, the script
records:
  * the scenario id (`scenario/id`)
  * the vehicle id (`state/id`)
  * whether the vehicle is the self-driving car (`state/is_sdc`)

Outputs are written to:
  * ``outputs/00_lane_change_events.csv`` – one row per lane-change vehicle
  * ``outputs/00_lane_change_traces.csv`` – per-timestep trace for those vehicles

Run with::

    ./00_detect.py

Optional arguments allow overriding the data directory or limiting processed
records (useful for debugging). TensorFlow must be available within the active
environment.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - runtime guard only
    raise SystemExit(
        "TensorFlow is required to parse TFRecord files. "
        "Install it in your virtual environment before running this script."
    ) from exc


# Constants describing the Waymo Motion dataset layout.
PAST_STEPS = 10
CURRENT_STEPS = 1
FUTURE_STEPS = 80
TOTAL_STEPS = PAST_STEPS + CURRENT_STEPS + FUTURE_STEPS

# Lane centre types according to the public schema.
LANE_CENTER_TYPES = {1, 2}
ROAD_LINE_TYPES = {6, 7, 8, 9, 10, 11, 12, 13}
PARALLEL_DOT_THRESHOLD = math.cos(math.radians(20))  # require ~20° or less difference

# Spatial index parameters (coarse grid hash for nearest-neighbour queries).
LANE_CELL_SIZE = 10.0  # metres
LANE_SEARCH_RADIUS = 6  # number of cells to expand when searching neighbours

# Output folder and filenames.
OUTPUT_DIR = Path("outputs")
EVENTS_CSV = OUTPUT_DIR / "00_lane_change_events.csv"
TRACES_CSV = OUTPUT_DIR / "00_lane_change_traces.csv"


def get_float_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[float]:
    feat = feature_map.get(key)
    return feat.float_list.value if feat is not None else ()


def get_int_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[int]:
    feat = feature_map.get(key)
    return feat.int64_list.value if feat is not None else ()


def get_bytes_list(feature_map: Dict[str, tf.train.Feature], key: str) -> Sequence[bytes]:
    feat = feature_map.get(key)
    return feat.bytes_list.value if feat is not None else ()


def decode_track_ids(feature_map: Dict[str, tf.train.Feature], num_tracks: int) -> List[str]:
    """Best-effort decode of state/id values across different dataset variants."""
    decoded: List[str] = []

    byte_ids = get_bytes_list(feature_map, "state/id")
    float_ids = get_float_list(feature_map, "state/id")
    int_ids = get_int_list(feature_map, "state/id")

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


def is_valid_coord(x: float, y: float, z: float) -> bool:
    """Treat (-1,-1,-1) entries as padding; everything else is considered valid."""
    return not (
        math.isclose(x, -1.0, abs_tol=1e-3)
        and math.isclose(y, -1.0, abs_tol=1e-3)
        and math.isclose(z, -1.0, abs_tol=1e-3)
    )


def slice_track(values: Sequence[Any], track_idx: int, steps: int, fill: Any) -> List[Any]:
    if not values:
        return [fill] * steps
    base = track_idx * steps
    out: List[Any] = []
    for step in range(steps):
        idx = base + step
        if idx < len(values):
            out.append(values[idx])
        else:
            out.append(fill)
    return out


def gather_vehicle_trace(feature_map: Dict[str, tf.train.Feature], track_idx: int, num_tracks: int) -> List[Dict[str, float]]:
    """Return chronological timeline for the vehicle with timestamps and coordinates."""
    trace: List[Dict[str, float]] = []

    segments = [
        ("state/past", PAST_STEPS),
        ("state/current", CURRENT_STEPS),
        ("state/future", FUTURE_STEPS),
    ]

    offset = 0
    for prefix, steps in segments:
        if steps <= 0:
            continue
        xs = get_float_list(feature_map, f"{prefix}/x")
        ys = get_float_list(feature_map, f"{prefix}/y")
        zs = get_float_list(feature_map, f"{prefix}/z")
        ts_float = get_float_list(feature_map, f"{prefix}/timestamp_micros")
        ts_int = get_int_list(feature_map, f"{prefix}/timestamp_micros")
        valids = get_float_list(feature_map, f"{prefix}/valid")

        series_x = [
            float(val) if val is not None else math.nan
            for val in slice_track(xs, track_idx, steps, None)
        ]
        series_y = [
            float(val) if val is not None else math.nan
            for val in slice_track(ys, track_idx, steps, None)
        ]
        series_z = [
            float(val) if val is not None else math.nan
            for val in slice_track(zs, track_idx, steps, None)
        ]

        if ts_int:
            raw = slice_track(ts_int, track_idx, steps, None)
            timestamps = [int(val) if val is not None else None for val in raw]
        elif ts_float:
            raw = slice_track(ts_float, track_idx, steps, None)
            timestamps = [int(val) if val is not None else None for val in raw]
        else:
            timestamps = [None] * steps

        valid_series: Optional[List[float]] = (
            [float(val) for val in slice_track(valids, track_idx, steps, 0.0)]
            if valids
            else None
        )

        for step in range(steps):
            x = series_x[step]
            y = series_y[step]
            z = series_z[step]
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                continue
            if not is_valid_coord(x, y, z):
                continue
            if valid_series is not None and valid_series[step] <= 0.5:
                continue

            trace.append(
                {
                    "order": offset,
                    "timestamp_micros": timestamps[step],
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            offset += 1
    return trace


def build_lane_index(points: Iterable[Tuple[float, float, float, int]]) -> Dict[Tuple[int, int], List[Tuple[float, float, float, int]]]:
    grid: Dict[Tuple[int, int], List[Tuple[float, float, float, int]]] = {}
    for x, y, z, lane_id in points:
        cell = (int(math.floor(x / LANE_CELL_SIZE)), int(math.floor(y / LANE_CELL_SIZE)))
        grid.setdefault(cell, []).append((x, y, z, lane_id))
    return grid


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


def lanes_are_parallel(lane_before: Optional[int], lane_after: Optional[int], lane_dir_map: Dict[int, Tuple[float, float]]) -> bool:
    if lane_before is None or lane_after is None:
        return False
    dir_before = lane_dir_map.get(int(lane_before))
    dir_after = lane_dir_map.get(int(lane_after))
    if dir_before is None or dir_after is None:
        return False
    dot = abs(dir_before[0] * dir_after[0] + dir_before[1] * dir_after[1])
    return dot >= PARALLEL_DOT_THRESHOLD


def crosses_road_line_between(
    positions: List[Tuple[float, float]],
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
        x0, y0 = positions[idx]
        x1, y1 = positions[idx + 1]
        if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(x1) and math.isfinite(y1)):
            continue
        for seg in road_line_segments:
            q0x, q0y, q1x, q1y = seg
            if segments_intersect((x0, y0), (x1, y1), (q0x, q0y), (q1x, q1y)):
                return True
    return False


def extract_lane_change_events(
    lane_ids: List[Optional[int]],
    positions: List[Tuple[float, float]],
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

        events.append((current_lane, lane, idx))
        current_lane = lane
        last_idx = idx

    return events


def collect_lane_points(feature_map: Dict[str, tf.train.Feature]) -> Tuple[
    Dict[Tuple[int, int], List[Tuple[float, float, float, int]]],
    Dict[int, Tuple[float, float]],
]:
    types = get_int_list(feature_map, "roadgraph_samples/type")
    ids = get_int_list(feature_map, "roadgraph_samples/id")
    xyz = get_float_list(feature_map, "roadgraph_samples/xyz")
    directions = get_float_list(feature_map, "roadgraph_samples/dir")
    lane_points: List[Tuple[float, float, float, int]] = []
    dir_map: Dict[int, List[Tuple[float, float]]] = {}

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

    lane_dir_avg: Dict[int, Tuple[float, float]] = {}
    for lane_id, vectors in dir_map.items():
        sx = sum(v[0] for v in vectors)
        sy = sum(v[1] for v in vectors)
        norm = math.hypot(sx, sy)
        if norm > 1e-6:
            lane_dir_avg[lane_id] = (sx / norm, sy / norm)

    return build_lane_index(lane_points), lane_dir_avg


def lane_change_detected(lane_ids: List[Optional[int]]) -> bool:
    filtered = [lane for lane in lane_ids if lane is not None]
    if len(filtered) < 2:
        return False
    first = filtered[0]
    for lane in filtered[1:]:
        if lane != first:
            return True
    return False


def process_scenario(
    example: tf.train.Example,
    file_path: Path,
    record_index: int,
    events_writer: csv.writer,
    traces_writer: csv.writer,
) -> int:
    feature_map = example.features.feature
    scenario_bytes = get_bytes_list(feature_map, "scenario/id")
    scenario_id = (
        scenario_bytes[0].decode("utf-8", errors="ignore")
        if scenario_bytes
        else f"{file_path.name}:{record_index}"
    )

    state_type = get_float_list(feature_map, "state/type")
    num_tracks = len(state_type)
    if num_tracks == 0:
        return 0

    vehicle_indices = [idx for idx, value in enumerate(state_type) if int(round(value)) == 1]
    if not vehicle_indices:
        return 0

    track_ids = decode_track_ids(feature_map, num_tracks)
    is_sdc_list = get_float_list(feature_map, "state/is_sdc")

    lane_index, lane_dir_map = collect_lane_points(feature_map)
    road_line_segments = build_road_line_segments(feature_map)

    lane_change_count = 0

    for track_idx in vehicle_indices:
        trace = gather_vehicle_trace(feature_map, track_idx, num_tracks)
        if not trace:
            continue

        lane_assignments: List[Optional[int]] = []
        positions: List[Tuple[float, float]] = []
        for sample in trace:
            lane_id = nearest_lane_id(lane_index, sample["x"], sample["y"], sample["z"])
            lane_assignments.append(lane_id)
            sample["lane_id"] = lane_id
            positions.append((sample["x"], sample["y"]))

        events = extract_lane_change_events(
            lane_assignments, positions, road_line_segments, lane_dir_map
        )
        if not events:
            continue

        lane_change_count += 1
        sv_id = track_ids[track_idx] or ""
        is_sdc = int(is_sdc_list[track_idx]) if track_idx < len(is_sdc_list) else 0

        for _, _, switch_index in events:
            switch_step = switch_index if switch_index is not None else ""
            events_writer.writerow([scenario_id, sv_id, is_sdc, switch_step])

        for step_idx, sample in enumerate(trace):
            lane_id = sample.get("lane_id")
            traces_writer.writerow(
                [
                    scenario_id,
                    sv_id,
                    is_sdc,
                    step_idx,
                    sample.get("timestamp_micros"),
                    sample["x"],
                    sample["y"],
                    sample["z"],
                    lane_id if lane_id is not None else "",
                ]
            )

    return lane_change_count


def iterate_examples(data_dir: Path, max_records: Optional[int] = None):
    processed = 0
    for tf_path in sorted(data_dir.glob("*.tfrecord*")):
        dataset = tf.data.TFRecordDataset(str(tf_path))
        for record_index, raw in enumerate(dataset):
            if max_records is not None and processed >= max_records:
                return
            example = tf.train.Example()
            example.ParseFromString(raw.numpy())
            yield tf_path, record_index, example
            processed += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect lane-change events in Waymo Motion TFRecords.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing TFRecord files.")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional limit on how many records to process (useful for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with EVENTS_CSV.open("w", newline="", encoding="utf-8") as events_fp, TRACES_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as traces_fp:
        events_writer = csv.writer(events_fp)
        traces_writer = csv.writer(traces_fp)

        events_writer.writerow(["scenario_id", "sv_id", "is_sdc", "switch_timestep"])
        traces_writer.writerow(
            [
                "scenario_id",
                "sv_id",
                "is_sdc",
                "timestep_index",
                "timestamp_micros",
                "x",
                "y",
                "z",
                "lane_id",
            ]
        )

        total_lane_changes = 0
        total_records = 0

        for file_path, record_index, example in iterate_examples(data_dir, args.max_records):
            total_records += 1
            lane_changes = process_scenario(example, file_path, record_index, events_writer, traces_writer)
            total_lane_changes += lane_changes

            if total_records % 100 == 0:
                print(f"Processed {total_records} records... lane changes detected so far: {total_lane_changes}")

        print(f"Finished. Processed {total_records} records and detected {total_lane_changes} lane-changing vehicles.")


if __name__ == "__main__":
    main()

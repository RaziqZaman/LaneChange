#!/usr/bin/env python3
"""Extract per-timestep data for vehicle state/id == 2520 across all scenarios.

The script scans every TFRecord stored in ``data/`` (Waymo Motion format),
locates the agent whose ``state/id`` equals 2520, reconstructs its 91-sample
trajectory (10 past + 1 current + 80 future), associates the nearest lane ID
for every timestamp, and writes the result to ``outputs/veh2520.csv``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
LANE_CENTER_TYPES = {1, 2}
LANE_CELL_SIZE = 10.0
LANE_SEARCH_RADIUS = 6
TARGET_ID = "2520"
OUTPUT_PATH = Path("outputs/veh2520.csv")


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


def is_padding(x: float, y: float, z: float) -> bool:
    return (
        math.isclose(x, -1.0, abs_tol=1e-3)
        and math.isclose(y, -1.0, abs_tol=1e-3)
        and math.isclose(z, -1.0, abs_tol=1e-3)
    )


def slice_track(values: Sequence[Any], track_idx: int, steps: int, fill: Any) -> List[Any]:
    if not values:
        return [fill] * steps
    base = track_idx * steps
    result: List[Any] = []
    for step in range(steps):
        idx = base + step
        if idx < len(values):
            result.append(values[idx])
        else:
            result.append(fill)
    return result


def gather_trace(feature_map: Dict[str, tf.train.Feature], track_idx: int, num_tracks: int) -> List[Dict[str, float]]:
    trace: List[Dict[str, float]] = []
    segments = [
        ("state/past", PAST_STEPS),
        ("state/current", CURRENT_STEPS),
        ("state/future", FUTURE_STEPS),
    ]
    order = 0
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
            raw_ts = slice_track(ts_int, track_idx, steps, None)
            timestamps = [int(val) if val is not None else None for val in raw_ts]
        elif ts_float:
            raw_ts = slice_track(ts_float, track_idx, steps, None)
            timestamps = [int(val) if val is not None else None for val in raw_ts]
        else:
            timestamps = [None] * steps

        valid_series = (
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
            if is_padding(x, y, z):
                continue
            if valid_series is not None and valid_series[step] <= 0.5:
                continue
            trace.append(
                {
                    "order": order,
                    "timestamp": timestamps[step],
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
            order += 1
    return trace


def build_lane_index(feature_map: Dict[str, tf.train.Feature]):
    types = get_int_list(feature_map, "roadgraph_samples/type")
    ids = get_int_list(feature_map, "roadgraph_samples/id")
    xyz = get_float_list(feature_map, "roadgraph_samples/xyz")
    grid: Dict[Tuple[int, int], List[Tuple[float, float, float, int]]] = {}
    for idx, lane_type in enumerate(types):
        if lane_type not in LANE_CENTER_TYPES:
            continue
        if 3 * idx + 2 >= len(xyz) or idx >= len(ids):
            continue
        x = float(xyz[3 * idx])
        y = float(xyz[3 * idx + 1])
        z = float(xyz[3 * idx + 2])
        lane_id = int(ids[idx])
        cell = (int(math.floor(x / LANE_CELL_SIZE)), int(math.floor(y / LANE_CELL_SIZE)))
        grid.setdefault(cell, []).append((x, y, z, lane_id))
    return grid


def nearest_lane_id(grid: Dict[Tuple[int, int], List[Tuple[float, float, float, int]]], x: float, y: float, z: float) -> Optional[int]:
    if not grid:
        return None
    cell_x = int(math.floor(x / LANE_CELL_SIZE))
    cell_y = int(math.floor(y / LANE_CELL_SIZE))
    best_dist2 = float("inf")
    best_lane: Optional[int] = None
    for radius in range(LANE_SEARCH_RADIUS + 1):
        any_found = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                bucket = (cell_x + dx, cell_y + dy)
                for px, py, pz, lane_id in grid.get(bucket, ()):
                    any_found = True
                    dist2 = (px - x) ** 2 + (py - y) ** 2 + (pz - z) ** 2
                    if dist2 < best_dist2:
                        best_dist2 = dist2
                        best_lane = lane_id
        if any_found and best_lane is not None:
            break
    return best_lane


def process_example(example: tf.train.Example, scenario_id: str, writer: csv.writer) -> bool:
    feature_map = example.features.feature
    state_type = get_float_list(feature_map, "state/type")
    num_tracks = len(state_type)
    if num_tracks == 0:
        return False

    track_ids = decode_track_ids(feature_map, num_tracks)
    try:
        target_idx = track_ids.index(TARGET_ID)
    except ValueError:
        return False

    trace = gather_trace(feature_map, target_idx, num_tracks)
    if not trace:
        return False

    lane_grid = build_lane_index(feature_map)
    rows: List[Tuple[Optional[int], int, str, Optional[int]]] = []

    for sample in trace:
        lane_id = nearest_lane_id(lane_grid, sample["x"], sample["y"], sample["z"])
        xyz = f"{sample['x']:.6f},{sample['y']:.6f},{sample['z']:.6f}"
        rows.append(
            (
                sample["timestamp"],
                sample["order"],
                xyz,
                lane_id if lane_id is not None else None,
            )
        )

    rows.sort(key=lambda item: ((item[0] if item[0] is not None else float("inf")), item[1]))

    for timestamp, order, xyz, lane_id in rows:
        writer.writerow([scenario_id, timestamp, xyz, lane_id if lane_id is not None else ""])
    return True


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
    parser = argparse.ArgumentParser(description="Extract vehicle 2520 trajectory across scenarios.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing TFRecord files.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional limit for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["scenario_id", "timestamp_micros", "xyz", "lane_id"])

        matches = 0
        total = 0
        for _, record_index, example in iterate_examples(data_dir, args.max_records):
            total += 1
            feature_map = example.features.feature
            scenario_bytes = get_bytes_list(feature_map, "scenario/id")
            scenario_id = (
                scenario_bytes[0].decode("utf-8", errors="ignore")
                if scenario_bytes
                else f"record-{record_index}"
            )
            if process_example(example, scenario_id, writer):
                matches += 1

        print(f"Processed {total} scenarios. Vehicle 2520 found in {matches} scenarios.")


if __name__ == "__main__":
    main()

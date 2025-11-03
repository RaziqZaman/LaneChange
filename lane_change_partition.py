#!/usr/bin/env python3
"""Extract scenarios that contain lane changes from Waymo-style TFRecords.

The script scans every TFRecord file in the provided ``data/`` directory,
identifies tracks marked as ``state/objects_of_interest == 1`` and analyses
their trajectories.  A lane change is flagged when the lateral displacement
of the track (estimated with PCA over its past/current/future XY samples)
exceeds configurable thresholds.

Usage::

    ./lane_change_partition.py --data-dir data --output lane_changes.json

Without ``--output`` a human-readable summary is printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    # TensorFlow gives convenient TFRecord readers and tf.train.Example parsing.
    import tensorflow as tf
except Exception as exc:  # pragma: no cover - handled at runtime only
    raise SystemExit(
        "TensorFlow is required to parse TFRecord files. "
        "Install it in your virtual environment before running this script."
    ) from exc


# Tunable thresholds for the PCA-based lateral displacement heuristic.
MIN_POINTS = 6
LATERAL_RANGE_THRESHOLD = 5.0  # metres
LATERAL_SHIFT_THRESHOLD = 3.0  # metres (start vs end)
LATERAL_SLOPE_THRESHOLD = 0.05  # metres per timestep (indicative trend)

# Only analyse tracks explicitly marked as interesting (typically lane changers).
INTEREST_FLAG_THRESHOLD = 0.5

TYPE_LABELS = {
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}


@dataclass
class LaneChangeObservation:
    file_path: str
    record_index: int
    scenario_id: str
    track_index: int
    track_id: Optional[str]
    agent_type: str
    num_points: int
    lateral_range: float
    lateral_shift: float
    lateral_slope: float


def agent_label(raw_value: float) -> str:
    """Map numeric agent type codes to short human labels."""
    return TYPE_LABELS.get(int(round(raw_value)), "unknown")


def decode_track_ids(feature_map: Dict[str, tf.train.Feature]) -> List[Optional[str]]:
    """Return decoded UTF-8 track ids when available."""
    if "state/id" not in feature_map:
        return []
    result: List[Optional[str]] = []
    for value in feature_map["state/id"].bytes_list.value:
        try:
            result.append(value.decode("utf-8"))
        except Exception:
            result.append(None)
    return result


def gather_timeline(
    feature_map: Dict[str, tf.train.Feature],
    track_idx: int,
    num_tracks: int,
) -> List[Tuple[float, float, float]]:
    """Collect (time_offset, x, y) samples for the given track index."""

    def _floats(name: str) -> Sequence[float]:
        feat = feature_map.get(name)
        return feat.float_list.value if feat is not None else []

    timeline: List[Tuple[float, float, float]] = []

    # Past trajectory (time offsets negative, oldest first).
    past_x = _floats("state/past/x")
    past_y = _floats("state/past/y")
    past_valid = _floats("state/past/valid")
    past_steps = len(past_x) // num_tracks if num_tracks else 0
    for step in range(past_steps):
        idx = track_idx + step * num_tracks
        if idx >= len(past_x) or idx >= len(past_y):
            break
        if _is_valid(idx, past_x, past_y, past_valid):
            offset = float(step - past_steps)
            timeline.append((offset, float(past_x[idx]), float(past_y[idx])))

    # Current position.
    current_x = _floats("state/current/x")
    current_y = _floats("state/current/y")
    current_valid = _floats("state/current/valid")
    if (
        track_idx < len(current_valid)
        and _is_valid(track_idx, current_x, current_y, current_valid)
    ):
        timeline.append((0.0, float(current_x[track_idx]), float(current_y[track_idx])))
    elif track_idx < len(current_x) and track_idx < len(current_y):
        if not current_valid and not _is_padding(current_x[track_idx], current_y[track_idx]):
            timeline.append((0.0, float(current_x[track_idx]), float(current_y[track_idx])))

    # Future trajectory (positive offsets).
    future_x = _floats("state/future/x")
    future_y = _floats("state/future/y")
    future_valid = _floats("state/future/valid")
    future_steps = len(future_x) // num_tracks if num_tracks else 0
    for step in range(future_steps):
        idx = track_idx + step * num_tracks
        if idx >= len(future_x) or idx >= len(future_y):
            break
        if _is_valid(idx, future_x, future_y, future_valid):
            offset = float(step + 1)
            timeline.append((offset, float(future_x[idx]), float(future_y[idx])))
        elif not future_valid and not _is_padding(future_x[idx], future_y[idx]):
            offset = float(step + 1)
            timeline.append((offset, float(future_x[idx]), float(future_y[idx])))

    return timeline


def _is_padding(x_val: float, y_val: float) -> bool:
    """Return True if the coordinate pair corresponds to padded (-1, -1) entries."""
    return math.isclose(x_val, -1.0, abs_tol=1e-3) and math.isclose(y_val, -1.0, abs_tol=1e-3)


def _is_valid(idx: int, xs: Sequence[float], ys: Sequence[float], valids: Sequence[float]) -> bool:
    """Determine whether a timestep should be considered valid."""
    if valids:
        return idx < len(valids) and valids[idx] > 0.5
    if idx >= len(xs) or idx >= len(ys):
        return False
    return not _is_padding(xs[idx], ys[idx])


def detect_lane_change(timeline: List[Tuple[float, float, float]]) -> Optional[Dict[str, float]]:
    """Return displacement metrics if the trajectory exhibits a lane change."""
    if len(timeline) < MIN_POINTS:
        return None

    timeline.sort(key=lambda sample: sample[0])
    points = np.asarray([[x, y] for _, x, y in timeline], dtype=np.float64)

    # Guard against degenerate trajectories.
    if not np.isfinite(points).all():
        return None
    if np.allclose(points.var(axis=0), 0.0):
        return None

    mean_xy = points.mean(axis=0)
    centered = points - mean_xy
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.isnan(eigvals).any() or np.isnan(eigvecs).any():
        return None

    principal = eigvecs[:, np.argmax(eigvals)]
    if np.linalg.norm(principal) < 1e-6:
        return None

    # Orient the principal axis with the overall direction of motion.
    travel_delta = points[-1] - points[0]
    if float(np.dot(principal, travel_delta)) < 0.0:
        principal = -principal

    lateral_axis = np.array([-principal[1], principal[0]], dtype=np.float64)
    lateral_norm = np.linalg.norm(lateral_axis)
    if lateral_norm < 1e-6:
        return None
    lateral_axis /= lateral_norm

    lateral_coords = centered @ lateral_axis
    if not np.isfinite(lateral_coords).all():
        return None

    lat_range = float(lateral_coords.max() - lateral_coords.min())
    lat_shift = float(lateral_coords[-1] - lateral_coords[0])

    offsets = np.asarray([offset for offset, _, _ in timeline], dtype=np.float64)
    offsets_centered = offsets - offsets.mean()
    denom = float(np.sum(offsets_centered ** 2))
    if denom < 1e-6:
        return None
    lateral_slope = float(
        np.dot(offsets_centered, lateral_coords - lateral_coords.mean()) / denom
    )

    if (
        lat_range >= LATERAL_RANGE_THRESHOLD
        and abs(lat_shift) >= LATERAL_SHIFT_THRESHOLD
        and abs(lateral_slope) >= LATERAL_SLOPE_THRESHOLD
    ):
        return {
            "lateral_range": lat_range,
            "lateral_shift": lat_shift,
            "lateral_slope": lateral_slope,
            "num_points": len(timeline),
        }
    return None


def analyse_example(
    example: tf.train.Example, file_path: Path, record_index: int
) -> List[LaneChangeObservation]:
    """Analyse a single tf.train.Example and return lane change observations."""
    feature_map = example.features.feature

    current_x = feature_map["state/current/x"].float_list.value
    num_tracks = len(current_x)
    if num_tracks == 0:
        return []

    objects_of_interest = feature_map["state/objects_of_interest"].int64_list.value
    agent_types = feature_map["state/type"].float_list.value
    track_ids = decode_track_ids(feature_map)
    scenario_bytes = feature_map.get("scenario/id")
    if scenario_bytes and scenario_bytes.bytes_list.value:
        scenario_id = scenario_bytes.bytes_list.value[0].decode("utf-8", errors="ignore")
    else:
        scenario_id = f"{file_path.name}:{record_index}"

    observation_list: List[LaneChangeObservation] = []

    for track_idx in range(num_tracks):
        if track_idx >= len(objects_of_interest):
            continue
        if objects_of_interest[track_idx] <= INTEREST_FLAG_THRESHOLD:
            continue

        timeline = gather_timeline(feature_map, track_idx, num_tracks)
        metrics = detect_lane_change(timeline)
        if not metrics:
            continue

        observation_list.append(
            LaneChangeObservation(
                file_path=str(file_path),
                record_index=record_index,
                scenario_id=scenario_id,
                track_index=track_idx,
                track_id=track_ids[track_idx] if track_idx < len(track_ids) else None,
                agent_type=agent_label(agent_types[track_idx])
                if track_idx < len(agent_types)
                else "unknown",
                num_points=metrics["num_points"],
                lateral_range=metrics["lateral_range"],
                lateral_shift=metrics["lateral_shift"],
                lateral_slope=metrics["lateral_slope"],
            )
        )

        # If the SDC is already flagged as an object of interest there is no
        # need to keep searching; typically only a handful of tracks per record
        # are marked.
    return observation_list


def scan_directory(
    data_dir: Path, max_records: Optional[int] = None
) -> List[LaneChangeObservation]:
    """Iterate over all TFRecord files in ``data_dir`` collecting observations."""
    results: List[LaneChangeObservation] = []
    record_counter = 0

    for tfrecord_path in sorted(data_dir.glob("*.tfrecord*")):
        dataset = tf.data.TFRecordDataset(str(tfrecord_path))
        for record_index, raw_record in enumerate(dataset):
            if max_records is not None and record_counter >= max_records:
                break
            record_counter += 1

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())
            results.extend(analyse_example(example, tfrecord_path, record_index))

        if max_records is not None and record_counter >= max_records:
            break

    return results


def observations_to_json(observations: Iterable[LaneChangeObservation]) -> List[Dict[str, object]]:
    """Convert dataclass observations into JSON-serialisable dictionaries."""
    json_ready: List[Dict[str, object]] = []
    for obs in observations:
        json_ready.append(
            {
                "scenario_id": obs.scenario_id,
                "file_path": obs.file_path,
                "record_index": obs.record_index,
                "track_index": obs.track_index,
                "track_id": obs.track_id,
                "agent_type": obs.agent_type,
                "num_points": obs.num_points,
                "lateral_range_m": round(obs.lateral_range, 3),
                "lateral_shift_m": round(obs.lateral_shift, 3),
                "lateral_slope": round(obs.lateral_slope, 4),
            }
        )
    return json_ready


def print_summary(observations: Iterable[LaneChangeObservation]) -> None:
    """Print a human-readable summary of detected lane-change scenarios."""
    observations = list(observations)
    if not observations:
        print("No lane-change records detected.")
        return

    grouped: Dict[Tuple[str, str, int], List[LaneChangeObservation]] = {}
    for obs in observations:
        key = (obs.file_path, obs.scenario_id, obs.record_index)
        grouped.setdefault(key, []).append(obs)

    for (file_path, scenario_id, record_index), tracks in grouped.items():
        print(f"{scenario_id}  (file={Path(file_path).name}, record={record_index})")
        for obs in tracks:
            track_label = obs.track_id or f"track-{obs.track_index}"
            print(
                f"  • track {obs.track_index} [{obs.agent_type}] {track_label}: "
                f"shift={obs.lateral_shift:.2f} m, "
                f"range={obs.lateral_range:.2f} m, "
                f"slope={obs.lateral_slope:.3f}"
            )
        print()

    print(f"Total lane-change scenarios: {len(grouped)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract lane-change scenarios from TFRecords.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing *.tfrecord files (default: data/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write JSON results. If omitted, a summary is printed.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Process at most this many records (useful for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise SystemExit(f"Data directory not found: {args.data_dir}")

    observations = scan_directory(args.data_dir, max_records=args.max_records)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(observations_to_json(observations), fh, indent=2)
        print(f"Wrote {len(observations)} lane-change track observations to {args.output}")
    else:
        print_summary(observations)


if __name__ == "__main__":
    main()

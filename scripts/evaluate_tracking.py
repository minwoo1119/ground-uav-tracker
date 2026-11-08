"""Evaluate one project MOT sequence with TrackEval's official metrics.

This adapter treats every bounding box as the same generic airborne-object class.
It does not evaluate detector classification and does not modify either input.
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


class EvaluationError(ValueError):
    """Raised when evaluation inputs or options violate the data contract."""


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    length: int
    ground_truth_path: Path


@dataclass(frozen=True)
class MotRow:
    frame_id: int
    track_id: int
    bbox_xywh: tuple[float, float, float, float]


def _load_dependencies() -> tuple[Any, Any]:
    try:
        import numpy as np
        import trackeval
    except ImportError as error:
        raise EvaluationError(
            "evaluation dependencies are missing; install "
            "requirements/evaluation.txt in a virtual environment"
        ) from error
    return np, trackeval


def _parse_positive_integer(raw_value: str, field: str, location: str) -> int:
    try:
        numeric_value = float(raw_value)
    except ValueError as error:
        raise EvaluationError(f"{location}: {field} must be numeric") from error
    if not math.isfinite(numeric_value) or not numeric_value.is_integer():
        raise EvaluationError(f"{location}: {field} must be an integer")
    value = int(numeric_value)
    if value < 1:
        raise EvaluationError(f"{location}: {field} must be >= 1")
    return value


def _parse_finite_float(raw_value: str, field: str, location: str) -> float:
    try:
        value = float(raw_value)
    except ValueError as error:
        raise EvaluationError(f"{location}: {field} must be numeric") from error
    if not math.isfinite(value):
        raise EvaluationError(f"{location}: {field} must be finite")
    return value


def read_sequence_info(sequence_dir: Path) -> SequenceInfo:
    sequence_dir = Path(sequence_dir)
    if not sequence_dir.is_dir():
        raise EvaluationError(f"MOT sequence directory not found: {sequence_dir}")

    info_path = sequence_dir / "seqinfo.ini"
    ground_truth_path = sequence_dir / "gt" / "gt.txt"
    if not info_path.is_file():
        raise EvaluationError(f"MOT sequence metadata not found: {info_path}")
    if not ground_truth_path.is_file():
        raise EvaluationError(f"MOT ground truth not found: {ground_truth_path}")

    parser = configparser.ConfigParser()
    try:
        with info_path.open("r", encoding="utf-8") as stream:
            parser.read_file(stream)
    except (OSError, configparser.Error) as error:
        raise EvaluationError(f"failed to read MOT sequence metadata: {info_path}") from error

    if "Sequence" not in parser:
        raise EvaluationError(f"{info_path}: missing [Sequence] section")
    section = parser["Sequence"]
    name = section.get("name", fallback=sequence_dir.name).strip()
    if not name:
        raise EvaluationError(f"{info_path}: sequence name must not be empty")
    try:
        length = section.getint("seqLength")
    except (ValueError, configparser.Error) as error:
        raise EvaluationError(f"{info_path}: seqLength must be an integer") from error
    if length is None or length < 1:
        raise EvaluationError(f"{info_path}: seqLength must be >= 1")

    return SequenceInfo(name=name, length=length, ground_truth_path=ground_truth_path)


def read_mot_rows(
    path: Path, sequence_length: int, *, ground_truth: bool
) -> list[MotRow]:
    path = Path(path)
    if not path.is_file():
        file_kind = "ground truth" if ground_truth else "result"
        raise EvaluationError(f"MOT {file_kind} not found: {path}")

    rows: list[MotRow] = []
    seen_keys: set[tuple[int, int]] = set()
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise EvaluationError(f"failed to read MOT file: {path}") from error

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        location = f"{path}:{line_number}"
        columns = [column.strip() for column in line.split(",")]
        if len(columns) != 10:
            raise EvaluationError(
                f"{location}: expected 10 MOT columns, got {len(columns)}"
            )

        frame_id = _parse_positive_integer(columns[0], "frame ID", location)
        track_id = _parse_positive_integer(columns[1], "track ID", location)
        if frame_id > sequence_length:
            raise EvaluationError(
                f"{location}: frame ID {frame_id} exceeds seqLength {sequence_length}"
            )

        x = _parse_finite_float(columns[2], "bbox x", location)
        y = _parse_finite_float(columns[3], "bbox y", location)
        width = _parse_finite_float(columns[4], "bbox width", location)
        height = _parse_finite_float(columns[5], "bbox height", location)
        if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
            raise EvaluationError(
                f"{location}: bbox requires x/y >= 0 and width/height > 0"
            )

        if ground_truth:
            mark = _parse_finite_float(columns[6], "ground-truth mark", location)
            if mark <= 0.0:
                continue

        key = (frame_id, track_id)
        if key in seen_keys:
            raise EvaluationError(
                f"{location}: duplicate frame/track pair {frame_id}/{track_id}"
            )
        seen_keys.add(key)
        rows.append(MotRow(frame_id, track_id, (x, y, width, height)))

    return rows


def _box_iou_matrix(gt_boxes: Any, tracker_boxes: Any, np: Any) -> Any:
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=float)

    gt_left_top = gt_boxes[:, :2]
    gt_right_bottom = gt_left_top + gt_boxes[:, 2:]
    tracker_left_top = tracker_boxes[:, :2]
    tracker_right_bottom = tracker_left_top + tracker_boxes[:, 2:]

    intersection_left_top = np.maximum(
        gt_left_top[:, np.newaxis, :], tracker_left_top[np.newaxis, :, :]
    )
    intersection_right_bottom = np.minimum(
        gt_right_bottom[:, np.newaxis, :], tracker_right_bottom[np.newaxis, :, :]
    )
    intersection_size = np.maximum(
        0.0, intersection_right_bottom - intersection_left_top
    )
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    gt_area = gt_boxes[:, 2] * gt_boxes[:, 3]
    tracker_area = tracker_boxes[:, 2] * tracker_boxes[:, 3]
    union = gt_area[:, np.newaxis] + tracker_area[np.newaxis, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=float),
        where=union > 0.0,
    )


def build_trackeval_data(
    ground_truth_rows: Sequence[MotRow],
    tracker_rows: Sequence[MotRow],
    sequence_length: int,
    np: Any,
) -> dict[str, Any]:
    gt_id_map = {
        track_id: index
        for index, track_id in enumerate(sorted({row.track_id for row in ground_truth_rows}))
    }
    tracker_id_map = {
        track_id: index
        for index, track_id in enumerate(sorted({row.track_id for row in tracker_rows}))
    }

    gt_by_frame: dict[int, list[MotRow]] = {}
    tracker_by_frame: dict[int, list[MotRow]] = {}
    for row in ground_truth_rows:
        gt_by_frame.setdefault(row.frame_id, []).append(row)
    for row in tracker_rows:
        tracker_by_frame.setdefault(row.frame_id, []).append(row)

    gt_ids: list[Any] = []
    tracker_ids: list[Any] = []
    similarity_scores: list[Any] = []
    for frame_id in range(1, sequence_length + 1):
        frame_gt = gt_by_frame.get(frame_id, [])
        frame_tracker = tracker_by_frame.get(frame_id, [])
        frame_gt_ids = np.asarray(
            [gt_id_map[row.track_id] for row in frame_gt], dtype=np.int64
        )
        frame_tracker_ids = np.asarray(
            [tracker_id_map[row.track_id] for row in frame_tracker], dtype=np.int64
        )
        gt_boxes = np.asarray(
            [row.bbox_xywh for row in frame_gt], dtype=float
        ).reshape((-1, 4))
        tracker_boxes = np.asarray(
            [row.bbox_xywh for row in frame_tracker], dtype=float
        ).reshape((-1, 4))
        gt_ids.append(frame_gt_ids)
        tracker_ids.append(frame_tracker_ids)
        similarity_scores.append(_box_iou_matrix(gt_boxes, tracker_boxes, np))

    return {
        "num_timesteps": sequence_length,
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
        "num_gt_dets": len(ground_truth_rows),
        "num_tracker_dets": len(tracker_rows),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarity_scores,
    }


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(value)


def evaluate_tracking(
    sequence_dir: Path,
    tracker_result_path: Path,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not math.isfinite(iou_threshold) or not 0.0 < iou_threshold <= 1.0:
        raise EvaluationError("IoU threshold must be in the range (0, 1]")

    np, trackeval = _load_dependencies()
    sequence_info = read_sequence_info(Path(sequence_dir))
    ground_truth_rows = read_mot_rows(
        sequence_info.ground_truth_path,
        sequence_info.length,
        ground_truth=True,
    )
    tracker_rows = read_mot_rows(
        Path(tracker_result_path), sequence_info.length, ground_truth=False
    )
    data = build_trackeval_data(
        ground_truth_rows, tracker_rows, sequence_info.length, np
    )

    hota_result = trackeval.metrics.HOTA().eval_sequence(data)
    clear_result = trackeval.metrics.CLEAR(
        {"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}
    ).eval_sequence(data)
    identity_result = trackeval.metrics.Identity(
        {"THRESHOLD": iou_threshold, "PRINT_CONFIG": False}
    ).eval_sequence(data)

    return {
        "schema_version": "1.0",
        "tool": {"name": "TrackEval", "version": version("trackeval")},
        "evaluation_scope": "tracking_only_from_ground_truth_detections",
        "inputs": {
            "sequence_dir": str(sequence_dir),
            "tracker_results": str(tracker_result_path),
        },
        "sequence": {
            "name": sequence_info.name,
            "length": sequence_info.length,
            "ground_truth_detections": len(ground_truth_rows),
            "tracker_detections": len(tracker_rows),
            "ground_truth_ids": data["num_gt_ids"],
            "tracker_ids": data["num_tracker_ids"],
        },
        "clear_identity_iou_threshold": iou_threshold,
        "metric_units": "ratio_0_to_1",
        "metrics": {
            "HOTA": {
                "HOTA": _as_float(np.mean(hota_result["HOTA"])),
                "DetA": _as_float(np.mean(hota_result["DetA"])),
                "AssA": _as_float(np.mean(hota_result["AssA"])),
                "LocA": _as_float(np.mean(hota_result["LocA"])),
            },
            "CLEAR": {
                "MOTA": _as_float(clear_result["MOTA"]),
                "MOTP": _as_float(clear_result["MOTP"]),
                "recall": _as_float(clear_result["CLR_Re"]),
                "precision": _as_float(clear_result["CLR_Pr"]),
                "true_positives": _as_int(clear_result["CLR_TP"]),
                "false_negatives": _as_int(clear_result["CLR_FN"]),
                "false_positives": _as_int(clear_result["CLR_FP"]),
                "identity_switches": _as_int(clear_result["IDSW"]),
                "fragmentations": _as_int(clear_result["Frag"]),
                "mostly_tracked": _as_int(clear_result["MT"]),
                "partially_tracked": _as_int(clear_result["PT"]),
                "mostly_lost": _as_int(clear_result["ML"]),
            },
            "Identity": {
                "IDF1": _as_float(identity_result["IDF1"]),
                "IDP": _as_float(identity_result["IDP"]),
                "IDR": _as_float(identity_result["IDR"]),
                "IDTP": _as_int(identity_result["IDTP"]),
                "IDFP": _as_int(identity_result["IDFP"]),
                "IDFN": _as_int(identity_result["IDFN"]),
            },
        },
        "limitations": [
            "Ground-truth track IDs are derived by the dataset conversion "
            "pipeline, not supplied by the source labels.",
            "Metrics evaluate tracking association from replayed ground-truth "
            "detections, not end-to-end detector performance.",
            "No pass/fail acceptance threshold is defined.",
        ],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one MOT sequence with TrackEval HOTA/CLEAR/Identity metrics."
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        required=True,
        help="MOT sequence containing seqinfo.ini and gt/gt.txt",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        required=True,
        help="MOTChallenge 10-column tracker result file",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="CLEAR and Identity matching IoU threshold (default: 0.5)",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    options = parser.parse_args(arguments)
    try:
        result = evaluate_tracking(
            options.sequence_dir, options.tracks, options.iou_threshold
        )
    except EvaluationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

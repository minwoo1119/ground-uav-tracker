"""Export the project master annotations and videos as a YOLOX-ready COCO dataset.

The annotation conversion is standard-library only. OpenCV is imported lazily
only when real video frames are extracted, so validation tests can use a small
in-memory frame source without adding dependencies to the core test suite.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple

from mcos_parser import CLASS_TO_ID


SPLIT_NAMES = ("train", "val", "test")
COCO_IMAGE_DIRS = {
    "train": "train2017",
    "val": "val2017",
    "test": "test2017",
}


class CocoExportError(RuntimeError):
    """Raised when source data cannot be exported without ambiguity."""


class FrameSource(Protocol):
    """Minimal video reader contract used by :class:`CocoExporter`."""

    width: int
    height: int
    frame_count: Optional[int]

    def read(self) -> Optional[Any]: ...

    def frame_size(self, frame: Any) -> Tuple[int, int]: ...

    def write_jpeg(self, path: Path, frame: Any, quality: int) -> None: ...

    def close(self) -> None: ...


FrameSourceFactory = Callable[[Path], FrameSource]


class OpenCvFrameSource:
    """OpenCV-backed frame source loaded only for real dataset export."""

    def __init__(self, path: Path) -> None:
        try:
            import cv2  # type: ignore
        except ImportError as error:
            raise CocoExportError(
                "OpenCV is required to extract video frames; install the training "
                "requirements in an isolated environment"
            ) from error

        self._cv2 = cv2
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise CocoExportError(f"failed to open video: {path}")

        self.width = int(round(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height = int(round(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        reported_count = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.frame_count = reported_count if reported_count > 0 else None

    def read(self) -> Optional[Any]:
        ok, frame = self._capture.read()
        return frame if ok else None

    @staticmethod
    def frame_size(frame: Any) -> Tuple[int, int]:
        height, width = frame.shape[:2]
        return int(width), int(height)

    def write_jpeg(self, path: Path, frame: Any, quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = self._cv2.imwrite(
            str(path), frame, [int(self._cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            raise CocoExportError(f"failed to write extracted frame: {path}")

    def close(self) -> None:
        self._capture.release()


@dataclass(frozen=True)
class ExportSummary:
    sequences: Dict[str, int]
    images: Dict[str, int]
    annotations: Dict[str, int]
    clipped_boxes: Dict[str, int]

    def to_dict(self) -> Dict[str, Dict[str, int]]:
        return {
            "sequences": self.sequences,
            "images": self.images,
            "annotations": self.annotations,
            "clipped_boxes": self.clipped_boxes,
        }


def _load_json_object(path: Path, description: str) -> Dict[str, Any]:
    if not path.is_file():
        raise CocoExportError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CocoExportError(f"failed to read {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise CocoExportError(f"{description} must contain a JSON object: {path}")
    return value


def _validate_categories(master_data: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw_classes = master_data.get("classes")
    if not isinstance(raw_classes, dict):
        raise CocoExportError("master JSON is missing the 'classes' object")

    expected = {str(class_id): name for name, class_id in CLASS_TO_ID.items()}
    if raw_classes != expected:
        raise CocoExportError(
            f"master class mapping mismatch: expected {expected}, got {raw_classes}"
        )

    return [
        {
            "id": class_id + 1,
            "name": name,
            "supercategory": "airborne_object",
            "project_class_id": class_id,
        }
        for name, class_id in sorted(CLASS_TO_ID.items(), key=lambda item: item[1])
    ]


def _normalize_splits(split_data: Mapping[str, Any]) -> Dict[str, List[str]]:
    raw_splits = split_data.get("splits", split_data)
    if not isinstance(raw_splits, dict):
        raise CocoExportError("split JSON is missing the 'splits' object")

    normalized: Dict[str, List[str]] = {}
    owner: Dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        raw_names = raw_splits.get(split_name)
        if not isinstance(raw_names, list) or not all(
            isinstance(name, str) and name for name in raw_names
        ):
            raise CocoExportError(
                f"split '{split_name}' must be a list of non-empty sequence names"
            )
        normalized[split_name] = list(raw_names)
        for sequence_name in raw_names:
            previous = owner.get(sequence_name)
            if previous is not None:
                raise CocoExportError(
                    f"sequence '{sequence_name}' appears in both '{previous}' and "
                    f"'{split_name}' splits"
                )
            owner[sequence_name] = split_name
    return normalized


def _build_video_index(videos_dir: Path) -> Dict[str, Path]:
    if not videos_dir.is_dir():
        raise CocoExportError(f"video directory not found: {videos_dir}")

    index: Dict[str, Path] = {}
    for path in sorted(videos_dir.rglob("*.mp4")):
        key = path.stem
        if key in index:
            raise CocoExportError(
                f"multiple videos have the same sequence name '{key}': "
                f"{index[key]} and {path}"
            )
        index[key] = path
    return index


def _clip_bbox(
    raw_bbox: Any,
    width: int,
    height: int,
    location: str,
) -> Tuple[List[float], bool]:
    if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
        raise CocoExportError(f"{location}: bbox_xywh must contain four numbers")
    if not all(isinstance(value, (int, float)) for value in raw_bbox):
        raise CocoExportError(f"{location}: bbox_xywh contains a non-numeric value")

    x, y, box_width, box_height = (float(value) for value in raw_bbox)
    if box_width <= 0.0 or box_height <= 0.0:
        raise CocoExportError(f"{location}: bbox width and height must be positive")

    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(width), x + box_width)
    y2 = min(float(height), y + box_height)
    if x2 <= x1 or y2 <= y1:
        raise CocoExportError(f"{location}: bbox lies outside the image")

    clipped = x1 != x or y1 != y or x2 != x + box_width or y2 != y + box_height
    return [x1, y1, x2 - x1, y2 - y1], clipped


class CocoExporter:
    """Create a deterministic COCO directory without publishing partial output."""

    def __init__(
        self,
        master_data: Mapping[str, Any],
        split_data: Mapping[str, Any],
        videos_dir: Path,
        output_dir: Path,
        jpeg_quality: int = 95,
        frame_source_factory: FrameSourceFactory = OpenCvFrameSource,
    ) -> None:
        if jpeg_quality < 1 or jpeg_quality > 100:
            raise CocoExportError("JPEG quality must be between 1 and 100")
        self.master_data = master_data
        self.splits = _normalize_splits(split_data)
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.jpeg_quality = jpeg_quality
        self.frame_source_factory = frame_source_factory
        self.categories = _validate_categories(master_data)

    def export(self) -> ExportSummary:
        if self.output_dir.exists():
            raise CocoExportError(
                f"output directory already exists; refusing to overwrite: {self.output_dir}"
            )

        raw_sequences = self.master_data.get("sequences")
        if not isinstance(raw_sequences, dict):
            raise CocoExportError("master JSON is missing the 'sequences' object")
        video_index = _build_video_index(self.videos_dir)

        for split_name, sequence_names in self.splits.items():
            for sequence_name in sequence_names:
                if sequence_name not in raw_sequences:
                    raise CocoExportError(
                        f"split '{split_name}' references unknown sequence '{sequence_name}'"
                    )
                if sequence_name not in video_index:
                    raise CocoExportError(
                        f"video for sequence '{sequence_name}' was not found under "
                        f"{self.videos_dir}"
                    )

        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=self.output_dir.parent, prefix=f".{self.output_dir.name}-staging-"
        ) as temporary_root:
            payload = Path(temporary_root) / "payload"
            payload.mkdir()
            summary = self._write_payload(payload, raw_sequences, video_index)
            payload.replace(self.output_dir)
        return summary

    def _write_payload(
        self,
        payload: Path,
        raw_sequences: Mapping[str, Any],
        video_index: Mapping[str, Path],
    ) -> ExportSummary:
        sequence_counts: Dict[str, int] = {}
        image_counts: Dict[str, int] = {}
        annotation_counts: Dict[str, int] = {}
        clipped_counts: Dict[str, int] = {}

        annotations_dir = payload / "annotations"
        annotations_dir.mkdir()

        for split_name in SPLIT_NAMES:
            coco_images: List[Dict[str, Any]] = []
            coco_annotations: List[Dict[str, Any]] = []
            next_image_id = 1
            next_annotation_id = 1
            clipped_count = 0

            for sequence_name in self.splits[split_name]:
                sequence = raw_sequences[sequence_name]
                if not isinstance(sequence, dict):
                    raise CocoExportError(
                        f"sequence '{sequence_name}' must be a JSON object"
                    )
                image_delta, annotation_delta, clipped_delta = self._export_sequence(
                    payload=payload,
                    split_name=split_name,
                    sequence_name=sequence_name,
                    sequence=sequence,
                    video_path=video_index[sequence_name],
                    first_image_id=next_image_id,
                    first_annotation_id=next_annotation_id,
                    coco_images=coco_images,
                    coco_annotations=coco_annotations,
                )
                next_image_id += image_delta
                next_annotation_id += annotation_delta
                clipped_count += clipped_delta

            document = {
                "info": {
                    "description": "Ground UAV Tracker RGB dataset",
                    "source_annotation_schema": self.master_data.get(
                        "schema_version", "unknown"
                    ),
                },
                "licenses": [
                    {
                        "id": 1,
                        "name": "CC0-1.0",
                        "url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    }
                ],
                "categories": self.categories,
                "images": coco_images,
                "annotations": coco_annotations,
            }
            annotation_path = (
                annotations_dir / f"instances_{COCO_IMAGE_DIRS[split_name]}.json"
            )
            annotation_path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            sequence_counts[split_name] = len(self.splits[split_name])
            image_counts[split_name] = len(coco_images)
            annotation_counts[split_name] = len(coco_annotations)
            clipped_counts[split_name] = clipped_count

        summary = ExportSummary(
            sequences=sequence_counts,
            images=image_counts,
            annotations=annotation_counts,
            clipped_boxes=clipped_counts,
        )
        manifest = {
            "format": "COCO",
            "image_layout": (
                "<train2017|val2017|test2017>/<sequence>/"
                "frame_<zero-based-index>.jpg"
            ),
            "project_to_coco_category_id": {
                str(class_id): class_id + 1 for class_id in sorted(CLASS_TO_ID.values())
            },
            "summary": summary.to_dict(),
        }
        (payload / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return summary

    def _export_sequence(
        self,
        *,
        payload: Path,
        split_name: str,
        sequence_name: str,
        sequence: Mapping[str, Any],
        video_path: Path,
        first_image_id: int,
        first_annotation_id: int,
        coco_images: List[Dict[str, Any]],
        coco_annotations: List[Dict[str, Any]],
    ) -> Tuple[int, int, int]:
        width = sequence.get("image_width")
        height = sequence.get("image_height")
        frames = sequence.get("frames")
        if not isinstance(width, int) or width <= 0:
            raise CocoExportError(f"{sequence_name}: invalid image_width")
        if not isinstance(height, int) or height <= 0:
            raise CocoExportError(f"{sequence_name}: invalid image_height")
        if not isinstance(frames, list):
            raise CocoExportError(f"{sequence_name}: frames must be a list")

        declared_total = sequence.get("total_frames")
        if declared_total != len(frames):
            raise CocoExportError(
                f"{sequence_name}: total_frames={declared_total} but master contains "
                f"{len(frames)} frame records"
            )

        source = self.frame_source_factory(video_path)
        try:
            if source.width != width or source.height != height:
                raise CocoExportError(
                    f"{sequence_name}: video size {source.width}x{source.height} does not "
                    f"match master {width}x{height}"
                )
            if source.frame_count is not None and source.frame_count != len(frames):
                raise CocoExportError(
                    f"{sequence_name}: video reports {source.frame_count} frames but "
                    f"master contains {len(frames)}"
                )

            annotation_id = first_annotation_id
            clipped_count = 0
            for offset, frame_record in enumerate(frames):
                if not isinstance(frame_record, dict):
                    raise CocoExportError(
                        f"{sequence_name}: frame record {offset} must be an object"
                    )
                frame_index = frame_record.get("frame_index")
                if frame_index != offset:
                    raise CocoExportError(
                        f"{sequence_name}: expected frame_index {offset}, got {frame_index}"
                    )

                frame = source.read()
                if frame is None:
                    raise CocoExportError(
                        f"{sequence_name}: video ended before frame {frame_index}"
                    )
                actual_width, actual_height = source.frame_size(frame)
                if actual_width != width or actual_height != height:
                    raise CocoExportError(
                        f"{sequence_name}: frame {frame_index} size "
                        f"{actual_width}x{actual_height} does not match master "
                        f"{width}x{height}"
                    )

                image_id = first_image_id + offset
                relative_name = (
                    Path(sequence_name) / f"frame_{frame_index:06d}.jpg"
                ).as_posix()
                output_path = (
                    payload / COCO_IMAGE_DIRS[split_name] / Path(relative_name)
                )
                source.write_jpeg(output_path, frame, self.jpeg_quality)
                coco_images.append(
                    {
                        "id": image_id,
                        "file_name": relative_name,
                        "width": width,
                        "height": height,
                        "license": 1,
                        "sequence_name": sequence_name,
                        "frame_index": frame_index,
                    }
                )

                raw_annotations = frame_record.get("annotations")
                if not isinstance(raw_annotations, list):
                    raise CocoExportError(
                        f"{sequence_name} frame {frame_index}: annotations must be a list"
                    )
                for annotation_offset, annotation in enumerate(raw_annotations):
                    location = (
                        f"{sequence_name} frame {frame_index} annotation "
                        f"{annotation_offset}"
                    )
                    if not isinstance(annotation, dict):
                        raise CocoExportError(f"{location}: annotation must be an object")
                    class_id = annotation.get("class_id")
                    class_name = annotation.get("class_name")
                    if class_id not in CLASS_TO_ID.values():
                        raise CocoExportError(f"{location}: invalid class_id {class_id}")
                    expected_name = next(
                        name for name, value in CLASS_TO_ID.items() if value == class_id
                    )
                    if class_name != expected_name:
                        raise CocoExportError(
                            f"{location}: class_name '{class_name}' does not match "
                            f"class_id {class_id} ('{expected_name}')"
                        )
                    bbox, was_clipped = _clip_bbox(
                        annotation.get("bbox_xywh"), width, height, location
                    )
                    clipped_count += int(was_clipped)
                    coco_annotations.append(
                        {
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id) + 1,
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )
                    annotation_id += 1
        finally:
            source.close()

        return len(frames), annotation_id - first_annotation_id, clipped_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export master annotations and RGB videos to YOLOX-ready COCO"
    )
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    master_data = _load_json_object(args.master, "master JSON")
    split_data = _load_json_object(args.splits, "split JSON")
    summary = CocoExporter(
        master_data=master_data,
        split_data=split_data,
        videos_dir=args.videos_dir,
        output_dir=args.output_dir,
        jpeg_quality=args.jpeg_quality,
    ).export()
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except CocoExportError as error:
        raise SystemExit(f"error: {error}") from error

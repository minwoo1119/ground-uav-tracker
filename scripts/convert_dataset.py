"""Dataset Conversion and Export Pipeline for Ground-Based UAV Tracking.

This module provides an extensible pipeline centered around a Master JSON
intermediate representation. It converts MATLAB MCOS groundTruth annotations
into:
    1. Master JSON (full metadata, per-frame BBoxes, 4-class mapping)
    2. YOLO format (normalized center_x, center_y, w, h txt files + data.yaml)
    3. MOT Challenge format (gt.txt with frame_id, track_id, bbox + seqinfo.ini)
    4. Stratified video-level Train/Val/Test splits (70/15/15, sensor-aware)
    5. Dataset statistics and bounding-box size distribution report (analyze)

Usage:
    python scripts/convert_dataset.py build-master --labels-dir data/raw/Video_V --output data/processed/dataset_master.json
    python scripts/convert_dataset.py export-yolo --master data/processed/dataset_master.json --output-dir data/processed/yolo
    python scripts/convert_dataset.py export-mot --master data/processed/dataset_master.json --output-dir data/processed/mot
    python scripts/convert_dataset.py generate-splits --master data/processed/dataset_master.json --output-dir data/processed/splits
    python scripts/convert_dataset.py analyze --master data/processed/dataset_master.json --output data/processed/dataset_stats.json
    python scripts/convert_dataset.py all --labels-dir data/raw/Video_V --output-dir data/processed
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from mcos_parser import (
    CLASS_COLUMNS,
    CLASS_TO_ID,
    ID_TO_CLASS,
    BoundingBox,
    McosLabelParser,
    SequenceAnnotation,
)
from metadata_parser import MetadataCatalog, SequenceMetadata
from video_probe import probe_mp4

# Fallback geometry per sensor, used only when the matching .mp4 is not
# available for probing. Observed from the MP4 headers of V_DRONE_001
# (640x512, 30 fps) and IR_BIRD_001 (320x256, 30 fps); other videos are
# unverified until probed, which the Master JSON records in "video_info_source".
SENSOR_DEFAULTS: Dict[str, Dict[str, float]] = {
    "V": {"width": 640, "height": 512, "fps": 30.0},
    "IR": {"width": 320, "height": 256, "fps": 30.0},
}


def calculate_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Calculate IoU between two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    iw = max(0.0, xi2 - xi1)
    ih = max(0.0, yi2 - yi1)
    inter_area = iw * ih
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


class MasterDatasetBuilder:
    """Constructs the canonical Master JSON from .mat files and metadata."""

    def __init__(
        self,
        labels_dir: Path,
        metadata_file: Optional[Path] = None,
        videos_dir: Optional[Path] = None,
    ) -> None:
        self.labels_dir = labels_dir
        self.metadata_file = metadata_file
        self.videos_dir = videos_dir
        self.parser = McosLabelParser()
        self.catalog = (
            MetadataCatalog.from_xlsx(metadata_file)
            if metadata_file and metadata_file.is_file()
            else MetadataCatalog()
        )
        self.warnings: List[str] = []

    def _video_info(self, seq_name: str, sensor: str, label_frames: int) -> Dict[str, Any]:
        """Probe the matching .mp4 when available, else fall back to sensor defaults."""
        if self.videos_dir is not None:
            candidates = list(self.videos_dir.rglob(f"{seq_name}.mp4"))
            if candidates:
                info = probe_mp4(candidates[0])
                if info.frame_count != label_frames:
                    self.warnings.append(
                        f"{seq_name}: video has {info.frame_count} frames but labels have {label_frames}"
                    )
                return {
                    "fps": info.fps,
                    "image_width": info.width,
                    "image_height": info.height,
                    "video_frame_count": info.frame_count,
                    "video_info_source": "mp4_header",
                }
        if sensor not in SENSOR_DEFAULTS:
            raise ValueError(
                f"{seq_name}: unknown sensor '{sensor}' and no video to probe; "
                "pass --videos-dir so resolution can be read from the MP4"
            )
        d = SENSOR_DEFAULTS[sensor]
        return {
            "fps": d["fps"],
            "image_width": int(d["width"]),
            "image_height": int(d["height"]),
            "video_frame_count": None,
            "video_info_source": "sensor_default_unverified",
        }

    def build(self) -> Dict[str, Any]:
        mat_files = sorted(self.labels_dir.glob("*_LABELS.mat"))
        if not mat_files:
            # Also search recursively
            mat_files = sorted(self.labels_dir.rglob("*_LABELS.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No *_LABELS.mat files found in {self.labels_dir}"
            )

        sequences_data: Dict[str, Any] = {}
        total_frames = 0
        total_boxes = {cls_name: 0 for cls_name in CLASS_TO_ID}

        for mat_path in mat_files:
            seq_annotation = self.parser.parse_file(mat_path)
            seq_name = seq_annotation.sequence_name
            meta = self.catalog.get(seq_name)
            video = self._video_info(seq_name, meta.sensor, seq_annotation.total_frames)

            # Assign simple track IDs across frames
            frames_with_tracks = self._assign_track_ids(seq_annotation, video["fps"])

            seq_box_counts = {cls_name: 0 for cls_name in CLASS_TO_ID}
            for frame in frames_with_tracks:
                for ann in frame["annotations"]:
                    seq_box_counts[ann["class_name"]] += 1
                    total_boxes[ann["class_name"]] += 1

            total_frames += seq_annotation.total_frames
            sequences_data[seq_name] = {
                "sequence_name": seq_name,
                "sensor": meta.sensor,
                "primary_class": meta.primary_class,
                "number": meta.number,
                "distance_bin": meta.distance_bin,
                "drone_type": meta.drone_type,
                "source": meta.source,
                "is_multi_drone": meta.is_multi_drone,
                "label_file": str(mat_path.name),
                "total_frames": seq_annotation.total_frames,
                **video,
                "class_box_counts": seq_box_counts,
                "frames": frames_with_tracks,
            }

        master_doc = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "classes": {str(v): k for k, v in CLASS_TO_ID.items()},
            "statistics": {
                "total_sequences": len(sequences_data),
                "total_frames": total_frames,
                "class_box_counts": total_boxes,
            },
            "warnings": list(self.warnings),
            "sequences": sequences_data,
        }
        return master_doc

    def _assign_track_ids(
        self, seq: SequenceAnnotation, fps: Optional[float]
    ) -> List[Dict[str, Any]]:
        """Assign track IDs to bboxes (using simple greedy IoU association).

        The source labels carry no identities, so these IDs are derived:
        per class, each previous track greedily takes the current box with the
        highest IoU (> 0.05). A frame without boxes of a class ends all its
        tracks, so a one-frame label gap starts new IDs.
        """
        active_tracks: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {
            cls: {} for cls in CLASS_TO_ID
        }
        next_track_id: Dict[str, int] = {cls: 1 for cls in CLASS_TO_ID}
        frames_out: List[Dict[str, Any]] = []

        for frame in seq.frames:
            frame_anns: List[Dict[str, Any]] = []
            sec = round(frame.frame_index / fps, 4) if fps else None

            for cls_name, boxes in frame.boxes_by_class.items():
                if not boxes:
                    active_tracks[cls_name] = {}
                    continue

                cls_id = CLASS_TO_ID[cls_name]
                new_active: Dict[int, Tuple[float, float, float, float]] = {}

                # If single target sequence, always use track ID 1
                if len(boxes) == 1 and not active_tracks[cls_name]:
                    tid = next_track_id[cls_name]
                    b_tuple = boxes[0].to_tuple()
                    new_active[tid] = b_tuple
                    frame_anns.append(
                        {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "track_id": tid,
                            "bbox_xywh": list(b_tuple),
                        }
                    )
                else:
                    # Match with active tracks via IoU
                    unmatched_boxes = [b.to_tuple() for b in boxes]
                    prev_tracks = list(active_tracks[cls_name].items())

                    for tid, prev_box in prev_tracks:
                        if not unmatched_boxes:
                            break
                        best_iou = 0.0
                        best_idx = -1
                        for i, cur_box in enumerate(unmatched_boxes):
                            iou = calculate_iou(prev_box, cur_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_idx = i
                        if best_iou > 0.05 and best_idx >= 0:
                            matched_box = unmatched_boxes.pop(best_idx)
                            new_active[tid] = matched_box
                            frame_anns.append(
                                {
                                    "class_id": cls_id,
                                    "class_name": cls_name,
                                    "track_id": tid,
                                    "bbox_xywh": list(matched_box),
                                }
                            )

                    # Any remaining boxes get fresh track IDs
                    for leftover_box in unmatched_boxes:
                        tid = next_track_id[cls_name]
                        next_track_id[cls_name] += 1
                        new_active[tid] = leftover_box
                        frame_anns.append(
                            {
                                "class_id": cls_id,
                                "class_name": cls_name,
                                "track_id": tid,
                                "bbox_xywh": list(leftover_box),
                            }
                        )

                active_tracks[cls_name] = new_active

            frames_out.append(
                {
                    "frame_index": frame.frame_index,
                    "timestamp_sec": sec,
                    "annotations": frame_anns,
                }
            )

        return frames_out


class YoloExporter:
    """Exports Master JSON annotations to YOLO format."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def export(
        self,
        master_data: Dict[str, Any],
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        split_catalog: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        labels_base = self.output_dir / "labels"
        labels_base.mkdir(parents=True, exist_ok=True)

        for seq_name, seq_info in master_data["sequences"].items():
            w = image_width or seq_info.get("image_width")
            h = image_height or seq_info.get("image_height")
            if not w or not h:
                raise ValueError(
                    f"{seq_name}: image size unknown; rebuild the Master JSON or pass --image-width/--image-height"
                )

            # Determine destination folder (by split if provided, else sequence)
            split_name = ""
            if split_catalog:
                for s_name, seqs in split_catalog.items():
                    if seq_name in seqs:
                        split_name = s_name
                        break
            dest_dir = (
                labels_base / split_name if split_name else labels_base / seq_name
            )
            dest_dir.mkdir(parents=True, exist_ok=True)

            for frame in seq_info["frames"]:
                f_idx = frame["frame_index"]
                txt_filename = (
                    f"{seq_name}_{f_idx:06d}.txt"
                    if split_name
                    else f"frame_{f_idx:06d}.txt"
                )
                txt_path = dest_dir / txt_filename

                lines = []
                for ann in frame["annotations"]:
                    cid = ann["class_id"]
                    x, y, bw, bh = ann["bbox_xywh"]
                    bbox = BoundingBox(x=x, y=y, width=bw, height=bh)
                    cx, cy, nw, nh = bbox.to_yolo(w, h)
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

                txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        # Write data.yaml descriptor
        yaml_path = self.output_dir / "data.yaml"
        yaml_content = [
            "# Ground UAV Tracker - 4-Class Dataset Configuration",
            "nc: 4",
            "names:",
            "  0: AIRPLANE",
            "  1: BIRD",
            "  2: DRONE",
            "  3: HELICOPTER",
        ]
        if split_catalog:
            yaml_content.extend(
                [
                    "train: images/train",
                    "val: images/val",
                    "test: images/test",
                ]
            )
        yaml_path.write_text("\n".join(yaml_content) + "\n", encoding="utf-8")


class MotExporter:
    """Exports Master JSON annotations to MOT Challenge format."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def export(self, master_data: Dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for seq_name, seq_info in master_data["sequences"].items():
            seq_dir = self.output_dir / seq_name
            gt_dir = seq_dir / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)

            gt_lines = []
            for frame in seq_info["frames"]:
                # MOT Challenge frame IDs are 1-based
                f_id = frame["frame_index"] + 1
                for ann in frame["annotations"]:
                    t_id = ann["track_id"]
                    x, y, w, h = ann["bbox_xywh"]
                    # Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
                    gt_lines.append(
                        f"{f_id},{t_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1"
                    )

            (gt_dir / "gt.txt").write_text(
                "\n".join(gt_lines) + ("\n" if gt_lines else ""), encoding="utf-8"
            )

            # Write seqinfo.ini
            fps = seq_info.get("fps")
            ini_lines = [
                "[Sequence]",
                f"name={seq_name}",
                "imDir=img1",
                f"frameRate={round(fps) if fps else 'unknown'}",
                f"seqLength={seq_info.get('total_frames', len(seq_info['frames']))}",
                f"imWidth={seq_info['image_width']}",
                f"imHeight={seq_info['image_height']}",
                "imExt=.jpg",
            ]
            (seq_dir / "seqinfo.ini").write_text(
                "\n".join(ini_lines) + "\n", encoding="utf-8"
            )


class SplitGenerator:
    """Generates video-level stratified Train/Val/Test splits."""

    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        keep_multi_in_test: bool = True,
        sensor_filter: Optional[str] = None,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {train_ratio}+{val_ratio}+{test_ratio}={total})"
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.keep_multi_in_test = keep_multi_in_test
        self.sensor_filter = sensor_filter.upper() if sensor_filter else None

    def generate(
        self, master_data: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        rng = random.Random(self.seed)

        sequences = list(master_data["sequences"].values())
        if self.sensor_filter:
            sequences = [
                s for s in sequences if s.get("sensor", "").upper() == self.sensor_filter
            ]

        train_seqs: List[str] = []
        val_seqs: List[str] = []
        test_seqs: List[str] = []

        # Group by stratum: (sensor, primary_class, distance_bin)
        strata: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for s in sequences:
            if self.keep_multi_in_test and s.get("is_multi_drone", False):
                test_seqs.append(s["sequence_name"])
                continue
            key = (
                s.get("sensor", "UNKNOWN"),
                s.get("primary_class", "UNKNOWN"),
                s.get("distance_bin", "UNKNOWN"),
            )
            strata.setdefault(key, []).append(s)

        for key in sorted(strata.keys()):
            group = strata[key]
            rng.shuffle(group)
            n = len(group)
            n_train = int(round(n * self.train_ratio))
            n_val = int(round(n * self.val_ratio))
            # Remaining to test
            n_test = n - n_train - n_val
            if n_test < 0:
                n_train = max(0, n_train + n_test)
                n_test = 0

            train_seqs.extend(s["sequence_name"] for s in group[:n_train])
            val_seqs.extend(s["sequence_name"] for s in group[n_train : n_train + n_val])
            test_seqs.extend(s["sequence_name"] for s in group[n_train + n_val :])

        train_seqs.sort()
        val_seqs.sort()
        test_seqs.sort()

        return {
            "train": train_seqs,
            "val": val_seqs,
            "test": test_seqs,
        }

    def save_splits(
        self, splits: Dict[str, List[str]], output_dir: Path
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, seqs in splits.items():
            (output_dir / f"{name}_videos.txt").write_text(
                "\n".join(seqs) + "\n", encoding="utf-8"
            )

        summary = {
            "sensor_filter": self.sensor_filter,
            "stratification_axes": ["sensor", "primary_class", "distance_bin"],
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "counts": {k: len(v) for k, v in splits.items()},
            "splits": splits,
        }
        (output_dir / "splits.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def _quantile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return round(sorted_vals[0], 2)
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return round(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac, 2)


class DatasetAnalyzer:
    """Computes sequence, frame, and bounding-box size distributions from Master JSON."""

    TINY_AREA = 16.0 * 16.0       # 256 px^2
    SMALL_AREA = 32.0 * 32.0      # 1024 px^2 (COCO Small threshold)
    MEDIUM_AREA = 96.0 * 96.0     # 9216 px^2 (COCO Medium threshold)

    def analyze(self, master_data: Dict[str, Any]) -> Dict[str, Any]:
        sequences = master_data.get("sequences", {})
        by_class: Dict[str, Dict[str, Any]] = {}
        for cls_name in CLASS_TO_ID:
            by_class[cls_name] = {
                "sequence_count": 0,
                "sequence_frames": 0,
                "frames_with_class": 0,
                "empty_frames_in_primary_seqs": 0,
                "multi_object_frames_in_primary_seqs": 0,
                "box_count": 0,
                "widths": [],
                "heights": [],
                "areas": [],
                "size_buckets": {
                    "tiny_lt_16x16": 0,
                    "small_16x16_to_32x32": 0,
                    "coco_small_lt_32x32": 0,
                    "coco_medium_32x32_to_96x96": 0,
                    "coco_large_ge_96x96": 0,
                },
            }

        total_frames = 0
        total_empty_frames = 0
        total_multi_box_frames = 0
        all_areas: List[float] = []
        all_widths: List[float] = []
        all_heights: List[float] = []
        overall_buckets = {
            "tiny_lt_16x16": 0,
            "small_16x16_to_32x32": 0,
            "coco_small_lt_32x32": 0,
            "coco_medium_32x32_to_96x96": 0,
            "coco_large_ge_96x96": 0,
        }

        for seq in sequences.values():
            p_cls = seq.get("primary_class", "UNKNOWN")
            n_frames = seq.get("total_frames", len(seq.get("frames", [])))
            total_frames += n_frames
            if p_cls in by_class:
                by_class[p_cls]["sequence_count"] += 1
                by_class[p_cls]["sequence_frames"] += n_frames

            for frame in seq.get("frames", []):
                anns = frame.get("annotations", [])
                if not anns:
                    total_empty_frames += 1
                    if p_cls in by_class:
                        by_class[p_cls]["empty_frames_in_primary_seqs"] += 1
                elif len(anns) >= 2:
                    total_multi_box_frames += 1
                    if p_cls in by_class:
                        by_class[p_cls]["multi_object_frames_in_primary_seqs"] += 1

                present_classes: Set[str] = set()
                for ann in anns:
                    c_name = ann["class_name"]
                    present_classes.add(c_name)
                    _, _, w, h = ann["bbox_xywh"]
                    area = w * h
                    all_widths.append(w)
                    all_heights.append(h)
                    all_areas.append(area)

                    c_entry = by_class[c_name]
                    c_entry["box_count"] += 1
                    c_entry["widths"].append(w)
                    c_entry["heights"].append(h)
                    c_entry["areas"].append(area)

                    for buckets in (c_entry["size_buckets"], overall_buckets):
                        if area < self.TINY_AREA:
                            buckets["tiny_lt_16x16"] += 1
                            buckets["coco_small_lt_32x32"] += 1
                        elif area < self.SMALL_AREA:
                            buckets["small_16x16_to_32x32"] += 1
                            buckets["coco_small_lt_32x32"] += 1
                        elif area < self.MEDIUM_AREA:
                            buckets["coco_medium_32x32_to_96x96"] += 1
                        else:
                            buckets["coco_large_ge_96x96"] += 1

                for c_name in present_classes:
                    if c_name in by_class:
                        by_class[c_name]["frames_with_class"] += 1

        class_summaries: Dict[str, Any] = {}
        for cls_name, c_entry in by_class.items():
            ws = sorted(c_entry["widths"])
            hs = sorted(c_entry["heights"])
            ars = sorted(c_entry["areas"])
            bc = c_entry["box_count"]
            sb = c_entry["size_buckets"]
            class_summaries[cls_name] = {
                "sequence_count": c_entry["sequence_count"],
                "sequence_frames": c_entry["sequence_frames"],
                "frames_with_class": c_entry["frames_with_class"],
                "empty_frames_in_primary_seqs": c_entry["empty_frames_in_primary_seqs"],
                "multi_object_frames_in_primary_seqs": c_entry["multi_object_frames_in_primary_seqs"],
                "box_count": bc,
                "width_px": {
                    "min": _quantile(ws, 0.0),
                    "p25": _quantile(ws, 0.25),
                    "median": _quantile(ws, 0.50),
                    "p75": _quantile(ws, 0.75),
                    "max": _quantile(ws, 1.0),
                    "mean": round(sum(ws) / len(ws), 2) if ws else None,
                },
                "height_px": {
                    "min": _quantile(hs, 0.0),
                    "p25": _quantile(hs, 0.25),
                    "median": _quantile(hs, 0.50),
                    "p75": _quantile(hs, 0.75),
                    "max": _quantile(hs, 1.0),
                    "mean": round(sum(hs) / len(hs), 2) if hs else None,
                },
                "area_px2": {
                    "min": _quantile(ars, 0.0),
                    "p25": _quantile(ars, 0.25),
                    "median": _quantile(ars, 0.50),
                    "p75": _quantile(ars, 0.75),
                    "max": _quantile(ars, 1.0),
                    "mean": round(sum(ars) / len(ars), 2) if ars else None,
                },
                "size_buckets": sb,
                "size_ratios": {
                    k: round(v / bc, 4) if bc > 0 else 0.0 for k, v in sb.items()
                },
            }

        all_ws = sorted(all_widths)
        all_hs = sorted(all_heights)
        all_ars = sorted(all_areas)
        total_boxes = len(all_areas)
        return {
            "total_sequences": len(sequences),
            "total_frames": total_frames,
            "total_empty_frames": total_empty_frames,
            "total_multi_box_frames": total_multi_box_frames,
            "total_boxes": total_boxes,
            "overall_width_px": {
                "min": _quantile(all_ws, 0.0),
                "p25": _quantile(all_ws, 0.25),
                "median": _quantile(all_ws, 0.50),
                "p75": _quantile(all_ws, 0.75),
                "max": _quantile(all_ws, 1.0),
                "mean": round(sum(all_ws) / len(all_ws), 2) if all_ws else None,
            },
            "overall_height_px": {
                "min": _quantile(all_hs, 0.0),
                "p25": _quantile(all_hs, 0.25),
                "median": _quantile(all_hs, 0.50),
                "p75": _quantile(all_hs, 0.75),
                "max": _quantile(all_hs, 1.0),
                "mean": round(sum(all_hs) / len(all_hs), 2) if all_hs else None,
            },
            "overall_area_px2": {
                "min": _quantile(all_ars, 0.0),
                "p25": _quantile(all_ars, 0.25),
                "median": _quantile(all_ars, 0.50),
                "p75": _quantile(all_ars, 0.75),
                "max": _quantile(all_ars, 1.0),
                "mean": round(sum(all_ars) / len(all_ars), 2) if all_ars else None,
            },
            "overall_size_buckets": overall_buckets,
            "overall_size_ratios": {
                k: round(v / total_boxes, 4) if total_boxes > 0 else 0.0
                for k, v in overall_buckets.items()
            },
            "by_class": class_summaries,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset conversion pipeline for Ground UAV Tracker"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: build-master
    p_build = subparsers.add_parser("build-master", help="Build Master JSON from .mat files")
    p_build.add_argument("--labels-dir", type=Path, required=True, help="Path to folder with *_LABELS.mat")
    p_build.add_argument("--metadata-file", type=Path, default=None, help="Path to Video_dataset_description.xlsx")
    p_build.add_argument("--videos-dir", type=Path, default=None, help="Folder with matching .mp4 files; resolution/FPS are read from their headers")
    p_build.add_argument("--output", type=Path, default=Path("dataset_master.json"), help="Output path for master JSON")

    # Subcommand: export-yolo
    p_yolo = subparsers.add_parser("export-yolo", help="Export Master JSON to YOLO format")
    p_yolo.add_argument("--master", type=Path, required=True, help="Path to dataset_master.json")
    p_yolo.add_argument("--output-dir", type=Path, required=True, help="Output directory for YOLO labels")
    p_yolo.add_argument("--image-width", type=int, default=None, help="Override per-sequence image width from the Master JSON")
    p_yolo.add_argument("--image-height", type=int, default=None, help="Override per-sequence image height from the Master JSON")
    p_yolo.add_argument("--splits-dir", type=Path, default=None, help="Directory containing train/val/test video lists")

    # Subcommand: export-mot
    p_mot = subparsers.add_parser("export-mot", help="Export Master JSON to MOT Challenge format")
    p_mot.add_argument("--master", type=Path, required=True, help="Path to dataset_master.json")
    p_mot.add_argument("--output-dir", type=Path, required=True, help="Output directory for MOT sequences")

    # Subcommand: generate-splits
    p_split = subparsers.add_parser("generate-splits", help="Generate video-level stratified splits")
    p_split.add_argument("--master", type=Path, required=True, help="Path to dataset_master.json")
    p_split.add_argument("--output-dir", type=Path, required=True, help="Output directory for split files")
    p_split.add_argument("--train-ratio", type=float, default=0.70)
    p_split.add_argument("--val-ratio", type=float, default=0.15)
    p_split.add_argument("--test-ratio", type=float, default=0.15)
    p_split.add_argument("--seed", type=int, default=42)
    p_split.add_argument("--sensor", type=str, default=None, help="Optional sensor filter (e.g. V or IR)")

    # Subcommand: analyze
    p_analyze = subparsers.add_parser("analyze", help="Compute dataset statistics and BBox size distributions")
    p_analyze.add_argument("--master", type=Path, required=True, help="Path to dataset_master.json")
    p_analyze.add_argument("--output", type=Path, default=None, help="Optional path to save stats JSON")

    # Subcommand: all
    p_all = subparsers.add_parser("all", help="Run full pipeline: build-master, splits, yolo, and mot")
    p_all.add_argument("--labels-dir", type=Path, required=True)
    p_all.add_argument("--metadata-file", type=Path, default=None)
    p_all.add_argument("--videos-dir", type=Path, default=None)
    p_all.add_argument("--sensor", type=str, default=None, help="Optional sensor filter for splits (e.g. V or IR)")
    p_all.add_argument("--output-dir", type=Path, default=Path("data/processed"))

    args = parser.parse_args()

    if args.command == "build-master":
        builder = MasterDatasetBuilder(
            labels_dir=args.labels_dir,
            metadata_file=args.metadata_file,
            videos_dir=args.videos_dir,
        )
        data = builder.build()
        for w in data["warnings"]:
            print(f"WARNING: {w}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Master JSON created at: {args.output} ({len(data['sequences'])} sequences)")

    elif args.command == "export-yolo":
        master_data = json.loads(args.master.read_text(encoding="utf-8"))
        split_catalog = None
        if args.splits_dir and args.splits_dir.is_dir():
            split_catalog = {}
            for s in ("train", "val", "test"):
                f = args.splits_dir / f"{s}_videos.txt"
                if f.is_file():
                    split_catalog[s] = [
                        line.strip()
                        for line in f.read_text(encoding="utf-8").splitlines()
                        if line.strip()
                    ]
        exporter = YoloExporter(output_dir=args.output_dir)
        exporter.export(
            master_data=master_data,
            image_width=args.image_width,
            image_height=args.image_height,
            split_catalog=split_catalog,
        )
        print(f"YOLO labels exported to: {args.output_dir}")

    elif args.command == "export-mot":
        master_data = json.loads(args.master.read_text(encoding="utf-8"))
        exporter = MotExporter(output_dir=args.output_dir)
        exporter.export(master_data=master_data)
        print(f"MOT Challenge sequences exported to: {args.output_dir}")

    elif args.command == "generate-splits":
        master_data = json.loads(args.master.read_text(encoding="utf-8"))
        gen = SplitGenerator(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            sensor_filter=args.sensor,
        )
        splits = gen.generate(master_data)
        gen.save_splits(splits, args.output_dir)
        print(
            f"Splits generated: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )

    elif args.command == "analyze":
        master_data = json.loads(args.master.read_text(encoding="utf-8"))
        analyzer = DatasetAnalyzer()
        report = analyzer.analyze(master_data)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Analysis report saved to: {args.output}")
        else:
            print(json.dumps(report, indent=2))

    elif args.command == "all":
        out = args.output_dir
        out.mkdir(parents=True, exist_ok=True)
        master_path = out / "dataset_master.json"

        print("=== 1. Building Master JSON ===")
        builder = MasterDatasetBuilder(
            labels_dir=args.labels_dir,
            metadata_file=args.metadata_file,
            videos_dir=args.videos_dir,
        )
        data = builder.build()
        for w in data["warnings"]:
            print(f"WARNING: {w}")
        master_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Master JSON saved to: {master_path}")

        print("=== 2. Generating Stratified Splits ===")
        splits_dir = out / "splits"
        gen = SplitGenerator(sensor_filter=args.sensor)
        splits = gen.generate(data)
        gen.save_splits(splits, splits_dir)
        print(f"Splits saved to: {splits_dir}")

        print("=== 3. Exporting YOLO Labels ===")
        yolo_dir = out / "yolo"
        yolo_exp = YoloExporter(output_dir=yolo_dir)
        yolo_exp.export(master_data=data, split_catalog=splits)
        print(f"YOLO labels saved to: {yolo_dir}")

        print("=== 4. Exporting MOT Sequences ===")
        mot_dir = out / "mot"
        mot_exp = MotExporter(output_dir=mot_dir)
        mot_exp.export(master_data=data)
        print(f"MOT sequences saved to: {mot_dir}")

        print("=== 5. Analyzing Dataset Statistics ===")
        stats_path = out / "dataset_stats.json"
        report = DatasetAnalyzer().analyze(data)
        stats_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Dataset statistics saved to: {stats_path}")

        print("=== Full Pipeline Completed Successfully ===")


if __name__ == "__main__":
    main()

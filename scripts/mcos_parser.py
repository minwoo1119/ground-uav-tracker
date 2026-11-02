"""Pure-Python MATLAB v5 MCOS groundTruth Label Parser.

This module parses MATLAB v5 .mat files containing MCOS groundTruth objects
without requiring MATLAB or third-party packages (only Python standard library:
struct, zlib, pathlib).

Layout details:
    MATLAB CV Toolbox Video Labeler saves groundTruth objects as an MCOS heap
    serialized into the opaque __function_workspace__ byte array.
    Inside the heap, groundTruth.LabelData is stored as a 5-column timetable:
        Col 0: Time (hh:mm:ss duration metadata, cell size 72B)
        Col 1: AIRPLANE
        Col 2: BIRD
        Col 3: DRONE
        Col 4: HELICOPTER

    Each class cell at depth 5 has size:
        48 + N * 32 bytes
        - N == 0: 48 bytes (empty 0x0 double array, target absent)
        - N == 1: 80 bytes (1x4 double array, 1 bbox)
        - N == 2: 112 bytes (2x4 double array, 2 bboxes)
        - N == 3: 144 bytes (3x4 double array, 3 bboxes)
        ...
        Each bbox consists of 4 64-bit doubles: (x, y, width, height).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
from typing import BinaryIO, Dict, List, Optional, Tuple, Union
import zlib

CLASS_COLUMNS = {
    1: "AIRPLANE",
    2: "BIRD",
    3: "DRONE",
    4: "HELICOPTER",
}

CLASS_TO_ID = {
    "AIRPLANE": 0,
    "BIRD": 1,
    "DRONE": 2,
    "HELICOPTER": 3,
}

ID_TO_CLASS = {v: k for k, v in CLASS_TO_ID.items()}


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in pixel coordinates (top-left x, y, width, height)."""

    x: float
    y: float
    width: float
    height: float

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.width, self.height)

    def to_yolo(
        self, image_width: float, image_height: float
    ) -> Tuple[float, float, float, float]:
        """Convert to normalized YOLO format (center_x, center_y, width, height)."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_width}x{image_height}")
        cx = (self.x + self.width / 2.0) / image_width
        cy = (self.y + self.height / 2.0) / image_height
        w = self.width / image_width
        h = self.height / image_height
        # Clamp to [0.0, 1.0]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        return (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))


@dataclass
class FrameAnnotation:
    """Per-frame annotation record across the 4 classes."""

    frame_index: int  # 0-based index
    boxes_by_class: Dict[str, List[BoundingBox]]

    @property
    def total_boxes(self) -> int:
        return sum(len(boxes) for boxes in self.boxes_by_class.values())

    def has_target(self) -> bool:
        return self.total_boxes > 0


@dataclass
class SequenceAnnotation:
    """Complete video sequence annotation."""

    sequence_name: str
    total_frames: int
    frames: List[FrameAnnotation]

    @property
    def class_counts(self) -> Dict[str, int]:
        counts = {cls: 0 for cls in CLASS_TO_ID}
        for frame in self.frames:
            for cls, boxes in frame.boxes_by_class.items():
                counts[cls] += len(boxes)
        return counts

    def to_dict(self) -> dict:
        return {
            "sequence_name": self.sequence_name,
            "total_frames": self.total_frames,
            "class_box_counts": self.class_counts,
            "frames": [
                {
                    "frame_index": f.frame_index,
                    "boxes": {
                        cls: [b.to_tuple() for b in boxes]
                        for cls, boxes in f.boxes_by_class.items()
                        if boxes
                    },
                }
                for f in self.frames
            ],
        }


class McosLabelParser:
    """Pure-Python parser for MATLAB groundTruth MCOS label files."""

    def __init__(self, endian: str = "<") -> None:
        self.endian = endian

    def parse_file(self, mat_path: Union[str, Path]) -> SequenceAnnotation:
        path = Path(mat_path)
        if not path.is_file():
            raise FileNotFoundError(f"Label file not found: {path}")
        raw = path.read_bytes()
        seq_name = path.stem.replace("_LABELS", "")
        return self.parse_bytes(raw, sequence_name=seq_name)

    def parse_bytes(
        self, raw: bytes, sequence_name: str = "sequence"
    ) -> SequenceAnnotation:
        if len(raw) < 128:
            raise ValueError(
                f"File buffer too small ({len(raw)} bytes) for MAT5 header"
            )

        # Verify MAT5 header and extract endianness
        endian_indicator = raw[126:128]
        if endian_indicator == b"MI":
            endian = ">"
        elif endian_indicator == b"IM":
            endian = "<"
        else:
            endian = "<"

        fw_bytes = self._extract_function_workspace(raw, endian)
        if not fw_bytes:
            raise ValueError(
                "Could not locate __function_workspace__ in MAT file (not an MCOS groundTruth file)"
            )

        table_buffer, table_offset, table_size = self._locate_timetable(
            fw_bytes, endian
        )
        return self._decode_timetable(
            table_buffer, table_offset, table_size, sequence_name, endian
        )

    def _extract_function_workspace(self, raw: bytes, endian: str) -> Optional[bytes]:
        """Traverse top-level MAT5 elements and decompress the MCOS heap."""
        off = 128
        n = len(raw)
        while off + 8 <= n:
            t, sz = struct.unpack_from(f"{endian}II", raw, off)
            pay_off = off + 8
            if t == 15:  # miCOMPRESSED
                try:
                    decomp = zlib.decompress(raw[pay_off : pay_off + sz])
                except zlib.error:
                    decomp = None
                if decomp:
                    fw = self._find_uint8_array(decomp, endian)
                    if fw is not None:
                        return fw
            next_off = pay_off + sz
            if t != 15 and (next_off % 8 != 0):
                next_off += 8 - (next_off % 8)
            if next_off <= off:
                break
            off = next_off
        return None

    def _find_uint8_array(self, buf: bytes, endian: str) -> Optional[bytes]:
        """Locate the large uint8 array containing the MCOS heap."""
        p = 8
        n = len(buf)
        while p + 8 <= n:
            st, ssz = struct.unpack_from(f"{endian}HH", buf, p)
            if ssz != 0 and st < 20:
                et, esz, epay = st, ssz, p + 4
            else:
                et, esz = struct.unpack_from(f"{endian}II", buf, p)
                epay = p + 8
            if et == 2 and esz > 50:  # miUINT8 containing MCOS heap
                return buf[epay : epay + esz]
            np = epay + esz
            if np % 8 != 0:
                np += 8 - (np % 8)
            if np <= p:
                break
            p = np
        return None

    def _locate_timetable(
        self, buf: bytes, endian: str
    ) -> Tuple[bytes, int, int]:
        """Find the depth-4 miMATRIX timetable container (rows * 5 cells)."""
        candidates: List[Tuple[int, int, int]] = []

        def walk(start: int, length: int, depth: int) -> None:
            if depth > 6:
                return
            o, end = start, start + length
            while o + 8 <= end:
                st, ssz = struct.unpack_from(f"{endian}HH", buf, o)
                if ssz != 0 and st < 20:
                    et, esz, po = st, ssz, o + 4
                else:
                    et, esz = struct.unpack_from(f"{endian}II", buf, o)
                    po = o + 8
                if et <= 0 or et > 20 or po + esz > end:
                    return
                if et == 14:  # miMATRIX
                    co, cend = po, po + esz
                    child_count = 0
                    rows = 0
                    while co + 8 <= cend:
                        cst, cssz = struct.unpack_from(f"{endian}HH", buf, co)
                        if cssz != 0 and cst < 20:
                            cet, cesz, cpo = cst, cssz, co + 4
                        else:
                            cet, cesz = struct.unpack_from(f"{endian}II", buf, co)
                            cpo = co + 8
                        if cet <= 0 or cet > 20 or cpo + cesz > cend:
                            break
                        if cet == 5 and cesz == 8 and rows == 0:
                            rows = struct.unpack_from(f"{endian}i", buf, cpo)[0]
                        if cet == 14:
                            child_count += 1
                        cno = cpo + cesz
                        rem = (cno - po) % 8
                        if rem != 0:
                            cno += 8 - rem
                        if cno <= co:
                            break
                        co = cno

                    if rows > 0 and child_count == rows * 5:
                        candidates.append((rows, po, esz))
                    walk(po, esz, depth + 1)
                no = po + esz
                rem = (no - start) % 8
                if rem != 0:
                    no += 8 - rem
                if no <= o:
                    return
                o = no

        walk(8, len(buf) - 8, 0)
        if not candidates:
            raise ValueError(
                "Failed to find 5-column timetable container in MCOS stream"
            )

        # Select the candidate with the highest frame count / size
        candidates.sort(key=lambda c: (c[0], c[2]), reverse=True)
        rows, po, esz = candidates[0]
        return buf, po, esz

    def _decode_timetable(
        self,
        buf: bytes,
        start: int,
        length: int,
        seq_name: str,
        endian: str,
    ) -> SequenceAnnotation:
        """Decode all rows in the timetable using 5-column stride."""
        co, cend = start, start + length
        cells: List[Tuple[int, int]] = []  # (payload_offset, payload_size)
        rows = 0

        while co + 8 <= cend:
            st, ssz = struct.unpack_from(f"{endian}HH", buf, co)
            if ssz != 0 and st < 20:
                cet, cesz, cpo = st, ssz, co + 4
            else:
                cet, cesz = struct.unpack_from(f"{endian}II", buf, co)
                cpo = co + 8
            if cet <= 0 or cet > 20 or cpo + cesz > cend:
                break
            if cet == 5 and cesz == 8 and rows == 0:
                rows = struct.unpack_from(f"{endian}i", buf, cpo)[0]
            if cet == 14:
                cells.append((cpo, cesz))
            cno = cpo + cesz
            rem = (cno - start) % 8
            if rem != 0:
                cno += 8 - rem
            if cno <= co:
                break
            co = cno

        if rows == 0 or len(cells) != rows * 5:
            raise ValueError(
                f"Timetable corrupt: rows={rows}, cells={len(cells)}, expected {rows * 5}"
            )

        frames: List[FrameAnnotation] = []
        for r in range(rows):
            frame_cells = cells[r * 5 : (r + 1) * 5]
            boxes_by_class: Dict[str, List[BoundingBox]] = {
                cls: [] for cls in CLASS_TO_ID
            }
            # Col 0 is Time/duration; Cols 1..4 map to CLASS_COLUMNS
            for col_idx in (1, 2, 3, 4):
                cpo, cesz = frame_cells[col_idx]
                class_name = CLASS_COLUMNS[col_idx]
                boxes = self._extract_bboxes_from_cell(buf, cpo, cesz, endian)
                boxes_by_class[class_name] = boxes

            frames.append(
                FrameAnnotation(frame_index=r, boxes_by_class=boxes_by_class)
            )

        return SequenceAnnotation(
            sequence_name=seq_name, total_frames=rows, frames=frames
        )

    def _extract_bboxes_from_cell(
        self, buf: bytes, start: int, length: int, endian: str
    ) -> List[BoundingBox]:
        """Extract length-4 double tuples from an element subtree."""
        bboxes: List[BoundingBox] = []
        o, end = start, start + length

        while o + 8 <= end:
            st, ssz = struct.unpack_from(f"{endian}HH", buf, o)
            if ssz != 0 and st < 20:
                et, esz, po = st, ssz, o + 4
            else:
                et, esz = struct.unpack_from(f"{endian}II", buf, o)
                po = o + 8
            if et <= 0 or et > 20 or po + esz > end:
                break
            if et == 9 and esz >= 32 and (esz % 32 == 0):  # miDOUBLE array
                # MATLAB stores an N x 4 matrix in column-major order:
                # [x1..xN, y1..yN, w1..wN, h1..hN]. Reading it row-major
                # silently corrupts every frame with two or more objects.
                num_boxes = esz // 32
                vals = struct.unpack_from(f"{endian}{4 * num_boxes}d", buf, po)
                for i in range(num_boxes):
                    bboxes.append(
                        BoundingBox(
                            x=round(vals[i], 2),
                            y=round(vals[num_boxes + i], 2),
                            width=round(vals[2 * num_boxes + i], 2),
                            height=round(vals[3 * num_boxes + i], 2),
                        )
                    )
            elif et == 14:  # Nested miMATRIX
                bboxes.extend(
                    self._extract_bboxes_from_cell(buf, po, esz, endian)
                )
            no = po + esz
            rem = (no - start) % 8
            if rem != 0:
                no += 8 - rem
            if no <= o:
                break
            o = no

        return bboxes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode MATLAB MCOS groundTruth .mat file in pure Python"
    )
    parser.add_argument("mat_path", type=str, help="Path to *_LABELS.mat file")
    parser.add_argument(
        "--json", action="store_true", help="Output full JSON dump"
    )
    args = parser.parse_args()

    mcos_parser = McosLabelParser()
    annotation = mcos_parser.parse_file(args.mat_path)

    if args.json:
        print(json.dumps(annotation.to_dict(), indent=2))
    else:
        print(f"Sequence:     {annotation.sequence_name}")
        print(f"Total Frames: {annotation.total_frames}")
        print("Class box counts:")
        for cls, count in annotation.class_counts.items():
            print(f"  {cls:12s}: {count} boxes")


if __name__ == "__main__":
    main()

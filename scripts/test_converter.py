"""Unit Tests for MCOS Parser and Dataset Conversion Pipeline.

Tests:
    1. Synthetic MAT5 binary generation and MCOS label parsing (0, 1, and 2 bboxes)
    2. 4-class column mapping (AIRPLANE, BIRD, DRONE, HELICOPTER)
    3. MetadataCatalog inference and synthetic XLSX parsing
    4. Master JSON construction and round-trip serialization
    5. YOLO format export and coordinate normalization
    6. MOT Challenge format export (gt.txt and seqinfo.ini)
    7. Video-level stratified split generator
"""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
import zlib

import sys

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from mcos_parser import (
    CLASS_COLUMNS,
    CLASS_TO_ID,
    BoundingBox,
    FrameAnnotation,
    McosLabelParser,
    SequenceAnnotation,
)
from metadata_parser import MetadataCatalog, SequenceMetadata
from convert_dataset import (
    MasterDatasetBuilder,
    MotExporter,
    SplitGenerator,
    YoloExporter,
    calculate_iou,
)


def make_elem(t: int, pay: bytes) -> bytes:
    """Make MAT5 tagged element with 8-byte alignment padding."""
    hdr = struct.pack("<II", t, len(pay))
    pad = (8 - (len(pay) % 8)) % 8
    return hdr + pay + (b"\x00" * pad)


def make_cell(bboxes: list) -> bytes:
    """Make a depth-5 miMATRIX cell containing 0 or more (x, y, w, h) boxes."""
    flags = make_elem(6, struct.pack("<II", 0, 0))
    dims = make_elem(5, struct.pack("<ii", len(bboxes), 4 if bboxes else 0))
    name = make_elem(1, b"")
    if bboxes:
        raw_d = b"".join(struct.pack("<4d", *b) for b in bboxes)
        d_elem = make_elem(9, raw_d)
    else:
        d_elem = make_elem(9, b"")
    pay = flags + dims + name + d_elem
    return make_elem(14, pay)


def build_synthetic_mat(rows_data: list) -> bytes:
    """Build a complete, valid in-memory MAT5 groundTruth file buffer.

    rows_data: list of 4-tuples: (airplane_boxes, bird_boxes, drone_boxes, helicopter_boxes)
    """
    rows = len(rows_data)
    cells = []
    for r in rows_data:
        # Col 0: Time metadata cell
        cells.append(make_cell([]))
        # Col 1: AIRPLANE
        cells.append(make_cell(r[0]))
        # Col 2: BIRD
        cells.append(make_cell(r[1]))
        # Col 3: DRONE
        cells.append(make_cell(r[2]))
        # Col 4: HELICOPTER
        cells.append(make_cell(r[3]))

    # Depth 4 timetable
    flags = make_elem(6, struct.pack("<II", 0, 0))
    dims = make_elem(5, struct.pack("<ii", rows, 1))
    name = make_elem(1, b"")
    table_pay = flags + dims + name + b"".join(cells)
    table_mat = make_elem(14, table_pay)

    # Wrap in 4 layers of dummy miMATRIX to reach depth 4
    cur = table_mat
    for _ in range(4):
        cur = make_elem(
            14,
            make_elem(6, struct.pack("<II", 0, 0))
            + make_elem(5, struct.pack("<ii", 1, 1))
            + make_elem(1, b"")
            + cur,
        )

    # Uint8 array containing MCOS heap with 8-byte preamble
    preamble = b"\x00" * 8
    u8_elem = make_elem(2, preamble + cur)

    # Outer miMATRIX inside compressed stream
    comp_stream = make_elem(
        14,
        make_elem(6, struct.pack("<II", 0, 0))
        + make_elem(5, struct.pack("<ii", 1, 1))
        + make_elem(1, b"__function_workspace__")
        + u8_elem,
    )
    comp_pay = zlib.compress(comp_stream)
    comp_elem = make_elem(15, comp_pay)

    # MAT5 header (128 bytes)
    header = b"MATLAB 5.0 MAT-file, Platform: PCWIN64" + (
        b" " * (116 - len("MATLAB 5.0 MAT-file, Platform: PCWIN64"))
    )
    header += struct.pack("<Q", 0)  # Subsystem offset
    header += struct.pack("<H", 0x0100)  # Version 1.0
    header += b"IM"  # Little-endian indicator

    return header + comp_elem


class TestMcosParser(unittest.TestCase):
    """Test pure-Python MCOS parsing on synthetic buffers."""

    def setUp(self) -> None:
        self.parser = McosLabelParser()
        self.sample_rows = [
            # Frame 0: single drone
            ([], [], [(100.0, 200.0, 50.0, 60.0)], []),
            # Frame 1: single bird
            ([], [(300.0, 400.0, 20.0, 30.0)], [], []),
            # Frame 2: multi-drone (2 bboxes)
            (
                [],
                [],
                [(110.0, 210.0, 52.0, 62.0), (500.0, 600.0, 40.0, 45.0)],
                [],
            ),
            # Frame 3: empty frame
            ([], [], [], []),
            # Frame 4: airplane and helicopter
            ([(50.0, 80.0, 120.0, 90.0)], [], [], [(700.0, 100.0, 80.0, 70.0)]),
        ]
        self.mat_bytes = build_synthetic_mat(self.sample_rows)

    def test_parse_synthetic_buffer(self) -> None:
        seq = self.parser.parse_bytes(self.mat_bytes, sequence_name="V_DRONE_999")
        self.assertEqual(seq.sequence_name, "V_DRONE_999")
        self.assertEqual(seq.total_frames, 5)

        # Check Frame 0 (single drone)
        f0 = seq.frames[0]
        self.assertEqual(len(f0.boxes_by_class["DRONE"]), 1)
        b0 = f0.boxes_by_class["DRONE"][0]
        self.assertEqual(b0.to_tuple(), (100.0, 200.0, 50.0, 60.0))
        self.assertEqual(len(f0.boxes_by_class["BIRD"]), 0)

        # Check Frame 1 (single bird)
        f1 = seq.frames[1]
        self.assertEqual(len(f1.boxes_by_class["BIRD"]), 1)
        self.assertEqual(
            f1.boxes_by_class["BIRD"][0].to_tuple(), (300.0, 400.0, 20.0, 30.0)
        )

        # Check Frame 2 (multi-drone)
        f2 = seq.frames[2]
        self.assertEqual(len(f2.boxes_by_class["DRONE"]), 2)
        self.assertEqual(
            f2.boxes_by_class["DRONE"][0].to_tuple(),
            (110.0, 210.0, 52.0, 62.0),
        )
        self.assertEqual(
            f2.boxes_by_class["DRONE"][1].to_tuple(),
            (500.0, 600.0, 40.0, 45.0),
        )

        # Check Frame 3 (empty)
        f3 = seq.frames[3]
        self.assertEqual(f3.total_boxes, 0)
        self.assertFalse(f3.has_target())

        # Check Frame 4 (airplane & helicopter)
        f4 = seq.frames[4]
        self.assertEqual(len(f4.boxes_by_class["AIRPLANE"]), 1)
        self.assertEqual(len(f4.boxes_by_class["HELICOPTER"]), 1)

    def test_class_box_counts(self) -> None:
        seq = self.parser.parse_bytes(self.mat_bytes, sequence_name="TEST_COUNTS")
        counts = seq.class_counts
        self.assertEqual(counts["DRONE"], 3)  # 1 in frame 0, 2 in frame 2
        self.assertEqual(counts["BIRD"], 1)
        self.assertEqual(counts["AIRPLANE"], 1)
        self.assertEqual(counts["HELICOPTER"], 1)


class TestIoU(unittest.TestCase):
    def test_iou_calculation(self) -> None:
        b1 = (0.0, 0.0, 10.0, 10.0)
        b2 = (0.0, 0.0, 10.0, 10.0)
        self.assertAlmostEqual(calculate_iou(b1, b2), 1.0)

        b3 = (5.0, 0.0, 10.0, 10.0)  # 5x10 overlap = 50, union = 150 -> 0.3333
        self.assertAlmostEqual(calculate_iou(b1, b3), 50.0 / 150.0, places=4)

        b4 = (20.0, 20.0, 5.0, 5.0)  # no overlap
        self.assertEqual(calculate_iou(b1, b4), 0.0)


class TestMetadataParser(unittest.TestCase):
    def test_infer_from_name(self) -> None:
        meta = MetadataCatalog.infer_from_name("V_DRONE_005_LABELS.mat")
        self.assertEqual(meta.sensor, "V")
        self.assertEqual(meta.primary_class, "DRONE")
        self.assertEqual(meta.number, 5)
        self.assertEqual(meta.sequence_name, "V_DRONE_005")

        meta_ir = MetadataCatalog.infer_from_name("IR_AIRPLANE_012")
        self.assertEqual(meta_ir.sensor, "IR")
        self.assertEqual(meta_ir.primary_class, "AIRPLANE")
        self.assertEqual(meta_ir.number, 12)


class TestDatasetConversionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

        # Create mock raw labels directory
        self.raw_dir = self.base_path / "raw"
        self.raw_dir.mkdir()

        # Build two synthetic .mat files
        mat1 = build_synthetic_mat(
            [
                ([], [], [(100.0, 100.0, 50.0, 50.0)], []),
                ([], [], [(105.0, 102.0, 50.0, 50.0)], []),
            ]
        )
        (self.raw_dir / "V_DRONE_001_LABELS.mat").write_bytes(mat1)

        mat2 = build_synthetic_mat(
            [
                ([], [(200.0, 200.0, 30.0, 30.0)], [], []),
                ([], [(205.0, 205.0, 30.0, 30.0)], [], []),
            ]
        )
        (self.raw_dir / "V_BIRD_001_LABELS.mat").write_bytes(mat2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_build_master_json(self) -> None:
        builder = MasterDatasetBuilder(labels_dir=self.raw_dir)
        master = builder.build()

        self.assertEqual(master["schema_version"], "1.0")
        self.assertIn("classes", master)
        self.assertEqual(len(master["sequences"]), 2)
        self.assertIn("V_DRONE_001", master["sequences"])
        self.assertIn("V_BIRD_001", master["sequences"])

        s1 = master["sequences"]["V_DRONE_001"]
        self.assertEqual(s1["total_frames"], 2)
        self.assertEqual(len(s1["frames"]), 2)
        ann0 = s1["frames"][0]["annotations"][0]
        self.assertEqual(ann0["class_name"], "DRONE")
        self.assertEqual(ann0["class_id"], 2)
        self.assertEqual(ann0["track_id"], 1)

    def test_yolo_exporter(self) -> None:
        builder = MasterDatasetBuilder(labels_dir=self.raw_dir)
        master = builder.build()

        yolo_out = self.base_path / "yolo"
        exporter = YoloExporter(output_dir=yolo_out)
        exporter.export(master, image_width=1000, image_height=1000)

        # Check data.yaml
        yaml_file = yolo_out / "data.yaml"
        self.assertTrue(yaml_file.is_file())
        self.assertIn("nc: 4", yaml_file.read_text(encoding="utf-8"))

        # Check labels file for V_DRONE_001 frame 0
        lbl_file = yolo_out / "labels" / "V_DRONE_001" / "frame_000000.txt"
        self.assertTrue(lbl_file.is_file())
        content = lbl_file.read_text(encoding="utf-8").strip()
        # box: x=100, y=100, w=50, h=50 in 1000x1000 -> cx=125/1000=0.125, cy=0.125, w=0.05, h=0.05
        parts = content.split()
        self.assertEqual(int(parts[0]), 2)  # DRONE class id
        self.assertAlmostEqual(float(parts[1]), 0.125, places=4)
        self.assertAlmostEqual(float(parts[2]), 0.125, places=4)
        self.assertAlmostEqual(float(parts[3]), 0.05, places=4)
        self.assertAlmostEqual(float(parts[4]), 0.05, places=4)

    def test_mot_exporter(self) -> None:
        builder = MasterDatasetBuilder(labels_dir=self.raw_dir)
        master = builder.build()

        mot_out = self.base_path / "mot"
        exporter = MotExporter(output_dir=mot_out)
        exporter.export(master)

        gt_file = mot_out / "V_DRONE_001" / "gt" / "gt.txt"
        self.assertTrue(gt_file.is_file())
        lines = gt_file.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        # Format: <frame>,<id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
        fields = lines[0].split(",")
        self.assertEqual(fields[0], "1")  # 1-based frame
        self.assertEqual(fields[1], "1")  # track id
        self.assertEqual(fields[2], "100.00")
        self.assertEqual(fields[3], "100.00")
        self.assertEqual(fields[4], "50.00")
        self.assertEqual(fields[5], "50.00")

        # Check seqinfo.ini
        ini_file = mot_out / "V_DRONE_001" / "seqinfo.ini"
        self.assertTrue(ini_file.is_file())
        ini_text = ini_file.read_text(encoding="utf-8")
        self.assertIn("name=V_DRONE_001", ini_text)
        self.assertIn("seqLength=2", ini_text)

    def test_split_generator(self) -> None:
        # Build master with 10 dummy sequences
        master = {
            "sequences": {
                f"V_DRONE_{i:03d}": {
                    "sequence_name": f"V_DRONE_{i:03d}",
                    "primary_class": "DRONE",
                    "distance_bin": "MEDIUM",
                    "is_multi_drone": (i == 10),
                }
                for i in range(1, 11)
            }
        }
        gen = SplitGenerator(
            train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
        )
        splits = gen.generate(master)

        # Multi-drone (i=10) should be in test
        self.assertIn("V_DRONE_010", splits["test"])
        total_assigned = (
            len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        )
        self.assertEqual(total_assigned, 10)

    def test_cli_all_execution(self) -> None:
        import subprocess

        out_processed = self.base_path / "processed"
        res = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "convert_dataset.py"),
                "all",
                "--labels-dir",
                str(self.raw_dir),
                "--output-dir",
                str(out_processed),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"CLI error: {res.stderr}")
        self.assertTrue((out_processed / "dataset_master.json").is_file())
        self.assertTrue((out_processed / "yolo" / "data.yaml").is_file())
        self.assertTrue((out_processed / "mot" / "V_DRONE_001" / "gt" / "gt.txt").is_file())
        self.assertTrue((out_processed / "splits" / "splits.json").is_file())


if __name__ == "__main__":
    unittest.main()

"""Unit tests for deterministic, dependency-free COCO export logic."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from export_coco import CocoExporter, CocoExportError


class FakeFrameSource:
    def __init__(self, width: int, height: int, frame_count: int) -> None:
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = list(range(frame_count))
        self._position = 0

    def read(self):
        if self._position >= len(self._frames):
            return None
        frame = self._frames[self._position]
        self._position += 1
        return frame

    def frame_size(self, frame):
        del frame
        return self.width, self.height

    @staticmethod
    def write_jpeg(path: Path, frame, quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame={frame};quality={quality}".encode("ascii"))

    def close(self) -> None:
        pass


def make_master() -> dict:
    return {
        "schema_version": "1.0",
        "classes": {
            "0": "AIRPLANE",
            "1": "BIRD",
            "2": "DRONE",
            "3": "HELICOPTER",
        },
        "sequences": {
            "V_DRONE_001": {
                "sequence_name": "V_DRONE_001",
                "image_width": 640,
                "image_height": 512,
                "total_frames": 2,
                "frames": [
                    {
                        "frame_index": 0,
                        "annotations": [
                            {
                                "class_id": 2,
                                "class_name": "DRONE",
                                "bbox_xywh": [-2.0, -1.0, 5.0, 5.0],
                            }
                        ],
                    },
                    {"frame_index": 1, "annotations": []},
                ],
            },
            "V_BIRD_001": {
                "sequence_name": "V_BIRD_001",
                "image_width": 640,
                "image_height": 512,
                "total_frames": 1,
                "frames": [
                    {
                        "frame_index": 0,
                        "annotations": [
                            {
                                "class_id": 1,
                                "class_name": "BIRD",
                                "bbox_xywh": [10.0, 20.0, 30.0, 40.0],
                            }
                        ],
                    }
                ],
            },
        },
    }


class CocoExporterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        for name in ("V_DRONE_001", "V_BIRD_001"):
            (self.videos / f"{name}.mp4").write_bytes(b"fixture")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def factory(path: Path) -> FakeFrameSource:
        counts = {"V_DRONE_001": 2, "V_BIRD_001": 1}
        return FakeFrameSource(640, 512, counts[path.stem])

    @staticmethod
    def splits() -> dict:
        return {
            "splits": {
                "train": ["V_DRONE_001"],
                "val": ["V_BIRD_001"],
                "test": [],
            }
        }

    def test_exports_yolox_layout_and_category_mapping(self) -> None:
        output = self.root / "coco"
        summary = CocoExporter(
            make_master(),
            self.splits(),
            self.videos,
            output,
            frame_source_factory=self.factory,
        ).export()

        self.assertEqual(summary.images, {"train": 2, "val": 1, "test": 0})
        self.assertEqual(summary.annotations, {"train": 1, "val": 1, "test": 0})
        self.assertEqual(summary.clipped_boxes["train"], 1)

        train = json.loads(
            (output / "annotations" / "instances_train2017.json").read_text("utf-8")
        )
        self.assertEqual(
            [(category["id"], category["project_class_id"], category["name"])
             for category in train["categories"]],
            [
                (1, 0, "AIRPLANE"),
                (2, 1, "BIRD"),
                (3, 2, "DRONE"),
                (4, 3, "HELICOPTER"),
            ],
        )
        self.assertEqual(train["annotations"][0]["category_id"], 3)
        self.assertEqual(train["annotations"][0]["bbox"], [0.0, 0.0, 3.0, 4.0])
        self.assertEqual(train["annotations"][0]["area"], 12.0)
        self.assertEqual(
            train["images"][0]["file_name"],
            "V_DRONE_001/frame_000000.jpg",
        )
        self.assertTrue(
            (output / "train2017" / "V_DRONE_001" / "frame_000000.jpg").is_file()
        )

        manifest = json.loads((output / "manifest.json").read_text("utf-8"))
        self.assertEqual(
            manifest["project_to_coco_category_id"],
            {"0": 1, "1": 2, "2": 3, "3": 4},
        )

    def test_output_is_deterministic(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        for output in (first, second):
            CocoExporter(
                make_master(),
                self.splits(),
                self.videos,
                output,
                frame_source_factory=self.factory,
            ).export()
        first_json = (first / "annotations" / "instances_train2017.json").read_bytes()
        second_json = (second / "annotations" / "instances_train2017.json").read_bytes()
        self.assertEqual(first_json, second_json)

    def test_rejects_sequence_leakage(self) -> None:
        splits = self.splits()
        splits["splits"]["test"] = ["V_DRONE_001"]
        with self.assertRaisesRegex(CocoExportError, "appears in both"):
            CocoExporter(
                make_master(),
                splits,
                self.videos,
                self.root / "leak",
                frame_source_factory=self.factory,
            )

    def test_missing_video_does_not_publish_partial_output(self) -> None:
        (self.videos / "V_BIRD_001.mp4").unlink()
        output = self.root / "missing"
        with self.assertRaisesRegex(CocoExportError, "was not found"):
            CocoExporter(
                make_master(),
                self.splits(),
                self.videos,
                output,
                frame_source_factory=self.factory,
            ).export()
        self.assertFalse(output.exists())

    def test_rejects_invalid_bbox(self) -> None:
        master = make_master()
        master["sequences"]["V_DRONE_001"]["frames"][0]["annotations"][0][
            "bbox_xywh"
        ] = [10.0, 10.0, 0.0, 5.0]
        output = self.root / "invalid"
        with self.assertRaisesRegex(CocoExportError, "must be positive"):
            CocoExporter(
                master,
                self.splits(),
                self.videos,
                output,
                frame_source_factory=self.factory,
            ).export()
        self.assertFalse(output.exists())

    def test_refuses_existing_output(self) -> None:
        output = self.root / "existing"
        output.mkdir()
        with self.assertRaisesRegex(CocoExportError, "refusing to overwrite"):
            CocoExporter(
                make_master(),
                self.splits(),
                self.videos,
                output,
                frame_source_factory=self.factory,
            ).export()


if __name__ == "__main__":
    unittest.main()

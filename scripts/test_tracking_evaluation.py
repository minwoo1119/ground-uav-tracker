"""Unit tests for the TrackEval adapter using synthetic MOT sequences."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_tracking import EvaluationError, evaluate_tracking, main


PERFECT_GT = """\
1,1,10,20,30,40,1,-1,-1,-1
2,1,12,20,30,40,1,-1,-1,-1
"""

PERFECT_TRACKS = """\
1,1,10,20,30,40,-1,-1,-1,-1
2,1,12,20,30,40,-1,-1,-1,-1
"""


class TrackingEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_directory.name)
        self.sequence_dir = self.root / "SYNTH_SEQUENCE"
        (self.sequence_dir / "gt").mkdir(parents=True)
        (self.sequence_dir / "seqinfo.ini").write_text(
            "[Sequence]\n"
            "name=SYNTH_SEQUENCE\n"
            "frameRate=30\n"
            "seqLength=2\n"
            "imWidth=640\n"
            "imHeight=512\n",
            encoding="utf-8",
        )
        (self.sequence_dir / "gt" / "gt.txt").write_text(
            PERFECT_GT, encoding="utf-8"
        )
        self.tracks_path = self.root / "tracks.txt"

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def evaluate(self, tracks: str) -> dict:
        self.tracks_path.write_text(tracks, encoding="utf-8")
        return evaluate_tracking(self.sequence_dir, self.tracks_path)

    def test_perfect_tracking_scores_one(self) -> None:
        result = self.evaluate(PERFECT_TRACKS)
        self.assertAlmostEqual(result["metrics"]["HOTA"]["HOTA"], 1.0)
        self.assertAlmostEqual(result["metrics"]["CLEAR"]["MOTA"], 1.0)
        self.assertAlmostEqual(result["metrics"]["Identity"]["IDF1"], 1.0)
        self.assertEqual(result["metrics"]["CLEAR"]["identity_switches"], 0)

    def test_missing_detection_increases_false_negatives(self) -> None:
        result = self.evaluate("1,1,10,20,30,40,-1,-1,-1,-1\n")
        self.assertEqual(result["metrics"]["CLEAR"]["false_negatives"], 1)
        self.assertAlmostEqual(result["metrics"]["CLEAR"]["MOTA"], 0.5)
        self.assertLess(result["metrics"]["HOTA"]["HOTA"], 1.0)

    def test_identity_change_is_counted(self) -> None:
        switched_tracks = """\
1,1,10,20,30,40,-1,-1,-1,-1
2,2,12,20,30,40,-1,-1,-1,-1
"""
        result = self.evaluate(switched_tracks)
        self.assertEqual(result["metrics"]["CLEAR"]["identity_switches"], 1)
        self.assertAlmostEqual(result["metrics"]["CLEAR"]["MOTA"], 0.5)
        self.assertLess(result["metrics"]["Identity"]["IDF1"], 1.0)

    def test_invalid_rows_are_rejected(self) -> None:
        invalid_cases = {
            "duplicate": (
                "1,1,10,20,30,40,-1,-1,-1,-1\n"
                "1,1,11,20,30,40,-1,-1,-1,-1\n",
                "duplicate frame/track pair",
            ),
            "invalid_bbox": (
                "1,1,10,20,0,40,-1,-1,-1,-1\n",
                "width/height > 0",
            ),
            "out_of_range": (
                "3,1,10,20,30,40,-1,-1,-1,-1\n",
                "exceeds seqLength",
            ),
        }
        for name, (contents, message) in invalid_cases.items():
            with self.subTest(name=name):
                self.tracks_path.write_text(contents, encoding="utf-8")
                with self.assertRaisesRegex(EvaluationError, message):
                    evaluate_tracking(self.sequence_dir, self.tracks_path)

    def test_cli_prints_json(self) -> None:
        self.tracks_path.write_text(PERFECT_TRACKS, encoding="utf-8")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(
                [
                    "--sequence-dir",
                    str(self.sequence_dir),
                    "--tracks",
                    str(self.tracks_path),
                ]
            )
        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["sequence"]["name"], "SYNTH_SEQUENCE")
        self.assertEqual(result["tool"], {"name": "TrackEval", "version": "1.3.0"})
        self.assertEqual(
            result["evaluation_scope"],
            "tracking_only_from_ground_truth_detections",
        )


if __name__ == "__main__":
    unittest.main()

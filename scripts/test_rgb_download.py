"""Offline tests for pinned RGB dataset manifest selection and verification."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from download_rgb_videos import (
    DownloadError,
    EXPECTED_RGB_VIDEOS,
    _git_blob_sha,
    _select_rgb_videos,
)


class RgbDownloadTest(unittest.TestCase):
    def test_selects_exact_visible_video_set(self) -> None:
        entries = [
            {
                "path": f"Data/Video_V/V_DRONE_{index:03d}.mp4",
                "type": "blob",
                "size": index,
                "sha": f"sha-{index}",
            }
            for index in range(1, EXPECTED_RGB_VIDEOS + 1)
        ]
        entries.extend(
            [
                {"path": "Data/Video_IR/IR_DRONE_001.mp4", "type": "blob"},
                {"path": "Data/Video_V/V_DRONE_001_LABELS.mat", "type": "blob"},
            ]
        )
        selected = _select_rgb_videos({"truncated": False, "tree": entries})
        self.assertEqual(len(selected), EXPECTED_RGB_VIDEOS)
        self.assertTrue(all(item["path"].endswith(".mp4") for item in selected))

    def test_rejects_changed_upstream_count(self) -> None:
        with self.assertRaisesRegex(DownloadError, "expected 285"):
            _select_rgb_videos({"truncated": False, "tree": []})

    def test_git_blob_sha_matches_known_value(self) -> None:
        self.assertEqual(
            _git_blob_sha(b"test\n"),
            "9daeafb9864cf43055ae93beb0afd6c7d144bfa4",
        )


if __name__ == "__main__":
    unittest.main()

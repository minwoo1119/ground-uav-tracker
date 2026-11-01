"""Minimal MP4 header probe (stdlib only).

Reads the visual sample entry (codec, width, height), the media timescale and
the sample count/durations from an ISO-BMFF (.mp4) file without decoding any
frames. Used to attach verified resolution/FPS to the Master JSON instead of
guessing them.

Usage:
    python scripts/video_probe.py data/raw/Video_V/V_DRONE_001.mp4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import sys
from typing import List, Optional, Tuple

_CONTAINERS = {"moov", "trak", "mdia", "minf", "stbl"}


@dataclass(frozen=True)
class VideoInfo:
    codec: str
    width: int
    height: int
    frame_count: int
    fps: Optional[float]


def _walk(buf: bytes, start: int, end: int, tracks: List[dict], cur: Optional[dict]) -> None:
    o = start
    while o + 8 <= end:
        size, raw_type = struct.unpack_from(">I4s", buf, o)
        hdr = 8
        if size == 1:
            size = struct.unpack_from(">Q", buf, o + 8)[0]
            hdr = 16
        elif size == 0:
            size = end - o
        if size < hdr or o + size > end:
            raise ValueError(f"Corrupt MP4 box at offset {o}")
        box = raw_type.decode("latin1")
        body = o + hdr
        if box == "trak":
            cur = {}
            tracks.append(cur)
            _walk(buf, body, o + size, tracks, cur)
        elif box in _CONTAINERS:
            _walk(buf, body, o + size, tracks, cur)
        elif cur is not None and box == "mdhd":
            version = buf[body]
            if version == 1:
                cur["timescale"] = struct.unpack_from(">I", buf, body + 20)[0]
            else:
                cur["timescale"] = struct.unpack_from(">I", buf, body + 12)[0]
        elif cur is not None and box == "stsd":
            entry = body + 8
            cur["codec"] = buf[entry + 4 : entry + 8].decode("latin1")
            # VisualSampleEntry: 8B box header + 24B fixed fields, then width/height.
            cur["wh"] = struct.unpack_from(">HH", buf, entry + 32)
        elif cur is not None and box == "stts":
            count = struct.unpack_from(">I", buf, body + 4)[0]
            cur["stts"] = [
                struct.unpack_from(">II", buf, body + 8 + 8 * i) for i in range(count)
            ]
        o += size


def probe_mp4(path: Path) -> VideoInfo:
    """Return codec, resolution, frame count and FPS of the first video track."""
    buf = Path(path).read_bytes()
    tracks: List[dict] = []
    _walk(buf, 0, len(buf), tracks, None)
    for t in tracks:
        if "wh" not in t or "stts" not in t:
            continue
        entries: List[Tuple[int, int]] = t["stts"]
        frames = sum(n for n, _ in entries)
        fps: Optional[float] = None
        deltas = {d for _, d in entries}
        if len(deltas) == 1 and t.get("timescale"):
            fps = round(t["timescale"] / deltas.pop(), 3)
        w, h = t["wh"]
        return VideoInfo(codec=t.get("codec", "?"), width=w, height=h, frame_count=frames, fps=fps)
    raise ValueError(f"No video track found in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    for arg in sys.argv[1:]:
        print(arg, probe_mp4(Path(arg)))

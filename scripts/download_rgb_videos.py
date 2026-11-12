"""Download and verify the 285 official visible-spectrum dataset videos."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List


DEFAULT_REPOSITORY = "DroneDetectionThesis/Drone-detection-dataset"
DEFAULT_COMMIT = "442d0c6750708cc221636f7a1fd31cbd8cb0aeb8"
EXPECTED_RGB_VIDEOS = 285


class DownloadError(RuntimeError):
    """Raised when the pinned upstream dataset cannot be verified."""


def _request_json(url: str) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ground-uav-tracker-dataset-fetcher",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        raise DownloadError(f"failed to query dataset tree: {url}: {error}") from error
    if not isinstance(value, dict):
        raise DownloadError("GitHub tree response was not a JSON object")
    return value


def _select_rgb_videos(tree: Dict[str, Any]) -> List[Dict[str, Any]]:
    if tree.get("truncated"):
        raise DownloadError("GitHub tree response was truncated")
    entries = tree.get("tree")
    if not isinstance(entries, list):
        raise DownloadError("GitHub tree response is missing the 'tree' array")
    videos = sorted(
        (
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("type") == "blob"
            and isinstance(entry.get("path"), str)
            and entry["path"].startswith("Data/Video_V/")
            and entry["path"].endswith(".mp4")
        ),
        key=lambda entry: entry["path"],
    )
    if len(videos) != EXPECTED_RGB_VIDEOS:
        raise DownloadError(
            f"expected {EXPECTED_RGB_VIDEOS} RGB videos at the pinned commit, "
            f"found {len(videos)}"
        )
    return videos


def _git_blob_sha(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def _download_one(
    repository: str,
    commit: str,
    entry: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    relative_path = entry["path"]
    name = Path(relative_path).name
    destination = output_dir / name
    expected_size = entry.get("size")
    expected_sha = entry.get("sha")
    if not isinstance(expected_size, int) or not isinstance(expected_sha, str):
        raise DownloadError(f"upstream tree entry lacks size/SHA: {relative_path}")

    if destination.is_file():
        payload = destination.read_bytes()
        actual_sha = _git_blob_sha(payload)
        if len(payload) != expected_size or actual_sha != expected_sha:
            raise DownloadError(
                f"existing video does not match pinned upstream; refusing to overwrite: "
                f"{destination}"
            )
        return {
            "name": name,
            "bytes": len(payload),
            "git_blob_sha1": actual_sha,
            "status": "verified_existing",
        }

    url = f"https://raw.githubusercontent.com/{repository}/{commit}/{relative_path}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "ground-uav-tracker-dataset-fetcher"}
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise DownloadError(f"partial download already exists: {temporary}")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        actual_sha = _git_blob_sha(payload)
        if len(payload) != expected_size or actual_sha != expected_sha:
            raise DownloadError(
                f"download verification failed for {name}: expected "
                f"{expected_size}/{expected_sha}, got {len(payload)}/{actual_sha}"
            )
        temporary.write_bytes(payload)
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        if temporary.is_file():
            temporary.unlink()
        raise DownloadError(f"failed to download {name}: {error}") from error

    return {
        "name": name,
        "bytes": len(payload),
        "git_blob_sha1": actual_sha,
        "status": "downloaded",
    }


def download_rgb_videos(
    output_dir: Path,
    repository: str = DEFAULT_REPOSITORY,
    commit: str = DEFAULT_COMMIT,
    workers: int = 8,
) -> Dict[str, Any]:
    if workers <= 0:
        raise DownloadError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    tree_url = f"https://api.github.com/repos/{repository}/git/trees/{commit}?recursive=1"
    videos = _select_rgb_videos(_request_json(tree_url))

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_one, repository, commit, entry, output_dir)
            for entry in videos
        ]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["name"])
    manifest = {
        "source_repository": f"https://github.com/{repository}",
        "source_commit": commit,
        "license": "CC0-1.0",
        "video_count": len(results),
        "total_bytes": sum(item["bytes"] for item in results),
        "files": results,
    }
    (output_dir / "rgb_video_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official RGB dataset videos")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = download_rgb_videos(
        output_dir=args.output_dir,
        repository=args.repository,
        commit=args.commit,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "source_commit": manifest["source_commit"],
                "video_count": manifest["video_count"],
                "total_bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except DownloadError as error:
        raise SystemExit(f"error: {error}") from error

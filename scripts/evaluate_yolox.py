"""Evaluate a trained four-class YOLOX checkpoint with the official COCO API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _existing(path: Path, description: str, *, directory: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    valid = resolved.is_dir() if directory else resolved.is_file()
    if not valid:
        raise SystemExit(f"error: {description} not found: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the four-class YOLOX baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("training/yolox_s_uav.py"),
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("error: batch-size must be positive and workers non-negative")
    data_dir = _existing(args.data_dir, "COCO dataset", directory=True)
    checkpoint = _existing(args.checkpoint, "checkpoint")
    experiment = _existing(args.experiment, "YOLOX experiment")
    output = args.output.expanduser().resolve() if args.output is not None else None
    if output is not None and output.exists():
        raise SystemExit(f"error: refusing to overwrite evaluation result: {output}")

    annotation = data_dir / "annotations" / f"instances_{args.split}2017.json"
    image_dir = data_dir / f"{args.split}2017"
    if not annotation.is_file() or not image_dir.is_dir():
        raise SystemExit(
            f"error: incomplete COCO {args.split} split: {annotation}, {image_dir}"
        )

    os.environ["UAV_COCO_DATA_DIR"] = str(data_dir)
    os.environ["UAV_YOLOX_WORKERS"] = str(args.workers)

    try:
        import torch
        from yolox.exp import get_exp
    except ImportError as error:
        raise SystemExit("error: incomplete YOLOX evaluation environment") from error

    if not torch.cuda.is_available():
        cuda_build = torch.version.cuda or "none (CPU-only wheel)"
        raise SystemExit(
            "error: official YOLOX COCO evaluation requires CUDA. "
            f"Installed torch={torch.__version__}, CUDA build={cuda_build}."
        )

    exp = get_exp(str(experiment), None)
    if args.split == "test":
        exp.val_ann = exp.test_ann
    model = exp.get_model().cuda()
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = stored["model"] if isinstance(stored, dict) and "model" in stored else stored
    model.load_state_dict(state_dict, strict=True)

    evaluator = exp.get_evaluator(args.batch_size, is_distributed=False)
    ap50_95, ap50, summary = evaluator.evaluate(
        model,
        distributed=False,
        half=args.fp16,
    )
    result = {
        "split": args.split,
        "checkpoint": str(checkpoint),
        "ap50_95": float(ap50_95),
        "ap50": float(ap50),
        "summary": summary,
    }
    rendered = json.dumps(result, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

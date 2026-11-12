"""Run one CPU forward/backward step and save an explicitly untrained checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOX CPU architecture smoke test")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="optional official YOLOX checkpoint; incompatible class-head tensors are skipped",
    )
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("training/yolox_s_uav.py"),
    )
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 != 0:
        raise SystemExit("error: smoke size must be a positive multiple of 32")
    data_dir = args.data_dir.expanduser().resolve()
    experiment = args.experiment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    pretrained = (
        args.pretrained.expanduser().resolve()
        if args.pretrained is not None
        else None
    )
    if not data_dir.is_dir() or not experiment.is_file():
        raise SystemExit("error: COCO data directory or experiment file is missing")
    if output.exists():
        raise SystemExit(f"error: refusing to overwrite smoke checkpoint: {output}")
    if pretrained is not None and not pretrained.is_file():
        raise SystemExit(f"error: pretrained checkpoint not found: {pretrained}")

    os.environ["UAV_COCO_DATA_DIR"] = str(data_dir)
    import torch
    from yolox.exp import get_exp
    from yolox.utils import load_ckpt

    torch.manual_seed(42)
    exp = get_exp(str(experiment), None)
    model = exp.get_model().cpu().train()
    if pretrained is not None:
        checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        model = load_ckpt(model, state_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    inputs = torch.zeros(1, 3, args.size, args.size)
    targets = torch.zeros(1, 1, 5)
    targets[0, 0] = torch.tensor(
        [2.0, args.size / 2, args.size / 2, args.size / 4, args.size / 4]
    )

    started = time.perf_counter()
    losses = model(inputs, targets)
    loss = losses["total_loss"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite YOLOX smoke loss: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    elapsed = time.perf_counter() - started

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "artifact_kind": "untrained_cpu_smoke_checkpoint",
            "input_size": args.size,
        },
        output,
    )
    print(
        json.dumps(
            {
                "loss": float(loss.detach()),
                "elapsed_seconds": round(elapsed, 3),
                "input_shape": [1, 3, args.size, args.size],
                "checkpoint": str(output),
                "pretrained": str(pretrained) if pretrained is not None else None,
                "trained_model": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

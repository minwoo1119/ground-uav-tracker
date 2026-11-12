"""Launch the approved YOLOX-s baseline on a CUDA-capable Windows host."""

from __future__ import annotations

import argparse
from argparse import Namespace
import os
from pathlib import Path
import random
import sys
from typing import Any
import warnings


def _directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"error: {description} directory not found: {resolved}")
    return resolved


def _file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"error: {description} not found: {resolved}")
    return resolved


def _train(exp: Any, args: Namespace) -> None:
    import torch
    import torch.backends.cudnn as cudnn

    from yolox.utils import configure_nccl, configure_omp

    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "deterministic CUDA mode can reduce training speed",
            stacklevel=2,
        )
    configure_nccl()
    configure_omp()
    cudnn.benchmark = not cudnn.deterministic
    exp.get_trainer(args).train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the 4-class YOLOX-s baseline")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolox"))
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("training/yolox_s_uav.py"),
    )
    parser.add_argument("--pretrained", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cache", choices=("ram", "disk"), default=None)
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise SystemExit("error: epochs/batch-size must be positive and workers non-negative")

    data_dir = _directory(args.data_dir, "COCO dataset")
    experiment = _file(args.experiment, "YOLOX experiment")
    pretrained = (
        _file(args.pretrained, "pretrained checkpoint")
        if args.pretrained is not None
        else None
    )
    required = (
        data_dir / "annotations" / "instances_train2017.json",
        data_dir / "annotations" / "instances_val2017.json",
        data_dir / "train2017",
        data_dir / "val2017",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("error: incomplete COCO dataset:\n  " + "\n  ".join(missing))

    os.environ["UAV_COCO_DATA_DIR"] = str(data_dir)
    os.environ["UAV_YOLOX_OUTPUT_DIR"] = str(args.output_dir.resolve())
    os.environ["UAV_YOLOX_MAX_EPOCHS"] = str(args.epochs)
    os.environ["UAV_YOLOX_WORKERS"] = str(args.workers)

    try:
        import torch
        from yolox.core import launch
        from yolox.exp import check_exp_value, get_exp
        from yolox.utils import configure_module, get_num_devices
    except ImportError as error:
        raise SystemExit(
            "error: YOLOX training environment is incomplete; install the pinned "
            "training requirements"
        ) from error

    if not torch.cuda.is_available() or get_num_devices() < 1:
        cuda_build = torch.version.cuda or "none (CPU-only wheel)"
        raise SystemExit(
            "error: CUDA GPU not found; full YOLOX training is not supported on CPU. "
            f"Installed torch={torch.__version__}, CUDA build={cuda_build}. "
            "Install the pinned CUDA wheel or use scripts/yolox_smoke.py for a CPU "
            "architecture check."
        )

    configure_module()
    exp = get_exp(str(experiment), None)
    check_exp_value(exp)
    trainer_args = Namespace(
        experiment_name=exp.exp_name,
        name=None,
        dist_backend="nccl",
        dist_url=None,
        batch_size=args.batch_size,
        devices=1,
        exp_file=str(experiment),
        resume=args.resume,
        ckpt=str(pretrained) if pretrained is not None else None,
        start_epoch=None,
        num_machines=1,
        machine_rank=0,
        fp16=args.fp16,
        cache=args.cache,
        occupy=False,
        logger="tensorboard",
        opts=[],
    )
    launch(
        _train,
        1,
        1,
        0,
        backend="nccl",
        dist_url="auto",
        args=(exp, trainer_args),
    )


if __name__ == "__main__":
    main()

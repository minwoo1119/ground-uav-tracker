"""YOLOX-s experiment for the approved 4-class, fixed-640 baseline."""

from __future__ import annotations

import os
from pathlib import Path

from yolox.exp import Exp as YoloXBaseExp


def _required_directory(name: str) -> str:
    raw_value = os.environ.get(name)
    if not raw_value:
        raise RuntimeError(f"{name} must point to the generated COCO dataset")
    path = Path(raw_value).expanduser().resolve()
    if not path.is_dir():
        raise RuntimeError(f"{name} directory does not exist: {path}")
    return str(path)


class Exp(YoloXBaseExp):
    """Reproducible first baseline; these settings are not claimed to be optimal."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0.33
        self.width = 0.50
        self.num_classes = 4
        self.data_dir = _required_directory("UAV_COCO_DATA_DIR")
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_test2017.json"

        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.multiscale_range = 0
        self.seed = 42

        self.max_epoch = int(os.environ.get("UAV_YOLOX_MAX_EPOCHS", "50"))
        self.warmup_epochs = 1
        self.no_aug_epochs = 5
        self.eval_interval = 1
        self.data_num_workers = int(os.environ.get("UAV_YOLOX_WORKERS", "4"))
        self.save_history_ckpt = False

        output_dir = os.environ.get("UAV_YOLOX_OUTPUT_DIR", "outputs/yolox")
        self.output_dir = str(Path(output_dir).expanduser().resolve())
        self.exp_name = "yolox_s_uav_640"

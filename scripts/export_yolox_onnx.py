"""Export a trained 4-class YOLOX checkpoint and validate it with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and validate YOLOX ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("training/yolox_s_uav.py"),
    )
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    experiment = args.experiment.expanduser().resolve()
    output = args.output.expanduser().resolve()
    for path, description, kind in (
        (checkpoint, "checkpoint", "file"),
        (experiment, "experiment", "file"),
        (data_dir, "COCO dataset", "directory"),
    ):
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            raise SystemExit(f"error: {description} not found: {path}")
    if output.exists():
        raise SystemExit(f"error: refusing to overwrite existing ONNX file: {output}")

    os.environ["UAV_COCO_DATA_DIR"] = str(data_dir)
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from torch import nn
        from yolox.exp import get_exp
        from yolox.models.network_blocks import SiLU
        from yolox.utils import replace_module
    except ImportError as error:
        raise SystemExit("error: incomplete YOLOX/ONNX environment") from error

    exp = get_exp(str(experiment), None)
    model = exp.get_model()
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = stored["model"] if isinstance(stored, dict) and "model" in stored else stored
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = replace_module(model, nn.SiLU, SiLU)
    model.head.decode_in_inference = False

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".onnx-export-") as temp:
        staged = Path(temp) / output.name
        dummy = torch.zeros(1, 3, exp.test_size[0], exp.test_size[1])
        torch.onnx.export(
            model,
            dummy,
            staged,
            input_names=["images"],
            output_names=["output"],
            opset_version=args.opset,
            dynamic_axes=None,
            dynamo=False,
        )
        onnx_model = onnx.load(staged)
        onnx.checker.check_model(onnx_model)
        session = ort.InferenceSession(
            str(staged), providers=["CPUExecutionProvider"]
        )
        outputs = session.run(None, {"images": np.zeros((1, 3, 640, 640), np.float32)})
        if len(outputs) != 1 or outputs[0].ndim != 3:
            raise RuntimeError(f"unexpected YOLOX ONNX outputs: {[item.shape for item in outputs]}")
        if outputs[0].shape[0] != 1 or outputs[0].shape[2] != 9:
            raise RuntimeError(
                f"expected output [1, anchors, 9] for four classes, got {outputs[0].shape}"
            )
        staged.replace(output)

    print(
        json.dumps(
            {
                "output": str(output),
                "opset": args.opset,
                "input_shape": [1, 3, 640, 640],
                "output_shape": list(outputs[0].shape),
                "decode_in_inference": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

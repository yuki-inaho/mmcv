#!/usr/bin/env python3
"""mmcv cu12 GPU smoke: exercise mmcv.ops.box_iou_rotated on CUDA tensors.

Run via the repo's uv env:

    uv run python tools/cu12/smoke_mmcv.py --output-json /tmp/smoke_mmcv.json

Exit code 0 means the CUDA op ran on device='cuda' with torch.cuda available.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import run_smoke, standard_argv  # noqa: E402


def runner() -> dict[str, Any]:
    import torch
    from mmcv.ops import box_iou_rotated

    # [cx, cy, w, h, angle]
    boxes1 = torch.tensor(
        [[10.0, 10.0, 5.0, 5.0, 0.0], [20.0, 20.0, 4.0, 6.0, 0.3]],
        device="cuda",
        dtype=torch.float32,
    )
    boxes2 = torch.tensor(
        [[10.0, 10.0, 5.0, 5.0, 0.0], [25.0, 25.0, 4.0, 6.0, 0.6]],
        device="cuda",
        dtype=torch.float32,
    )
    iou = box_iou_rotated(boxes1, boxes2)
    torch.cuda.synchronize()
    return {
        "op": "mmcv.ops.box_iou_rotated",
        "cuda_op_device": iou.device.type,
        "output_shape": list(iou.shape),
        "output_dtype": str(iou.dtype),
        "self_iou_top_left": float(iou[0, 0].item()),
    }


def main() -> int:
    args = standard_argv("mmcv cu12 GPU smoke")
    return run_smoke(module_name="mmcv", runner=runner, output_json=args.output_json)


if __name__ == "__main__":
    sys.exit(main())

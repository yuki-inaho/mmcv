# Running this mmcv fork on cu12 (torch 2.1.0 + cu121)

This branch is part of the OpenMMLab cu12 migration whose end goal is to train
`tomato_pipe_rgbd_rtmdet_obb_training` on a cu12 base. The validated stack is
**Python 3.10 / torch 2.1.0+cu121 / mmcv 2.2.0** (see `mmopenlab_cu12_sandbox`).

## Quick start (uv)

```bash
just sync          # provision .venv with torch 2.1.0+cu121 + mmcv 2.2.0 (prebuilt cu121 wheel) + numpy<2
just smoke         # exercise mmcv.ops.box_iou_rotated on the GPU -> success: true / cuda_op_device: cuda
just env-doctor    # print GPU + torch/mmcv/numpy versions
```

`uv sync` reads `pyproject.toml`, a **virtual** uv project (`tool.uv.package = false`):
it does not build mmcv from this tree — it installs the prebuilt cu121 cp310 wheel
`mmcv-2.2.0-cp310-cp310-manylinux1_x86_64.whl` for fast, reproducible setup.

> Note: importing `mmcv` from this repo root would shadow the installed wheel with the
> (uncompiled) source package. The smoke and `env-doctor` import from a neutral cwd to
> avoid this; run downstream training from the consuming project, not from here.

## Building wheels from source per compute capability

For deploy GPUs you may need wheels built from this source tree. The build needs a
CUDA toolkit (`nvcc`); it does not run on this nvcc-less box but works in CI / RunPod.

```bash
just build-wheel CC=8.6     # -> dist/cc_8.6/mmcv-*.whl
just build-all             # 6.1 6.2 8.6 8.7 8.9 9.0 (sm_90 max; 10.0/12.0 deferred)
```

The GitHub Action `.github/workflows/build-cu121-wheels.yml` builds one specialized
wheel per compute capability and uploads them to the GitHub Release on tag push.

# mmcv cu12 (torch 2.1.0 + cu121) task runner.
#
#   just sync        # provision the cu12 uv env (.venv) from pyproject.toml
#   just smoke       # run the box_iou_rotated GPU smoke in that env
#   just env-doctor  # print python/torch/mmcv/gpu state
#   just build-wheel CC=8.6   # build an mmcv wheel from THIS source for one arch (needs nvcc)
#   just build-all   # build wheels for all in-scope compute capabilities

VENV := ".venv"
PY := justfile_directory() + "/" + VENV + "/bin/python"
TORCH_VERSION := "2.1.0"
CU_INDEX := "https://download.pytorch.org/whl/cu121"

# Compute capabilities targeted by the wheel build (sm_90 max; 10.0/12.0 deferred).
CCS := "6.1 6.2 8.6 8.7 8.9 9.0"

default:
    @just --list

list:
    @just --list

# Provision the cu12 environment (.venv) declared in pyproject.toml.
sync:
    uv sync

# Read-only environment triage. Imports run from /tmp so the installed mmcv
# wheel is reported, not the (uncompiled) source package in this repo root.
env-doctor:
    @echo "=== gpu ==="
    @nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"
    @echo ""
    @echo "=== python / packages (installed, imported from /tmp) ==="
    @if [ -x "{{ PY }}" ]; then \
        "{{ PY }}" --version; \
        for pkg in torch mmcv mmengine numpy; do \
            (cd /tmp && "{{ PY }}" -c "import importlib; m=importlib.import_module('$pkg'); print('$pkg', getattr(m,'__version__','?'))") 2>/dev/null || echo "$pkg IMPORT_ERROR"; \
        done; \
        (cd /tmp && "{{ PY }}" -c "import torch; print('cuda_available', torch.cuda.is_available(), '| cuda', torch.version.cuda)"); \
    else \
        echo "{{ PY }} missing — run 'just sync' first."; \
    fi

# Run the mmcv CUDA-ops GPU smoke (box_iou_rotated) in the synced env.
smoke OUT="/tmp/smoke_mmcv_cu12.json": sync
    uv run python tools/cu12/smoke_mmcv.py --output-json "{{ OUT }}"

# Build an mmcv wheel FROM THIS SOURCE TREE for a single compute capability.
# Produces dist/cc_<CC>/mmcv-*.whl. Requires a CUDA toolkit (nvcc) for the ops.
build-wheel CC="8.6":
    @echo "Building mmcv wheel for compute capability {{ CC }} (torch {{ TORCH_VERSION }} + cu121)..."
    rm -rf "dist/cc_{{ CC }}"
    # Move the uv devenv pyproject.toml aside so `pip wheel .` builds mmcv from
    # setup.py (not the devenv project); restore it afterwards. setuptools<81 keeps
    # pkg_resources, which mmcv's setup.py imports.
    if [ -f pyproject.toml ]; then mv pyproject.toml .pyproject.devenv.bak; fi
    uv run --no-project --python 3.10 \
        --with "torch=={{ TORCH_VERSION }}" --with "numpy<2" \
        --with pip --with "setuptools<81" --with wheel --with ninja \
        --index-strategy unsafe-best-match --extra-index-url "{{ CU_INDEX }}" \
        env MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="{{ CC }}" \
        python -m pip wheel --no-build-isolation --no-deps -w "dist/cc_{{ CC }}" . -v ; \
        rc=$? ; if [ -f .pyproject.devenv.bak ]; then mv .pyproject.devenv.bak pyproject.toml; fi ; exit $rc
    @echo "Wheel(s) in dist/cc_{{ CC }}/:"
    @ls -lh "dist/cc_{{ CC }}/" || true

# Build wheels for every in-scope compute capability.
build-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for cc in {{ CCS }}; do
        just build-wheel "$cc"
    done
    echo "All wheels built under dist/cc_*/"

# Remove build artifacts.
clean:
    rm -rf dist build *.egg-info
    find . -name "*.so" -path "*/mmcv/*" -delete 2>/dev/null || true

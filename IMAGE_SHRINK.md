# Docker Image Size Reduction Plan

## Measured Disk Budget

The full `Dockerfile.trt`-built image sits on top of the 17.4 GB `nvcr.io/nvidia/tensorrt:26.01-py3`
base. Here is where the space actually goes:

### Base image internals (11 GB uncompressed `/`)

| Component | Size | Notes |
|---|---|---|
| CUDA math libs (`/usr/local/cuda-13.1/targets/sbsa-linux/lib`) | 2.4 GB | cublas 547M, cufft 283M, cusparseLt 262M, cusolver 148M, curand 142M, etc. Core CUDA compute — required |
| **TRT builder resources (all SM archs, `/usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_*`)** | **2.1 GB total** | sm90=437M, sm100=276M, sm110=264M, sm120=254M, ptx=232M, sm80=182M, sm89=181M, sm86=172M, sm75=115M — see Priority 3 below |
| `libnvinfer.so.10.14.1` (main TRT lib) | 790 MB | Required |
| **NsightSystems CLI profiler** | **559 MB** | Developer profiler, not needed in the export container |
| `libcusparseLt/` (sparse matrix) | 377 MB | Used for sparse neural network optimization; probably not needed for dense YOLO inference |
| CUDA compat (`/usr/local/cuda-13.1/compat`) | 396 MB | Driver forward-compat layer, required if driver on host is newer than image |
| `libnvinfer_lean.so.10.14.1` | 170 MB | Required (lean inference runtime) |
| **CUDA bin (nvcc, cuobjdump, etc.)** | **276 MB** | Compiler tools, not needed at runtime |
| CUDA static libs (`lib*.a`) | ~235 MB | Link-time only — not needed at runtime |
| **NCCL (`libnccl.so.2.29.2`)** | **183 MB** | Distributed training collective comms — not needed for single-GPU export |
| `libcudnn_engines_precompiled.so.9.17.1` | 253 MB | Required (cuDNN engine execution) |
| `libcudnn_adv.so.9.17.1` | 119 MB | Required (cuDNN convolutions) |
| **CUDA nvvm (PTX compiler backend)** | **126 MB** | JIT compiler for PTX, not needed if we're building engines ahead of time |
| **compute-sanitizer** | **38 MB** | Developer debugger, not needed |
| CUDA include headers | 35 MB | Headers, not needed at runtime |
| Python packages in base (tensorrt, cuda-python, numba, llvmlite, numpy, pillow) | 886 MB | Required |

**Bolded rows = potentially removable.** However, since these live in immutable base layers, deleting
them with `RUN rm` creates whiteout entries in a new layer but does not recover the compressed size
on the pull from a registry or a fresh install. True removal requires a multi-stage build with a
clean final stage (see Priority 2 and Priority 3 below).

### Our added layers (estimated)

| Component | Estimated Size | Notes |
|---|---|---|
| apt packages (cmake, build-essential, libgl1, etc.) | ~400 MB | cmake + build-essential are build-time only |
| torch + torchvision | ~2.5 GB | Largest pip addition |
| onnxruntime-gpu | ~500 MB | Needed |
| ultralytics + deps (matplotlib, scipy, etc.) | ~1 GB | Needed |
| tensorboard | ~500 MB | **Not used — remove** |
| textual[dev] extras | ~50 MB | **Not needed — use `textual` not `textual[dev]`** |
| opencv-python Qt GUI libs | ~100 MB | **Not needed — use `opencv-python-headless`** |
| onnxconverter-common, onnxscript | ~20 MB | Likely transitive deps of onnxruntime-gpu/ultralytics; verify before removing |
| Other (onnx, onnxslim, datasets, pyyaml, wget, etc.) | ~300 MB | All needed |

---

## Priority 1 — Quick Wins (low risk, do immediately)

### 1a. Clean up `requirements.txt`

```diff
 onnxruntime-gpu
-onnxconverter-common         # transitive dep of onnxruntime-gpu; verify then remove
 numpy
 ultralytics
-torch                        # already pulled in by ultralytics; fine to keep for pinning
-torchvision                  # same
 pyyaml
 wget
-tensorboard                  # NOT imported anywhere in the codebase — remove
 onnx
-onnxscript                   # verify: may be a dep of ultralytics ONNX dynamo path (we use dynamo=False)
 onnxslim
 datasets
 Pillow
-opencv-python                # replace with headless variant (saves Qt/GUI shared libs)
+opencv-python-headless
-textual[dev]                 # [dev] installs pytest, hypothesis, etc. — not needed in container
+textual
```

**Estimated saving: 600–750 MB** (mostly tensorboard and its heavy transitive deps).

> Note: before removing `onnxconverter-common` and `onnxscript`, verify they are not needed by
> running the export inside the container and checking for import errors. They are likely already
> pulled in by `onnxruntime-gpu` / `ultralytics` and do not need to be listed explicitly.

### 1b. Fix `COPY` scope — stop copying the whole repo

Replace the broad `COPY . .` with an explicit copy of only the app code:

```dockerfile
# Only the package we actually need inside the container
COPY PytorchWildlife_Export/ ./PytorchWildlife_Export/
```

`CameraTraps/` is not used by any code in `PytorchWildlife_Export/`. It should not be copied.

### 1c. Strengthen `.dockerignore`

Add the following entries to prevent model artifacts, scratch files, and docs from entering the
build context and invalidating the `COPY` layer:

```
# Model artifacts — never copy weights/engines into the image
*.pt
*.onnx
*.engine
*.npy

# Profiling output
onnxruntime_profile_*.json

# Scratch / dev files
examine.py
image_prep.py
images/
CameraTraps/
PT_2_ONNX_2_TRT.md
QUANTIZE_PLAN.md
PREPROCESS_PLAN.md
IMAGE_SHRINK.md
cache/

# Git history
.git/
```

### 1d. Fix demo script output directories

All demo scripts currently hard-code their output inside the codebase:

```python
# yolov10_v9_trt.py  (and all other demo/*.py)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_output"))
```

This writes model files into `PytorchWildlife_Export/demo/demo_output/`, which then risks getting
picked up by `COPY . .` in future builds, and makes the `demo_output` volume mount in
`run_tensorrt_test.sh` fragile. Change to:

```python
OUTPUT_DIR = "/exported_models"
```

The `run_tensorrt_test.sh` mounts `exported_models/` (host) → `/exported_models` (container)
and already creates that dir on the host. No other changes needed there.

### 1e. Add output path guardrail in `export_tool.py`

Add a check immediately after `args = parser.parse_args()` to ensure output paths are never
written inside the source tree:

```python
# Resolve the output path and the codebase root
_output = Path(args.output_path).resolve()
_pkg_root = Path(__file__).resolve().parent          # PytorchWildlife_Export/
_repo_root = _pkg_root.parent                        # repo root

# Refuse to write inside the source package or above the repo root
_forbidden = [_pkg_root, _repo_root]
for _f in _forbidden:
    try:
        _output.relative_to(_f)
        # If we get here the path IS inside _f
        if _f == _pkg_root or _output == _repo_root:
            print(f"Error: output_path must not be inside {_f}. "
                  f"Use /exported_models or a path outside the repository.")
            sys.exit(1)
    except ValueError:
        pass  # not relative to _f — good
```

The rule: output must be below `_repo_root` but not inside `_pkg_root`, OR it must be an absolute
path outside the repository entirely (e.g. `/exported_models`).

### 1f. Add `ARG CACHE_BUSTER` and create `build_tensorrt.sh`

`Dockerfile.trt` does not have a `CACHE_BUSTER` arg, so there is no way to force-invalidate only
the `COPY` layer without `--no-cache` (which also re-runs the expensive `pip install`).

Add to `Dockerfile.trt` (after the `COPY requirements.txt` line):

```dockerfile
# CACHE_BUSTER is only passed when you want to force-refresh the code copy.
# It does NOT invalidate the pip install layer.
ARG CACHE_BUSTER=default
```

And put the `ARG CACHE_BUSTER` reference just before `COPY PytorchWildlife_Export/`:

```dockerfile
ARG CACHE_BUSTER=default
COPY PytorchWildlife_Export/ ./PytorchWildlife_Export/
```

Because `ARG` only affects the build cache if it changes, this lets you do:
`docker build --build-arg CACHE_BUSTER=$(date +%s)` to refresh code without re-running pip.

**`build_tensorrt.sh`** (new file):

```bash
#!/bin/bash
set -e

IMAGE_NAME="pytorch-wildlife-export-trt"

# Usage:
#   ./build_tensorrt.sh            — normal build, uses Docker layer cache
#   ./build_tensorrt.sh --fresh    — forces code layer refresh, keeps pip cache
#   ./build_tensorrt.sh --no-cache — full rebuild, no cache at all

CACHE_BUSTER="stable"
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --fresh)
            # Invalidate only the COPY layer, keep the expensive pip install cache
            CACHE_BUSTER=$(date +%s)
            ;;
        --no-cache)
            EXTRA_ARGS="--no-cache"
            ;;
    esac
done

echo "Building $IMAGE_NAME (CACHE_BUSTER=$CACHE_BUSTER)..."
docker build \
    -f Dockerfile.trt \
    --build-arg CACHE_BUSTER="$CACHE_BUSTER" \
    $EXTRA_ARGS \
    -t "$IMAGE_NAME" \
    .

echo "Build complete: $IMAGE_NAME"
docker images "$IMAGE_NAME"
```

---

## Priority 2 — Multi-stage Build (medium effort, ~400 MB saving)

Some of the packages in `apt-get install` are only needed during `pip install` (cmake,
build-essential, and their headers — required to compile wheels for onnx-simplifier and similar
packages from source). At runtime they are not needed.

Split `Dockerfile.trt` into two stages:

```dockerfile
# ── Stage 1: install Python deps (needs build tools) ───────────────────────
FROM nvcr.io/nvidia/tensorrt:26.01-py3 AS pip_builder

WORKDIR /app
COPY PytorchWildlife_Export/requirements.txt ./PytorchWildlife_Export/

# Install build-time apt packages (cmake, build-essential needed to compile wheels)
RUN apt-get update && apt-get install -y \
        cmake build-essential \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r ./PytorchWildlife_Export/requirements.txt

# ── Stage 2: final runtime image ────────────────────────────────────────────
FROM nvcr.io/nvidia/tensorrt:26.01-py3

WORKDIR /app

# Runtime-only apt packages (no cmake, no build-essential)
RUN apt-get update && apt-get install -y \
        libxcb1 libgl1 libsm6 libxext6 libxrender1 \
        libjpeg-dev libpng-dev libtiff-dev libwebp-dev ffmpeg \
        --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=pip_builder /usr/local/lib/python3.12/dist-packages \
                        /usr/local/lib/python3.12/dist-packages
COPY --from=pip_builder /usr/local/bin /usr/local/bin

# Copy application code
ARG CACHE_BUSTER=default
COPY PytorchWildlife_Export/ ./PytorchWildlife_Export/

RUN mkdir -p /exported_models /root/.cache/torch/hub/checkpoints

ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "PytorchWildlife_Export/tui_export.py"]
```

> **Important caveat:** both stages still use the same `nvcr.io/nvidia/tensorrt:26.01-py3` base,
> so those layers are shared. The saving comes from not including cmake/build-essential in the
> final image's unique layers. On a registry push/pull the shared base layers are not
> re-transferred, so this helps with push/pull size.

---

## Priority 3 — SM-specific Builder Resources (large effort, ~1.5 GB saving)

The biggest removable single chunk in the base image is the per-GPU-architecture builder resources
(`libnvinfer_builder_resource_sm*.so`). These are only needed to **compile** TRT engines for each
GPU microarchitecture. If you know your target GPU ahead of time, you only need the matching file:

| SM arch | GPU family | File | Size |
|---|---|---|---|
| sm75 | Turing (T4, RTX 20xx) | `libnvinfer_builder_resource_sm75.so` | 115 MB |
| sm80 | Ampere A100 | `libnvinfer_builder_resource_sm80.so` | 182 MB |
| sm86 | Ampere (RTX 30xx, A10) | `libnvinfer_builder_resource_sm86.so` | 172 MB |
| sm87 | Ampere (Jetson Orin) | compiled from sm86/sm89 resources | see below |
| sm89 | Ada Lovelace (RTX 40xx) | `libnvinfer_builder_resource_sm89.so` | 181 MB |
| sm90 | Hopper (H100) | `libnvinfer_builder_resource_sm90.so` | 437 MB |
| sm100 | Blackwell Gen1 | `libnvinfer_builder_resource_sm100.so` | 276 MB |
| sm110/120 | Blackwell Gen2+ | `libnvinfer_builder_resource_sm110/120.so` | 518 MB |
| PTX fallback | any | `libnvinfer_builder_resource_ptx.so` | 232 MB |

**For Jetson Orin (SM87, Ampere):** keep `sm80`, `sm86`, `sm89`. Remove all others (sm75, sm90,
sm100, sm110, sm120, ptx). **Savings: ~1.46 GB.**

Because these files come from the base image's immutable layers, a `RUN rm` in a later layer will
not shrink the compressed image size — Docker keeps the data in the lower layer and only adds a
whiteout entry. To truly remove them you must use a clean final stage:

```dockerfile
FROM nvcr.io/nvidia/tensorrt:26.01-py3 AS base_pruned

# Remove SM builder resources not needed for Jetson Orin (SM80/86/89 only)
# This must be in the SAME RUN as any other deletions to collapse into one layer.
RUN rm -f \
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm75.so.* \
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm90.so.* \
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm100.so.* \
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm110.so.* \
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_sm120.so.* \
    /usr/lib/aarch64-linux-gnu/libnvinfer_builder_resource_ptx.so.* \
    /usr/lib/aarch64-linux-gnu/libnccl.so.* \
    /usr/local/cuda-13.1/NsightSystems-cli-2025.6.1 \
    /usr/local/cuda-13.1/compute-sanitizer \
    && rm -rf /usr/local/cuda-13.1/bin \   # nvcc etc — not needed at runtime
              /usr/local/cuda-13.1/nvvm    # PTX compiler backend
```

**BUT** — `RUN rm` in a new layer atop the nvidia base still does not reclaim disk in the image
stored on disk or in a registry. To actually save space you need Docker's `--squash` flag
(experimental) or a true multi-stage where the final stage starts from a completely clean
`ubuntu:24.04` and you manually copy every required shared library. That is the high-maintenance
path the user noted is probably not worth doing.

**Practical recommendation:** defer Priority 3 unless image pull time on Jetson becomes a real
bottleneck. If it does, the cleanest approach would be to maintain a custom pruned base image tag
(`pytorch-wildlife-trt-base:26.01-orin`) built once from the nvidia image with the above deletions
squashed in, and use that as the FROM in `Dockerfile.trt`.

---

## Summary: Expected Size Reductions

| Priority | Change | Estimated Saving |
|---|---|---|
| 1a | Remove tensorboard, fix textual[dev], opencv-headless | ~650 MB |
| 1b–1c | Fix COPY scope, strengthen .dockerignore | ~0 MB image (prevents future bloat) |
| 1d–1e | Fix output dirs + guardrail | 0 MB image (correctness fix) |
| 1f | CACHE_BUSTER + build_tensorrt.sh | 0 MB (build speed / UX) |
| 2 | Multi-stage: separate build tools from runtime | ~400 MB |
| 3 | SM-specific builder resources (Orin only) | ~1.5 GB |
| **Total (P1+P2)** | | **~1 GB** |
| **Total (P1+P2+P3)** | | **~2.5 GB** |

---

## Implementation Order

1. **Do now:** Apply P1a–P1f changes together in one commit (all mechanical, low risk).
2. **Follow-up:** Switch to the multi-stage Dockerfile.trt (P2). Test that the pip packages still
   resolve correctly when copied across stages (watch for `.pth` files and console scripts in
   `/usr/local/bin`).
3. **Later (if needed):** Implement the custom pruned base image (P3), scoped to Jetson Orin.
   Pin the SM set to `sm80,sm86,sm89` and document it clearly so it's easy to update if a new
   target GPU is added.

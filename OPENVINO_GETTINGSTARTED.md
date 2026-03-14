# OpenVINO Getting Started

Goal: verify that the ONNX models produced by this pipeline run correctly under the
OpenVINO runtime, with particular focus on whether our TRT-calibrated INT8 QDQ ONNX
models are directly usable.  If they are not, document what changes are required.

Scope is **benchmarking and profiling only** — no model export is done in the OpenVINO
container.

---

## Host environment

| Item | Value |
|---|---|
| iGPU | Intel UHD Graphics 630 (CoffeeLake-S GT2, Gen9.5) |
| DRI nodes | `/dev/dri/card1`, `/dev/dri/renderD128` (Intel) |
| OpenCL driver | i915/Mesa OpenCL or intel-opencl-icd |
| Level Zero | **Not available** — UHD 630 predates the Xe stack; OpenVINO GPU plugin uses OpenCL |
| Kernel | 6.8.0-90-generic (Ubuntu 22.04) |

> **Verify the Intel DRI node** before running containers.  With two GPUs, numbering is
> not guaranteed.  Run `ls -la /dev/dri/` and cross-reference with
> `lspci | grep VGA` to confirm which render node belongs to the Intel device.

---

## Phase 1 — Assemble the OpenVINO utility image

### 1.1 Dockerfile

Create `Dockerfile.openvino` at the repo root.  The image only needs:

- OpenVINO runtime (for `benchmark_app` and the Python API)
- Intel OpenCL ICD (enables OpenVINO GPU plugin on UHD 630)
- Python + `openvino` wheel (for scripted accuracy checks)
- `clinfo` and `vainfo` (one-time host sanity tools)

```dockerfile
FROM openvino/ubuntu22_runtime:2024.6.0

# Intel OpenCL ICD — required for GPU plugin on Gen9/Gen11 (no Level Zero needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        intel-opencl-icd \
        clinfo \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python packages: openvino wheel, lightweight image I/O
RUN pip3 install --no-cache-dir openvino numpy opencv-python-headless

WORKDIR /app
```

Build:

```bash
docker build -f Dockerfile.openvino -t openvino-util .
```

### 1.2 Verify iGPU access inside the container

Before benchmarking, confirm OpenCL sees the Intel GPU:

```bash
docker run --rm \
    --device /dev/dri/card2:/dev/dri/card2 \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    --group-add $(stat -c "%g" /dev/dri/renderD129) \
    openvino-util \
    clinfo -l
```

Expected output includes a line like:
```
Platform #0: Intel(R) OpenCL HD Graphics
  Device #0: Intel(R) UHD Graphics 630
```

If `clinfo` shows no Intel platform, check that you are mounting `card1`/`renderD128`
(Intel) and not `card2`/`renderD129` (NVIDIA).  Do **not** bind-mount the host
`/etc/OpenCL/vendors` — on this machine it contains only `nvidia.icd` and would
prevent the container's built-in `intel-opencl-icd` from loading correctly.

### 1.3 Makefile target

Add to the Makefile for convenience:

```makefile
OV_IMAGE          ?= openvino-util
OV_DEVICE         ?= GPU   # or CPU — see Phase 2
OV_RENDER_NODE    ?= /dev/dri/renderD129

DOCKER_RUN_OV = docker run --rm \
    --device /dev/dri/card2:/dev/dri/card2 \
    --device $(OV_RENDER_NODE):$(OV_RENDER_NODE) \
    --group-add $(shell stat -c "%g" $(OV_RENDER_NODE)) \
    -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
    -v "$(CURDIR)/exported_models:/models:ro" \
    -v "$(CURDIR)/data:/data:ro" \
    -v "$(CURDIR):/app" \
    --workdir /app \
    $(OV_IMAGE)

ov-bench:
    @echo "--- OpenVINO benchmark: $(MODEL) on $(OV_DEVICE) ---"
    $(DOCKER_RUN_OV) benchmark_app \
        -m /models/$(MODEL) \
        -d $(OV_DEVICE) \
        -hint latency \
        -niter 200 \
        -api sync

ov-bench-throughput:
    @echo "--- OpenVINO throughput benchmark: $(MODEL) on $(OV_DEVICE) ---"
    $(DOCKER_RUN_OV) benchmark_app \
        -m /models/$(MODEL) \
        -d $(OV_DEVICE) \
        -hint throughput \
        -niter 200
```

---

## Phase 2 — Profile the ONNX models

All models benchmarked here are produced by `make sweep-export` and follow the naming
convention `{variant}_{precision}_{size}_denorm_nhwc_uint8input.onnx`.

**Model set:**

| File | Available |
|---|---|
| `MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.onnx` | from sweep-export |
| `MDV6-yolov10-e_float16_320_denorm_nhwc_uint8input.onnx` | from sweep-export |
| `MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.onnx` | from sweep-export |
| `MDV6-yolov10-c_float16_320_denorm_nhwc_uint8input.onnx` | from sweep-export |
| `MDV6-yolov10-e_int8_640_denorm_nhwc_uint8input.onnx` | generate via sweep-export (see 2.0) |
| `MDV6-yolov10-e_int8_320_denorm_nhwc_uint8input.onnx` | generate via sweep-export (see 2.0) |
| `MDV6-yolov10-c_int8_640_denorm_nhwc_uint8input.onnx` | generate via sweep-export (see 2.0) |
| `MDV6-yolov10-c_int8_320_denorm_nhwc_uint8input.onnx` | generate via sweep-export (see 2.0) |

### 2.0 Generate INT8 ONNX models

INT8 ONNX was previously skipped in `sweep_export` (the skip guard and the calibration
arg restriction have now both been removed).  Export all four INT8 ONNX variants using
the TRT container (calibration requires the GPU):

```bash
make sweep-export \
    SWEEP_FORMATS=int8 \
    SWEEP_RUNTIMES=onnx \
    SWEEP_DATASET_YAML=/data/md_ft/megadetector_ft.yaml \
    SWEEP_CALIB_IMAGES=100 \
    SWEEP_SKIP_EXISTING=--skip-existing
```

This runs inside `pytorch-wildlife-export-trt` (the existing TRT image) so calibration
can use the NVIDIA GPU.  Each model takes ~2 min.  Outputs land in `exported_models/`.

### 2.1 Float16 baselines — CPU and GPU

`benchmark_app` auto-detects input shapes from the ONNX graph.  Run latency mode
(single-stream, synchronous) for all eight models × two devices:

```bash
for MODEL in \
    MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.onnx \
    MDV6-yolov10-e_float16_320_denorm_nhwc_uint8input.onnx \
    MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.onnx \
    MDV6-yolov10-c_float16_320_denorm_nhwc_uint8input.onnx; do
    make ov-bench OV_DEVICE=CPU MODEL=$MODEL
    make ov-bench OV_DEVICE=GPU MODEL=$MODEL
done
```

Record `Latency: X ms` and `Throughput: Y FPS` from each run.

### 2.2 INT8 ONNX — CPU and GPU

After completing 2.0, run the same loop for INT8:

```bash
for MODEL in \
    MDV6-yolov10-e_int8_640_denorm_nhwc_uint8input.onnx \
    MDV6-yolov10-e_int8_320_denorm_nhwc_uint8input.onnx \
    MDV6-yolov10-c_int8_640_denorm_nhwc_uint8input.onnx \
    MDV6-yolov10-c_int8_320_denorm_nhwc_uint8input.onnx; do
    make ov-bench OV_DEVICE=CPU MODEL=$MODEL
    make ov-bench OV_DEVICE=GPU MODEL=$MODEL
done
```

**Expected failure modes to watch for:**

- `[GENERAL_ERROR] Failed to read IR` or `Unsupported op` — OpenVINO cannot parse the QDQ graph
- Model loads but `benchmark_app` reports FP32 execution — QDQ nodes were stripped/ignored
  rather than lowered to INT8 kernels (check the "Model precision" line in the output)
- Silent accuracy regression — model loads and runs but produces garbage detections
  (caught in Phase 2.3)

### 2.3 Quick accuracy smoke-test

`benchmark_app` measures throughput only.  Before declaring INT8 working, run a
detection sanity check using the OpenVINO Python API against one of the eval images:

```python
# scripts/ov_smoke_test.py
import sys, numpy as np, cv2
import openvino as ov

model_path, image_path = sys.argv[1], sys.argv[2]
core = ov.Core()
model = core.read_model(model_path)
compiled = core.compile_model(model, "CPU")   # swap to "GPU" as needed

img = cv2.imread(image_path)
h, w = img.shape[:2]
# Resize to model input (read from model.input)
inp = compiled.input(0)
_, ih, iw, _ = inp.shape    # NHWC
blob = cv2.resize(img, (iw, ih))[np.newaxis]   # uint8, NHWC

result = compiled(blob)[compiled.output(0)]   # (1, 300, 6)
dets = result[0]
dets = dets[dets[:, 4] > 0.25]   # confidence threshold
print(f"{len(dets)} detections above 0.25:")
for d in dets:
    print(f"  cls={int(d[5])} conf={d[4]:.3f} box={d[:4].astype(int)}")
```

```bash
docker run --rm \
    --device /dev/dri/renderD129:/dev/dri/renderD129 \
    --group-add $(stat -c "%g" /dev/dri/renderD129) \
    -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
    -v $(pwd)/exported_models:/models:ro \
    -v $(pwd):/app \
    openvino-util \
    python3 /app/scripts/ov_smoke_test.py \
        /models/MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.onnx \
        /app/data/md_ft/images/val/<some_image>.jpg
```

Run this against the float16 model first (expected: detections match TRT results),
then against each INT8 model.

### 2.4 Throughput mode (optional, for Frigate context)

For NVR use cases, asynchronous throughput matters more than single-frame latency.
Run throughput mode after confirming correctness:

```bash
make ov-bench-throughput OV_DEVICE=GPU MODEL=MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.onnx
make ov-bench-throughput OV_DEVICE=GPU MODEL=MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.onnx
```

---

## Phase 3 — Analyze results and plan next steps

### 3.1 Results table template

Fill this in as benchmarks complete.

| Model | Size | Precision | Device | Latency (ms) | Throughput (FPS) | Status |
|---|---|---|---|---|---|---|
| yolov10-e | 640 | float16 | CPU | | | |
| yolov10-e | 640 | float16 | GPU | | | |
| yolov10-c | 640 | float16 | CPU | | | |
| yolov10-c | 640 | float16 | GPU | | | |
| yolov10-e | 320 | float16 | CPU | | | |
| yolov10-e | 320 | float16 | GPU | | | |
| yolov10-c | 320 | float16 | CPU | | | |
| yolov10-c | 320 | float16 | GPU | | | |
| yolov10-e | 640 | int8 (trt_quant) | CPU | | | |
| yolov10-e | 640 | int8 (trt_quant) | GPU | | | |
| yolov10-e | 640 | int8 (mixed_demo) | CPU | | | |
| yolov10-e | 640 | int8 (mixed_demo) | GPU | | | |

Status values: ✓ correct | ⚠ runs but accuracy unknown | ✗ load error | — not tested

### 3.2 Decision tree for next steps

#### Case A — float16 ONNX loads and runs correctly

This is the expected happy path.  Proceed to INT8 testing.

If GPU latency is worse than CPU latency: this is normal for UHD 630 on small batch
sizes.  The UHD 630 has limited bandwidth and EU count — it is unlikely to beat a
modern CPU on per-frame inference for this model size.  Throughput mode (`-hint throughput`)
may be more competitive due to pipelining.

#### Case B — INT8 QDQ ONNX loads and produces correct detections

If `benchmark_app` confirms the model runs in INT8 (look for `Precision: INT8` in output)
and the smoke test shows correct detections:

1. Enable INT8 ONNX export in `sweep_export.py` (currently skipped with the comment
   "no benefit without TRT fusion") — the comment was written assuming ORT CUDA EP; it
   does not apply to OpenVINO.
2. Add `openvino` as a recognized runtime in `naming.py` and `sweep_export.py`.
3. Run `sweep_eval` against the INT8 ONNX files using the OpenVINO Python backend to
   get mAP numbers (needs an `ov` backend path in `eval.py` or a separate evaluator).

#### Case C — INT8 QDQ ONNX loads but OpenVINO silently ignores QDQ (runs in FP32/FP16)

OpenVINO 2024.x can sometimes import QDQ ONNX but strip the quantization nodes if it
does not recognize the calibration scale format.  Signs: model loads, but `benchmark_app`
reports FP32 precision and latency matches the float16 model.

Fix options:
1. **OpenVINO NNCF post-training quantization** — use `nncf.quantize()` on the float16
   ONNX with a calibration dataloader drawn from our existing `data/md_ft/` dataset.
   This produces an OV-native INT8 model that the GPU plugin will execute in INT8.
   This is the recommended path and is likely one or two days of work.
2. **Convert to OpenVINO IR first** — run `ovc MDV6-yolov10-e_float16_640.onnx` to
   produce `.xml`/`.bin` IR, then apply NNCF quantization to the IR.  Adds a step but
   ensures better optimization by the OV compiler.

#### Case D — INT8 QDQ ONNX fails to load (graph parse error)

The TRT-style QDQ nodes use `QuantizeLinear` / `DequantizeLinear` with per-channel
axis attributes that OpenVINO may reject if the opset version or attribute format
differs from what it expects.  Signs: `[GENERAL_ERROR]` on model load.

Check:
- Run `python3 -c "import openvino as ov; ov.Core().read_model('/models/..._int8.onnx')"`
  for a cleaner error message than `benchmark_app` gives.
- Run `polygraphy inspect model /models/..._int8.onnx --mode onnx` (if polygraphy is
  available) to see which nodes are problematic.

Fix: same as Case C — NNCF quantization on the float16 ONNX is cleaner than trying
to fix the QDQ graph for OV compatibility.

#### Case E — GPU plugin unavailable or slower than expected

UHD 630 is a Gen9.5 GPU with 24 EUs and ~192 GB/s memory bandwidth.  For the
models in this repo (640px input, ~20M parameters for -e), the GPU may not be
faster than CPU in single-stream mode.  This is acceptable — the goal of this
investigation is correctness of INT8, not outperforming TRT on a discrete GPU.

For a Frigate deployment scenario on a system with only this iGPU, CPU inference
with OpenVINO may be the practical recommendation.

---

## Appendix: useful diagnostic commands

```bash
# Check what devices OpenVINO sees
docker run --rm --device /dev/dri $(OV_RENDER_NODE) \
    -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
    openvino-util \
    python3 -c "import openvino as ov; print(ov.Core().available_devices)"

# Inspect ONNX model inputs/outputs (useful before benchmarking)
docker run --rm -v $(pwd)/exported_models:/models:ro openvino-util \
    python3 -c "
import openvino as ov
m = ov.Core().read_model('/models/MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.onnx')
for i in m.inputs:  print('IN: ', i.get_any_name(), i.shape, i.element_type)
for o in m.outputs: print('OUT:', o.get_any_name(), o.shape, o.element_type)
"

# Check OpenCL devices visible inside container
docker run --rm --device /dev/dri \
    -v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro \
    openvino-util clinfo -l

# Dump per-layer precision after compilation (confirms INT8 lowering)
# (Run this script inside the container)
python3 - <<'EOF'
import openvino as ov
core = ov.Core()
model = core.read_model("/models/MDV6-yolov10-e_trt_quant_int8.onnx")
compiled = core.compile_model(model, "GPU")
for node in compiled.get_runtime_model().get_ordered_ops():
    print(node.get_type_name(), [str(o.get_element_type()) for o in node.outputs()])
EOF
```

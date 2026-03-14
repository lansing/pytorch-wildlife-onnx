# Path to INT8 Kernel Execution in ORT

## Executive Summary

**ORT's CUDA Execution Provider has no native INT8 Conv kernels and never will in its current
architecture.** QDQ-wrapped Conv nodes on the CUDA EP always run as:

```
QuantizeLinear (float→int8, CUDA cast)
DequantizeLinear (int8→float, CUDA cast)
Conv (float32, cuDNN)
QuantizeLinear (float→int8, CUDA cast)
DequantizeLinear (int8→float, CUDA cast)
```

No actual INT8 arithmetic occurs. The profile confirms this: 594 Q/DQ events, 0 INT8 fused
kernels, all Conv nodes tagged `fp32`. The only path to GPU INT8 computation with ORT is the
**TensorRT Execution Provider** using *explicit quantization* mode.

---

## Why the CUDA EP Cannot Fuse QDQ+Conv

### The fusion transformer is EP-aware

ORT's `QDQSelectorActionTransformer` (Level2 graph optimization) contains a Conv fusion rule
with an explicit provider allowlist:

```
CPU EP, DirectML EP, ACL EP
```

CUDA EP is intentionally absent. The transformer is aware that CUDA EP has no `QLinearConv`
kernel and skips CUDA-owned Conv nodes entirely.

### No QLinearConv kernel on CUDA

ORT's CUDA Conv kernel (`conv.cc`) is only registered for `float`, `double`, `MLFloat16`,
and `BFloat16`. There is no `int8` registration. cuDNN does expose INT8 convolution internally,
but ORT has not wired a kernel to it. This holds for ORT 1.11 through at least 1.24.

### The bias DequantizeLinear runs on CPU

A secondary consequence: bias tensors (int32 initializers post-matmul scaling) are small enough
that ORT assigns their `DequantizeLinear` to the CPU EP, then issues a `MemcpyFromHost` to feed
the result into the CUDA Conv. This is the 76-CPU-event, 76-MemcpyFromHost pattern observed in
the profile — one per quantized Conv per inference call.

---

## The Correct Path: TensorRT EP with Explicit Quantization

### Two TRT INT8 modes — and why we want the non-obvious one

| Mode | `trt_int8_enable` | Network flag | Model format | Calibration |
|---|---|---|---|---|
| **Implicit quantization** | `True` | `WEAKLY_TYPED` | Float32 model | Calibration table required |
| **Explicit quantization** | `False` | `STRONGLY_TYPED` | QDQ model | NOT required (scales embedded in graph) |

Our model already has calibrated Q/DQ nodes with embedded scale/zero-point — this is the
explicit quantization format. Setting `trt_int8_enable=True` would activate implicit
quantization, which conflicts with embedded Q/DQ nodes and causes TRT to reject the model or
ignore the scales.

The correct session setup for our QDQ model:

```python
providers = [
    ('TensorrtExecutionProvider', {
        'trt_fp16_enable': False,          # optional — True allows mixed FP16/INT8 layers
        'trt_int8_enable': False,          # explicit quant: scales come from Q/DQ nodes
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': '/exported_models/trt_cache',
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
        'trt_detailed_build_log': True,    # log which layers go INT8 during first build
    }),
    ('CUDAExecutionProvider', {}),
    ('CPUExecutionProvider', {}),
]
session = ort.InferenceSession(qdq_model_path, providers=providers)
```

**On first run:** TRT parses the QDQ graph, identifies the embedded scale/zero-point values, and
builds an optimized INT8 engine. This compilation step takes 30–120 seconds but is cached.
Subsequent runs load the cached engine directly.

**On the profiling side:** The ORT profile will show TRT-assigned nodes under
`TensorrtExecutionProvider` rather than `CUDAExecutionProvider`. The `profile_analysis.py`
script already handles this via `PROV_SHORT` — add `"TensorrtExecutionProvider": "TRT "` and
look for INT8 in the `input_type_shape` fields of TRT-internal events.

---

## Graph Requirements for TRT Explicit Quantization

TRT's ONNX parser (`nvonnxparser::IParser`) accepts QDQ nodes as explicit precision constraints
under `STRONGLY_TYPED` networks. However it enforces stricter structural requirements than ORT's
`calibrate_conv_nodes_scales`:

### 1. Symmetric quantization (signed INT8 with zero-point = 0)

TRT requires `zero_point = 0` for activation tensors and weight tensors. Asymmetric
quantization (nonzero zero-point, as used in uint8 QAT) is not supported for Conv INT8.

Our current implementation uses `zero_point = 0` (the `zp_name` scalar is always 0). ✓

### 2. Per-tensor activation scales (not per-channel)

TRT supports per-channel weight quantization but requires per-tensor activation scales for most
fusion patterns. Our calibration function computes `max(abs(tensor)) / 127` which is per-tensor.
✓

### 3. Dedicated QDQ pairs per consumer node

TRT cannot share a single Q→DQ pair between two downstream nodes. If tensor `X` feeds into both
`Conv_A` and `Conv_B`, each must get its own independent Q→DQ pair with separately named
intermediate tensors.

Our `_qdq_pair()` helper uses `node_prefix` scoping to ensure this. ✓

### 4. Opset ≥ 13 for QuantizeLinear / DequantizeLinear

The opset 13 signature makes `axis` an attribute (not an input), which TRT's parser requires for
per-channel weight quantization. Our model uses the default ONNX opset from the export
(`opset_imports` copied from the base model). Verify with:

```bash
python3 -c "
import onnx
m = onnx.load('exported_models/MDV6-yolov10-c_quant_demo_mixed_int8.onnx')
for op in m.opset_import: print(op.domain or 'ai.onnx', op.version)
"
```

Expect `ai.onnx 17` or higher (YOLOv10 default export). ✓

### 5. Q/DQ scope covers both inputs and output

TRT fuses a Conv into INT8 only if it sees Q/DQ pairs on ALL inputs (activation + weight) AND
on the output. Our `wrap_node_in_int8_qdq` inserts:
- Q→DQ on activation input ✓
- DQ on pre-quantized INT8 weight ✓
- Q→DQ on Conv output ✓

Bias stays float32 — TRT accumulates bias in int32 internally and this is expected. ✓

---

## Gaps to Resolve Before TRT EP Will Accept the Graph

### Gap 1: Bias DequantizeLinear nodes

Our current quant wrapping adds `DequantizeLinear` nodes for the bias tensor (int32 → float32)
and runs them on CPU. TRT's ONNX parser may reject or ignore these. The standard pattern for
TRT-compatible INT8 Conv is to pass the bias as float32 directly (no DQ node around it). Check
whether our bias handling introduces a DQ node that TRT cannot parse. If so, remove the bias
DQ and pass the original float32 bias directly to the Conv — which is what our code already does
(`inp_b` is passed through unchanged without wrapping).

Re-confirm by reading `wrap_node_in_int8_qdq` in `quant.py`: the bias input `inp_b` is added
to `new_conv_inputs` directly without any Q/DQ wrapper. ✓

### Gap 2: SSA tensor naming

ORT accepts our current SSA-scoped naming. Verify TRT does not object to the long tensor names
(e.g. `YOLO/model.5/conv/Conv__YOLO/model.5/act/input__q_int8`). TRT generally has no name
length constraints but this should be confirmed in the TRT build log.

### Gap 3: Nodes that TRT cannot INT8-fuse

TRT will silently fall back to float32 for layers it cannot fuse. Layers where this is expected
and correct (based on our prior analysis):
- `YOLO/model.23/dfl/conv/Conv` — weights are `[0..15]`, upstream Softmax output → excluded ✓
- `YOLO/model.10/attn/MatMul_1` — upstream Softmax output → should be excluded ✓
- `one2one_cv3.*.2/Conv` and `one2one_cv2.*.2/Conv` — output heads, excluded ✓

These should remain excluded from QDQ wrapping and will run fp32 in TRT. No action needed.

---

## Implementation Plan

### Step 1: Add TRT EP support to `ONNXInferenceSession`

Extend `preferred_provider` handling to configure `TensorrtExecutionProvider` with the
explicit-quantization options above. Add a `trt_cache_path` init argument. The key requirement:
`trt_int8_enable=False` when the model already contains Q/DQ nodes.

### Step 2: Validate graph acceptance

On first TRT EP run, enable `trt_detailed_build_log=True` and check:
- Which layers were assigned to INT8 vs FP32 TRT layers
- Any parser errors for specific nodes
- Engine build warnings about fallback layers

The TRT build log prints per-layer precision decisions. This is the ground-truth check that
our Q/DQ nodes are being recognized as precision constraints.

### Step 3: Run `benchmark()` + `profile_analysis.py`

After TRT engine compilation (second run will load from cache):

```bash
python profile_analysis.py profile.json --warmup-runs 20 --total-runs 120
```

Expected output for a correctly INT8-fused TRT run:
- Provider: `TensorrtExecutionProvider` for most/all node events
- Quantization mode: `INT8 kernel` > 0 (TRT-internal events may expose INT8 I/O types)
- Dramatic latency reduction vs CUDA EP float: expect 1.5–2.5× on Ampere+

Note: TRT EP profiles may show fewer events (TRT fuses entire subgraphs into single engine
calls), so the slow-node table will be less granular. This is expected behavior.

### Step 4: Tune excluded nodes using the TRT build log

If specific nodes fall back to FP32 in the TRT log, decide whether to:
- Accept the fallback (output-facing nodes, Softmax-upstream nodes)
- Add them to the `exclude` list in `wrap_nodes_in_int8_qdq`
- Investigate whether the Q/DQ placement is correct for that node type

### Step 5: Quantify actual accuracy/latency trade-off

Run the existing confidence test (`run_quant_test.sh`) with the TRT EP session to measure:
- Confidence on the test image (target ≥ 0.95)
- End-to-end latency via `benchmark()` (compare vs float TRT EP baseline)

---

## What Not to Do

| Approach | Why it does not work |
|---|---|
| `trt_int8_enable=True` with QDQ model | Activates implicit quantization; conflicts with embedded Q/DQ scales; TRT rejects the model |
| `SessionOptions.graph_optimization_level = ORT_ENABLE_ALL` on CUDA EP | QDQSelectorActionTransformer excludes CUDA EP; no effect on INT8 fusion |
| Replacing Q/DQ with `QLinearConv` nodes | CUDA EP has no QLinearConv kernel; TRT ONNX parser does not support QLinearConv either |
| `QuantFormat.QOperator` (ORT quantization tool) | Produces QLinearConv nodes — same dead end |
| Adding `kCudaExecutionProvider` to INT8 session for anything other than Q/DQ cast ops | Zero INT8 arithmetic capability; confirmed across all ORT versions through 1.24 |

---

## Reference

- ORT source: `core/optimizer/qdq_transformer/selectors_actions/qdq_selector_action_transformer.cc`
- ORT source: `core/providers/cuda/nn/conv.cc` (no int8 registration)
- ORT issue #24807: "CUDA kernel not found for QLinearConv" — 13× CPU fallback slowdown
- ORT issue #12229: "INT8 quantization only improves the CPU with CUDA EP"
- TRT docs: "Explicit vs Implicit Quantization" — developer.nvidia.com/tensorrt
- TRT docs: `NetworkDefinitionCreationFlag::kSTRONGLY_TYPED` — required for QDQ models

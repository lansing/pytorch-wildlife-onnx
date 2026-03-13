# Quantization Sensitivity Analysis: YOLOv10 Detection Head

## Why Conv[82] (and Conv[81]) Cause Model Collapse

### Conv[82]: `YOLO/model.23/dfl/conv/Conv` — the DFL layer

This is the most sensitive node in the entire model. It is **not** a feature-extraction Conv — it is a fixed mathematical operator called the **Distribution Focal Loss (DFL)** expected-value layer.

```
input:  YOLO/model.23/dfl/Softmax_output_0   ← output of a Softmax!
weight: shape (1, 16, 1, 1), range [0.0, 15.0], std=4.61
output: feeds into bounding-box decode (Reshape → Concat → final output)
```

**What DFL computes:** The weight tensor contains the fixed values `[0, 1, 2, …, 15]`.
The Conv is computing a weighted sum — i.e., the *expected value* of a probability distribution:

```
output = sum(p_i * i,  i=0..15)    where p_i = Softmax(logits)[i]
```

This is the core of YOLOv10's bounding-box regression head. Its job is to decode a
predicted distribution over discrete offsets into a continuous box coordinate.

**Why INT8 destroys it:**

1. **Input is a Softmax output** — values are in (0, 1), often very small (e.g. 0.003).
   Quantizing with scale ≈ 1/127 maps these to 0 or ±1 in INT8, wiping out almost all
   the probabilistic information.

2. **Weights are special fixed constants** — [0..15] are not learned features; they are a
   "ruler". Any error in the probability inputs gets multiplied by values up to 15,
   amplifying the quantization error linearly.

3. **Output is the final box coordinate** — there is no downstream smoothing. The error
   maps directly onto the predicted bounding box position.

### Conv[81]: `YOLO/model.23/one2one_cv3.2/one2one_cv3.2.2/Conv` — the class head

```
input:  activation from cv3.2 branch
weight: shape (3, 64, 1, 1), range [-0.71, 0.31]
output: feeds into Reshape_5 → Concat → final output
```

**Only 3 output channels** (one per class: animal, person, vehicle). This is the final
classification Conv. With 3 output values total, each INT8 rounding step shifts scores
by `scale × 1_LSB`, which corresponds to a meaningful fraction of the confidence range.
There is no averaging across many channels to absorb the error.

### Conv[80]: boundary case

```
weight shape: (64, 64, 1, 1), std=0.2577
```

Also inside `model.23` (detection head). Its output feeds into the same cv3.2 branch
that produces the class scores. With n=81 you include this node, giving a slight confidence
drop (0.95 → 0.93) — it tolerates quantization because it has 64 output channels to
absorb the error, but it's near enough to the final output to show up.

---

## General Rules for Safe Quantization

### 1. Avoid detection head nodes (`model.23`)
All Convs in `model.23` are in the v10Detect head:
- `cv2.*` branch → box regression
- `cv3.*` branch → class scores
- `dfl` → box coordinate decoding

**Safe cutoff: quantize Conv[0..79], skip Conv[80..82].**

### 2. Avoid Convolutions whose input is a Softmax or Sigmoid output
Softmax/Sigmoid compress values into (0, 1). INT8 with a naive scale typically maps most
of these values to a handful of integer levels, destroying the distribution. In this model,
Conv[82]'s input is literally labeled `dfl/Softmax_output_0`.

Programmatic check: walk the graph backwards from the Conv's activation input; if you hit
a Softmax (or a Sigmoid feeding directly into the Conv without intervening normalization),
that Conv is high-risk.

### 3. Avoid Convolutions with very few output channels
Conv weight shape `(C_out, C_in, kH, kW)`. When `C_out` is small (1, 3, 4), each output
channel carries unique semantic meaning and has no redundancy to absorb quantization error.
**Rule of thumb: skip any Conv with C_out < 16.**

### 4. Avoid Convolutions with unusual weight distributions
Conv[82] has `std=4.61` on its weight range [0, 15] — the highest in the model by far.
This means per-tensor symmetric quantization wastes most of the INT8 range on values near
0 while the extremes (0 and 15) push against the clipping boundary.
**Heuristic: flag any Conv where `weight_std / weight_max > 0.25` as high-risk.**

### 5. Avoid Convolutions that directly produce graph outputs
Walk the ONNX graph output set and trace backwards through Reshape/Concat/Transpose to find
which Conv nodes actually produce the final tensors. These nodes have no downstream error
correction and must stay in float32 (or float16 at minimum).

### 6. Per-channel weight quantization significantly extends the safe range
The current implementation uses per-tensor weight scale (`scale = max(abs(W)) / 127`).
Per-channel scale (one scale per output channel) reduces weight quantization error by up
to 5× for layers with high inter-channel variance. This would likely make Conv[80] safe and
reduce the drop seen at Conv[81].

### 7. Histogram/percentile activation calibration vs. max
The current calibration uses `running_max`. A single outlier activation drives the scale
up, costing precision on the 99.9th percentile of actual values. Using the 99.9th percentile
of a histogram of activations typically tightens the scale and reduces quantization error —
especially important for head nodes if you choose to quantize them.

---

## Other Compute-Heavy Nodes Worth Quantizing

### MatMul — highest priority after Conv

The graph contains 2 MatMul nodes, both in the PSA (Partial Self-Attention) block at `model.10`:

```
YOLO/model.10/attn/MatMul     inputs: Transpose_output × Split_output_1
YOLO/model.10/attn/MatMul_1   inputs: Split_output_2   × Transpose_1_output
```

These are the Q×K and attn×V products of an attention mechanism. MatMul is one of the
highest-arithmetic-intensity operations on GPU and benefits more from INT8 than Conv does
(cuBLAS INT8 GEMM vs fp16 GEMM can be 2–4× faster on Ampere/Turing). Quantization strategy:
- Treat each MatMul input as an activation (calibrate scale from data)
- Per-tensor symmetric INT8 is standard for attention Q/K/V products

Unlike the detection head Convs, the attention MatMuls are in the backbone feature extractor.
Errors here are smoothed by many subsequent layers before reaching the output.

### Sigmoid+Mul (SiLU activation) — paired with Conv, free speedup

The model has 70 `Sigmoid` + 70 `Mul` nodes implementing `SiLU(x) = x * sigmoid(x)`.
These follow nearly every Conv. If you quantize a Conv to INT8 and also route the SiLU
through INT8 (keeping the activation tensor in INT8 rather than casting back to float32
after every Conv output), you avoid costly INT8→fp32→INT8 round-trips. This requires
fusing the QDQ pattern across the Conv+SiLU subgraph rather than wrapping only the Conv.
ORT and TensorRT both support this fusion — it's the "end-to-end INT8 subgraph" that
full graph quantization pipelines use.

### MaxPool — low risk, moderate gain

3 `MaxPool` nodes in the backbone. MaxPool in INT8 is mathematically equivalent to float32
(it only compares values, no arithmetic). It's safe to run in INT8 if the preceding Conv's
output is already INT8, avoiding a dequantize step. Minimal standalone speedup but
important for subgraph continuity.

### Resize (interpolation) — keep in float32

2 `Resize` nodes (nearest/bilinear upsampling for the FPN neck). These are memory-bandwidth
bound, not compute-bound, and bilinear interpolation in INT8 can introduce noticeable
spatial artifacts. Leave in float32.

### Softmax — keep in float32

2 `Softmax` nodes (one in the attention block, one in DFL). Both are numerically sensitive
and not compute-bound. Float32 is strongly recommended.

---

## Recommended Quantization Strategy for Maximum Speedup

| Layer group | Action | Rationale |
|---|---|---|
| Conv[0..79] backbone+neck | INT8 QDQ with calibrated scales | Safe, high compute volume |
| Conv[80..82] detection head | float16 at most; avoid INT8 | Directly produce outputs, low channel count, DFL is mathematically special |
| MatMul[0..1] attention | INT8 QDQ with calibrated scales | High arithmetic intensity, backbone position |
| Sigmoid+Mul (SiLU) after INT8 Conv | Fuse into INT8 subgraph | Avoids DQ/Q round-trips between Conv and activation |
| MaxPool | Run INT8 if input is already INT8 | Free, lossless |
| Softmax, Resize | float32 | Numerically sensitive / bandwidth-bound |

This strategy should yield 80–90% of the theoretical INT8 speedup while keeping the
detection head in a safe precision regime.

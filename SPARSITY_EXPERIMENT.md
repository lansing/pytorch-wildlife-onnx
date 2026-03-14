# Sparsity Experiment Proposal: 2:4 Structured Pruning for Ampere

**Goal:** Determine whether NVIDIA 2:4 structured sparsity delivers meaningful
throughput improvement on Ampere hardware, using crude magnitude-based pruning
with no retraining. Accuracy is intentionally not the primary concern at this
stage — this is a ceiling test to decide whether proper sparse QAT is worth
pursuing.

---

## Background: NVIDIA 2:4 Structured Sparsity

Ampere Tensor Cores (CC 8.0+) include dedicated hardware for **2:4 structured
sparsity**: for every group of 4 consecutive values in a weight row, exactly
2 must be zero. The hardware stores only the 2 non-zero values plus a 2-bit
index bitmask, halving the weight memory and skipping the zero multiply-adds
entirely.

- **Theoretical speedup**: 2× for the dense matrix math inside each sparse
  layer (GEMM / Conv). Actual speedup is lower due to memory bandwidth, other
  layers, and overhead.
- **Precision**: works for both FP16 and INT8 weights. INT8 sparse Tensor Core
  gives up to 4× INT8 throughput vs FP32 baseline (2× for INT8 over FP16,
  another 2× for sparsity).
- **TRT support**: `BuilderFlag::kSPARSE_WEIGHTS`. TRT verifies the 2:4 pattern
  in the weight initializers at engine build time and selects sparse kernels
  where the pattern is satisfied. Layers whose weights do NOT satisfy 2:4 fall
  back to dense kernels silently — no error, just no speedup.
- **Requirement**: weights must be physically laid out in 2:4 pattern in the
  ONNX initializer before engine build. The pattern is enforced per row of the
  weight matrix when reshaped to 2D ([C_out, C_in × kH × kW] for Conv).

---

## Crude Pruning Approach

Magnitude-based 2:4 pruning: in each group of 4 consecutive elements in a
weight row, zero out the 2 with the smallest absolute value. This enforces
the 2:4 pattern with the minimum change to each weight row.

```python
def apply_2_4_sparsity(w: np.ndarray) -> np.ndarray:
    """
    Enforce 2:4 structured sparsity on weight tensor w.
    w is reshaped to [C_out, C_in * kH * kW], then each row is processed
    in groups of 4 — the 2 smallest-magnitude elements per group are zeroed.
    """
    orig_shape = w.shape
    rows = w.shape[0]
    cols = int(np.prod(w.shape[1:]))
    # Pad cols to multiple of 4
    pad = (-cols) % 4
    w2d = np.pad(w.reshape(rows, cols), [(0, 0), (0, pad)])
    # Group: [rows, groups, 4]
    grouped = w2d.reshape(rows, -1, 4)
    # Zero the 2 smallest-magnitude entries per group
    idx = np.argsort(np.abs(grouped), axis=-1)
    np.put_along_axis(grouped, idx[:, :, :2], 0.0, axis=-1)
    return grouped.reshape(rows, -1)[:, :cols].reshape(orig_shape)
```

**What gets pruned:** All Conv weight tensors that are currently being
quantized (i.e., all nodes in `node_types` that are not excluded). The
detection head, PSA attention block, and model.0 remain excluded (same
as current INT8 excludes list) so accuracy-critical and format-incompatible
layers are untouched.

**Accuracy expectation:** Crude magnitude pruning without retraining will
cause accuracy regression. The magnitude of regression depends on how
weight-redundant the model already is. YOLOv10 was trained without any
sparsity constraint, so many weight groups will have two dominant values
and two near-zero values (natural sparsity in trained nets), but others
will lose significant signal. Expect mAP drop in the range of 2–10 points
vs the dense INT8 baseline; acceptable for a feasibility test.

---

## Scope of Changes

### 1. New function in `quant.py`: `apply_2_4_sparsity_to_model`

```python
def apply_2_4_sparsity_to_model(
    model: onnx.ModelProto,
    node_types: list[str] = None,
    exclude: list[str] = None,
) -> tuple[onnx.ModelProto, dict]:
    """
    Applies 2:4 magnitude-based structured sparsity to Conv weight
    initializers (and optionally other op types). Modifies weights
    in-place in the returned model.

    Returns:
        (modified_model, stats_dict) where stats_dict contains
        per-layer sparsity percentage and total weight count.
    """
```

This runs **before** INT8 quantization so the INT8 scales are computed from
the already-pruned weights (reflecting the actual values that will be used
at inference). Alternatively it can run after, recomputing INT8 scales — both
are valid; pre-quantization is simpler.

### 2. New flag in `trt_export.py`: `sparse_weights: bool = False`

```python
def onnx2engine(
    ...,
    sparse_weights: bool = False,
) -> str:
    ...
    if sparse_weights:
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
```

This flag tells TRT to attempt sparse Tensor Core kernels for layers whose
weight initializers satisfy the 2:4 pattern. It is a no-op for layers that
don't satisfy it (silent fallback to dense).

### 3. New CLI flag in `export_tool.py`: `--sparse_weights`

```
--sparse_weights   Apply 2:4 magnitude pruning to Conv weights and build
                   the TRT engine with BuilderFlag::SPARSE_WEIGHTS.
                   Intended for Ampere+ hardware. Accuracy will regress
                   without sparse QAT fine-tuning. Default: off.
```

Only valid with `--runtime tensorrt --format int8` (or float16 — 2:4 also
works in FP16 and is worth testing there too since it requires no calibration).

### 4. Wiring in `yolo_exporter.py`

Call `apply_2_4_sparsity_to_model` between steps 1 (base ONNX export) and
2 (INT8 QDQ application) in `export()`:

```python
if sparse_weights:
    base_model = apply_2_4_sparsity_to_model(
        base_model, node_types=["Conv"], exclude=excludes
    )
```

Pass `sparse_weights` through to `onnx2engine` at step 5b.

---

## Experiment Matrix

Run all builds on Ampere and profile with trtexec (same method as before).
For each build: record throughput (RPS), check layer dump for `SPARSE` tactic
names, and run a quick accuracy check on a small validation set (e.g. 100
images from the calibration dataset) to get a relative confidence score.

| Build | Format | Profile | Sparse | Notes |
|-------|--------|---------|--------|-------|
| A | FP16 | — | No | Ampere FP16 dense baseline |
| B | FP16 | — | Yes | FP16 + 2:4 sparsity (no calibration needed) |
| C | INT8 | conv | No | INT8 dense baseline on Ampere |
| D | INT8 | conv | Yes | INT8 + 2:4 (primary interest) |
| E | INT8 | blanket | No | Blanket dense on Ampere |
| F | INT8 | blanket | Yes | Blanket + 2:4 |

Builds A→B measures the pure sparsity speedup without INT8 complexity.
Builds C→D is the primary question: does sparsity compound with INT8?
Builds E/F tests whether blanket finally pays off on Ampere.

---

## What to Look For in the Profile

A positive result looks like:
- TRT layer dump shows tactic names containing `sparse` or `IMMA_SPARSE`
  for the majority of Conv layers
- `Reformatting CopyNode` entries are reduced or eliminated compared to
  the dense build (sparse kernels can have different format requirements)
- Overall throughput increases by >15% over the dense INT8 baseline

A negative result looks like:
- Tactic names remain the same (dense kernels) — means TRT rejected the
  2:4 pattern (possible if weight layout after INT8 quantization doesn't
  match TRT's expected sparse format)
- Throughput is unchanged or regresses
- Many `WARNING: sparse weights validation failed` messages from TRT builder

A partial result looks like:
- Some layers use sparse kernels (large channel-count Convs that clearly
  benefit from Tensor Core) and others don't (small or 3-channel)
- Net speedup 5–10% — may still be worth pursuing with proper sparse QAT

---

## Go / No-Go Criteria

| Outcome | Decision |
|---------|----------|
| >20% throughput gain, <5 mAP regression | Pursue sparse QAT training with the actual dataset; high confidence in ROI |
| 10–20% gain, <10 mAP regression | Pursue sparse QAT if training data is available; acceptable for most deployments |
| 5–10% gain | Marginal; only pursue if model is already at target accuracy and throughput goal is tight |
| <5% gain or regression | 2:4 sparsity not helpful for this model/hardware combination; close investigation |

---

## Notes on INT8 + Sparse Interaction

When combining INT8 QDQ with 2:4 sparsity, the weight quantization step in
`wrap_node_in_int8_qdq` computes the INT8 scale as `max(abs(W)) / 127`. If
`apply_2_4_sparsity` runs first (recommended), the scale is computed from
the pruned weights, which may have a lower max than the original — resulting
in a slightly finer quantization grid (better precision). If sparsity is
applied after INT8 quantization, the INT8 weight values are directly zeroed
(the INT8 zero value is 0 regardless of scale), which is clean and avoids
recomputing scales.

Either order is mathematically valid. Applying sparsity to the float32 weights
before quantization is cleaner and avoids surprises in the INT8 rounding step.

---

## Open Questions to Resolve During Experiment

1. Does TRT 10.x accept 2:4 pattern on INT8 initializers, or only FP16?
   (TRT documentation mentions FP16; INT8 sparse support varies by version.)
2. Does the 2:4 pattern need to be applied in the TRT-internal weight layout
   (which differs from ONNX NCHW layout), or does TRT reorder weights itself
   after reading the ONNX? If the former, our pruning might produce the wrong
   layout and TRT silently falls back.
3. Are there minimum channel-count requirements for sparse kernels? (Suspected:
   C_in × kH × kW ≥ 32 for IMMA sparse; small Convs like model.0 with 3
   input channels won't benefit even with padding.)

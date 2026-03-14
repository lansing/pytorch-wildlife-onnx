# Preprocessing + Channel-Padding INT8 for model.0

**Status:** Deferred — implement after Ampere baseline is established.
**Target hardware:** Ampere and later (CC 8.0+). Not worthwhile on Turing (CC 7.5).

---

## Motivation

model.0 is the largest remaining cost cluster in the INT8 engine on Turing:

| Component | Total (ms / 1022 runs) | Avg (ms) |
|-----------|------------------------|----------|
| Reformat: row-major FP16 → 2-wide vectorized FP16 | ~21 ms | 0.021 |
| model.0 Conv + SiLU (FP16) | ~85 ms | 0.083 |
| model.1 Q node (FP16 → INT8, 320×320×80) | ~62 ms | 0.061 |
| **Total** | **~168 ms** | **~0.164** |

model.0 runs FP16 because its input has 3 channels (RGB), and TRT's INT8
Tensor Core (IMMA) on Turing requires `C_in × kH × kW` to be a multiple
of 16. For model.0: `3 × 3 × 3 = 27`, not a multiple of 16.

On Ampere, the 2:4 structured sparsity feature means zero-padded channels
are effectively free, making the channel-padding approach viable.

---

## Approach

### 1. Preprocessing: uint8 → INT8 with channel zero-padding

Replace the current `uint8 → FP16 → /255 → Transpose` in
`input_preprocessing_wrapper.py` with a path that outputs INT8 directly:

```
uint8 [1, H, W, 3]
  → Pad channels to 32: [1, H, W, 32]  (zeros in channels 3..31)
  → Cast to INT32
  → Multiply by (127.0 / 255.0)        (incorporate normalization into scale)
  → Round + Clamp(-128, 127)
  → Cast to INT8 [1, H, W, 32]         scale = 1/127 (represents [0,1] range)
  → Transpose NHWC → NCHW: [1, 32, H, W] INT8
```

Scale math: normalized float = uint8 / 255; INT8 = round(float / s) where
s = 1/127. Therefore INT8 = round(uint8 × 127/255) ≈ uint8 >> 1.

The preprocessing output is a standard NCHW INT8 tensor. TRT will apply its
own reformat to the tile layout it needs for the Conv kernel — this cost is
similar to the existing reformat and cannot be avoided without custom CUDA.

### 2. model.0 weight zero-padding

In `wrap_node_in_int8_qdq` (or a dedicated pre-pass), before quantizing
model.0's Conv weights:

```python
# Original weight shape: [80, 3, 3, 3]
# Zero-pad input channel dim to 32:
w = np.zeros((80, 32, 3, 3), dtype=np.float32)
w[:, :3, :, :] = original_weights  # [80, 3, 3, 3]
# Now quantize w to INT8 as normal
```

The padded channels have weight = 0 and quantize trivially. The output of
model.0 is mathematically identical to the original (zero inputs × zero
weights = 0 contribution to accumulator).

With 32 input channels × 3×3 kernel = 288 = 18×16: satisfies IMMA
alignment. TRT should select an INT8 Tensor Core kernel for model.0.

### 3. Remove model.0 from excludes (Ampere profile only)

In `yolo_exporter.py`, the `_INT8_EXCLUDES` entry for yolov10/yolov10_v9_compatible
currently excludes `"model.0"`. Remove it when targeting Ampere so the
quantization pipeline wraps model.0 in QDQ.

Consider a separate `_INT8_EXCLUDES_AMPERE` dict or an `architecture` param
to `_apply_int8_qdq` that selects the right exclude list.

### 4. Eliminate the output Q node

With model.0 now producing INT8 output (via its output DQ/Q pair in the QDQ
wrapping), the standalone `model.1/conv/Conv__/model.0/act/Mul_output_0__QuantizeLinear`
node (~62ms) should be absorbed into model.0's QDQ output epilogue. Verify
in the profile that TRT fuses `Conv + SiLU + output_Q` into one kernel.

---

## Expected outcome on Ampere

| Component | Before | After (estimate) |
|-----------|--------|-----------------|
| Reformat before model.0 | ~21 ms | ~10 ms (new format from preprocessing) |
| model.0 Conv + SiLU | ~85 ms FP16 | ~30-40 ms INT8 with 2:4 sparsity on padded weights |
| Output Q node | ~62 ms | 0 ms (absorbed into Conv epilogue) |
| **Total** | **~168 ms** | **~40-50 ms** |

On Turing (no 2:4 sparsity): the padded Conv is 10.7× larger without zero-skipping
→ likely slower overall. Do not enable on Turing.

---

## Implementation checklist

- [ ] `input_preprocessing_wrapper.py`: add `uint8_int8_padded` mode that
      outputs INT8 [1, 32, H, W] with scale 1/127
- [ ] `quant.py` / `wrap_node_in_int8_qdq`: zero-pad model.0 weights to
      [80, 32, 3, 3] before INT8 quantization
- [ ] `yolo_exporter.py`: add `_INT8_EXCLUDES_AMPERE` without `"model.0"`
- [ ] `trt_export.py`: no changes needed (sparse flag handled by sparsity experiment)
- [ ] `export_tool.py` / `tui_config.yaml`: add `--target_arch ampere` flag
      (or similar) to select the Ampere-optimised exclude list and preprocessing mode
- [ ] Profile on Ampere and verify model.0 appears as INT8 Conv + SiLU kernel
      (not FP16) in the layer dump

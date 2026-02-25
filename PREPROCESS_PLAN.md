# INT8 TRT Preprocessing Option Plan

## The Core Obstacle

`EngineCalibrator` is a **local class defined inside `onnx2engine`**, not at module scope:

```python
# Inside onnx2engine():
class EngineCalibrator(trt.IInt8Calibrator):   # <-- local class, new each call
    def get_batch(self, names):
        im0s = next(self.data_iter)["img"] / 255.0   # hardcoded
        im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
        return [int(im0s.data_ptr())]

config.int8_calibrator = EngineCalibrator(dataset=dataset, ...)
```

The `/255.0` is baked into a class we cannot reference from outside. The class is
re-created fresh on every call so there is no stable module-level name to patch.

---

## What calibration data each preprocessing mode requires

The preprocessing wrapper's forward pass is: `uint8 cast → NHWC transpose → /255`.
So the merged ONNX model's input varies per flag combination:

| Flags active              | ONNX input dtype | ONNX input layout | ONNX input range |
|---------------------------|------------------|-------------------|------------------|
| none (baseline)           | float32          | NCHW              | 0–1              |
| nhwc_input only           | float32          | NHWC              | 0–1              |
| denormalized_input only   | float32          | NCHW              | 0–255            |
| uint8_input only          | uint8            | NCHW              | 0–255 (int)      |
| uint8 + denormalized      | uint8            | NCHW              | 0–255 (int)      |
| nhwc + denormalized       | float32          | NHWC              | 0–255            |
| uint8 + nhwc + denormalized | uint8          | NHWC              | 0–255 (int)      |

TRT's calibration interface needs data at the ONNX model's input (i.e. the very first
node), in exactly the dtype and layout shown above.

---

## Three approaches, ordered by complexity

---

### Approach A — "Compensating data" trick (no patching; NHWC only)

`get_batch` does `/ 255.0` on whatever `"img"` contains. For cases where the
desired result is **float32 in range 0–1**, we can yield uint8 data in the
correct layout and let the division do the work:

| Mode            | Yield from loader    | After get_batch's /255   | Correct? |
|-----------------|----------------------|--------------------------|----------|
| baseline        | uint8 NCHW 0–255     | float32 NCHW 0–1         | ✓        |
| nhwc_input only | uint8 NHWC 0–255     | float32 NHWC 0–1         | ✓        |
| denormalized    | float32 NCHW 0–255   | float32 NCHW 0–1 (WRONG) | ✗        |
| uint8_input     | uint8 NCHW           | float32 NCHW (WRONG dtype)| ✗       |

**Summary**: NHWC preprocessing is fully supported today with a trivial loader
change — just yield NHWC-layout uint8 tensors instead of NCHW. No patching needed.

**Denormalized and uint8_input are impossible** via this trick: the division
always produces float32 ≤ 1.0, so you can never get float32 0–255 or uint8 at
the pointer, regardless of what you yield.

**Immediate action**: lift the `nhwc_input` guardrail, update `TRTCalibrationDataLoader`
to yield NHWC when `nhwc_input=True`.

---

### Approach B — `trt.Builder` module-level proxy (patches get_batch on the instance)

This is the user's original idea, adapted for the "local class" reality.

**Mechanism** (four steps):

**Step 1**: Before calling `onnx2engine`, replace `tensorrt.Builder` in the
module namespace with a Python proxy class:

```python
import tensorrt as trt
original_builder_cls = trt.Builder

class _PatchedBuilder:
    def __init__(self, logger):
        object.__setattr__(self, '_real', original_builder_cls(logger))

    def create_builder_config(self):
        real_cfg = object.__getattribute__(self, '_real').create_builder_config()
        return _ConfigProxy(real_cfg, preprocessing_params)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_real'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_real'), name, value)

trt.Builder = _PatchedBuilder
try:
    ultralytics_onnx2engine(...)
finally:
    trt.Builder = original_builder_cls
```

Replacing a module attribute (`trt.Builder = ...`) is standard Python — modules
are regular objects with mutable `__dict__`. ✓

**Step 2**: `create_builder_config()` returns a `_ConfigProxy` that wraps the
real `IBuilderConfig` and intercepts `int8_calibrator =`:

```python
class _ConfigProxy:
    def __init__(self, real_cfg, params):
        object.__setattr__(self, '_real', real_cfg)
        object.__setattr__(self, '_params', params)

    def __setattr__(self, name, value):
        real = object.__getattribute__(self, '_real')
        if name == 'int8_calibrator':
            params = object.__getattribute__(self, '_params')
            import types
            def _patched_get_batch(self_cal, names):
                try:
                    img = next(self_cal.data_iter)["img"]
                    # img already in correct dtype/layout/range from our loader
                    img = img.contiguous().cuda()
                    return [int(img.data_ptr())]
                except StopIteration:
                    return None
            value.get_batch = types.MethodType(_patched_get_batch, value)
            real.int8_calibrator = value
        else:
            setattr(real, name, value)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_real'), name)
```

**Step 3**: Patch `get_batch` on the calibrator instance. Python's attribute
lookup checks `instance.__dict__` before `type.__dict__`, so `types.MethodType`
attached to the instance shadows the class method. TRT's C code calls `get_batch`
via the Python C API (`PyObject_CallMethod`), which follows the same resolution
order. ✓

**Step 4 — the critical risk**: `onnx2engine` eventually calls:
```python
engine = builder.build_serialized_network(network, config)
```
Here `config` is our `_ConfigProxy`, not a real `trt.IBuilderConfig`. TRT's
C/pybind11 bindings will do a type check on this argument. This is the main
risk of the approach.

**Possible mitigations**:
- Make `_ConfigProxy` inherit from `trt.IBuilderConfig`. If TRT's class is
  subclassable from Python (like `trt.IInt8Calibrator` demonstrably is, since
  `EngineCalibrator` inherits from it), this would satisfy pybind11's `isinstance`
  check while our `__setattr__` still intercepts the assignment. **This is the
  most likely path to success and should be tried first.**
- If `trt.IBuilderConfig` is not subclassable: intercept further upstream by
  also patching `_PatchedBuilder.build_serialized_network` to unwrap
  `_ConfigProxy` back to `real_cfg` before forwarding to `self._real`.

**Pros**: continues using ultralytics `onnx2engine` intact; elegant if it works.
**Cons**: relies on two uncertain TRT implementation details (subclassability of
`IBuilderConfig`; C-level type checks). Needs validation on a GPU machine.

---

### Approach C — Fork the INT8 build section (most robust)

Write our own `_onnx2engine_int8_preprocessing()` that re-implements the
~100-line critical section of ultralytics' function with a custom calibrator.
This is the failsafe if Approach B can't be made to work.

The key insight: we only need to change the calibrator class. Everything else
in `onnx2engine` (ONNX loading, layer flags, FP16, DLA, engine serialisation)
can be reproduced verbatim from the ultralytics source or delegated to
non-INT8 setup calls.

Rough structure:

```python
import tensorrt as trt

class _WildlifeCalibrator(trt.IInt8Calibrator):
    def __init__(self, dataloader, dla=None):
        trt.IInt8Calibrator.__init__(self)
        self.data_iter = iter(dataloader)
        self.batch_size = dataloader.batch_size
        self.algo = (
            trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
            if dla is not None
            else trt.CalibrationAlgoType.MINMAX_CALIBRATION
        )

    def get_algorithm(self): return self.algo
    def get_batch_size(self): return self.batch_size

    def get_batch(self, names):
        try:
            img = next(self.data_iter)["img"]   # already correct dtype/layout
            img = img.contiguous().cuda()
            return [int(img.data_ptr())]
        except StopIteration:
            return None

    def read_calibration_cache(self): ...
    def write_calibration_cache(self, cache): ...
```

Then a thin `_onnx2engine_int8_preprocessing()` that copies the TRT builder
logic from ultralytics — the ONNX parsing, builder flags, and
`build_serialized_network` call — but substitutes `_WildlifeCalibrator` for the
local `EngineCalibrator`. This is ~100 lines of stable TRT API code, not
ultralytics-internal logic.

**Call site** (in `yolo_exporter.export_tensorrt`):
```python
if export_format == "int8" and any([uint8_input, denormalized_input, nhwc_input]):
    # preprocessing flags active: use our own INT8 build path
    _onnx2engine_int8_preprocessing(
        onnx_file=merged_onnx_tmp_path,
        engine_file=engine_file,
        calibration_dataloader=calibration_dataloader,
        shape=tuple(final_input_shape),
        ...
    )
else:
    # standard path: delegate to ultralytics
    ultralytics_onnx2engine(...)
```

**Pros**: completely reliable; no dependency on TRT implementation details;
no risk of breakage from pybind11 type checks.
**Cons**: ~100 lines of TRT code to maintain; needs updating if ultralytics
changes their build logic significantly (unlikely — TRT's builder API is stable).

---

## Recommended implementation order

1. **Immediately**: lift `nhwc_input` guardrail; update loader to yield NHWC when
   `nhwc_input=True`. Zero risk, works today (Approach A).

2. **Primary attempt**: implement Approach B. Test whether `trt.IBuilderConfig`
   is subclassable and whether the proxy passes `build_serialized_network`'s type
   check. This can only be validated on the GPU machine.

3. **Fallback**: if Approach B fails at the C type-check, implement Approach C.
   It is strictly more work but is guaranteed to work.

---

## Dataloader changes needed (all approaches)

Regardless of which approach is taken, `TRTCalibrationDataLoader` must produce
data in the correct format for each flag combination. The `_letterbox` step
(resizing/padding) always produces uint8 CHW. We then transform to the target
format:

```python
def _to_calibration_format(self, tensor: torch.Tensor) -> torch.Tensor:
    # tensor: (3, H, W) uint8 CHW at this point

    # 1. Layout
    if self.nhwc_input:
        tensor = tensor.permute(1, 2, 0)   # CHW → HWC

    # 2. Add batch dim
    tensor = tensor.unsqueeze(0)           # → (1, C, H, W) or (1, H, W, C)

    # 3. Dtype and range
    if self.uint8_input:
        pass                               # keep uint8 0–255
    elif self.denormalized_input:
        tensor = tensor.float()            # uint8 → float32, range 0–255
    else:
        tensor = tensor.float() / 255.0    # float32 0–1 (baseline + nhwc cases)

    return tensor.contiguous()
```

Cache key must also include the preprocessing flags so different configurations
cache separately.

---

## Guard changes

Once the approach is implemented:
- Remove the `uint8_input`, `denormalized_input`, `nhwc_input` guardrails from
  `export_tensorrt` (they were added to block these combinations; now they are
  supported).
- Remove them from the TUI as well (the screens should reappear for INT8 TRT).
- The ONNX int8 `NotImplementedError` is unaffected.

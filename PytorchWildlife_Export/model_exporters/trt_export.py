"""
TensorRT engine export utilities.

onnx2engine_explicit: Build a TRT engine from an ONNX model that already
contains QDQ (QuantizeLinear / DequantizeLinear) calibration nodes produced
by wrap_nodes_in_int8_qdq().  TRT reads the embedded per-tensor scales
directly — no calibration table is required.  This is "explicit quantization"
mode.

How explicit quantization works (TRT 10):
  - The ONNX model contains Q/DQ node pairs around each quantized layer's
    inputs and weights.  Each Q/DQ pair carries a float32 scale + int8
    zero-point (always 0 for symmetric quantization).
  - Setting BuilderFlag.INT8 tells TRT that INT8 kernels are allowed.
    When Q/DQ nodes are present TRT's optimizer fuses them with the
    surrounding Conv / MatMul into a single INT8 kernel (explicit
    quantization mode), rather than using an IInt8Calibrator.
  - Setting BuilderFlag.FP16 (optional) lets layers that have no Q/DQ
    coverage (e.g. the excluded detection head) fall back to FP16 instead
    of FP32, giving a small additional speedup.
  - No calibrator object is set — presence of Q/DQ nodes is the signal to
    TRT to use explicit mode.

Compared with ultralytics onnx2engine:
  - INT8 calibration dataset + IInt8Calibrator replaced by Q/DQ nodes.
  - No metadata prefix written (engine can be read with plain file.read()).
  - Verbose logging of parsed I/O tensors is always printed.
"""

from __future__ import annotations

import json
from pathlib import Path


def onnx2engine_explicit(
    onnx_file: str,
    engine_file: str | None = None,
    workspace_gb: float = 4.0,
    fp16_fallback: bool = True,
    verbose: bool = False,
    metadata: dict | None = None,
) -> str:
    """Build a TensorRT engine from an ONNX model with embedded QDQ nodes.

    Steps (mirrors ultralytics onnx2engine for the explicit-quantization path):
        1. Create TRT Logger / Builder / BuilderConfig.
        2. Set workspace memory pool limit.
        3. Create network with EXPLICIT_BATCH flag (required for ONNX + QDQ).
        4. Set INT8 flag — enables INT8 kernels.  With Q/DQ nodes present TRT
           uses their embedded scale/zero-point ("explicit quantization" mode)
           instead of an IInt8Calibrator.
        5. Optionally set FP16 flag so layers without Q/DQ fall back to FP16.
        6. Set DETAILED profiling verbosity for per-layer timing.
        7. Parse the ONNX file with OnnxParser; raise on parse errors.
        8. Build and serialise engine to disk (optional JSON metadata prefix).

    Args:
        onnx_file: Path to the quantized ONNX model containing QDQ nodes.
        engine_file: Destination path.  Defaults to onnx_file with .engine
            suffix.
        workspace_gb: GPU memory for TRT optimisation workspace (default 4 GB).
        fp16_fallback: Allow FP16 for layers without INT8 Q/DQ coverage.
        verbose: Enable TRT VERBOSE logging (very chatty).
        metadata: Optional dict serialised to JSON and prepended to the engine
            file (same 4-byte-length-prefix format as ultralytics onnx2engine).
            Pass None (default) to write a plain engine file readable with a
            simple open().read() → deserialize_cuda_engine() call.

    Returns:
        Absolute path of the written engine file (str).

    Raises:
        RuntimeError: If ONNX parsing or engine build fails.
    """
    import tensorrt as trt

    onnx_path = Path(onnx_file)
    engine_path = Path(engine_file) if engine_file else onnx_path.with_suffix(".engine")
    is_trt10 = int(trt.__version__.split(".", 1)[0]) >= 10

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    # Workspace
    workspace_bytes = int(workspace_gb * (1 << 30))
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes  # type: ignore[attr-defined]

    # Network with EXPLICIT_BATCH — required for ONNX models and QDQ fusion
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    # INT8 flag — with Q/DQ nodes in the ONNX, TRT enters explicit
    # quantization mode and reads the embedded scales. No calibrator needed.
    config.set_flag(trt.BuilderFlag.INT8)
    if fp16_fallback and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Detailed profiling so per-layer timing is available via trtexec --profilingVerbosity
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # Parse ONNX
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(
            f"Failed to parse ONNX file: {onnx_path}\n" + "\n".join(errors)
        )

    for i in range(network.num_inputs):
        inp = network.get_input(i)
        print(f"  TRT input  [{i}]: {inp.name}  {tuple(inp.shape)}  {inp.dtype}")
    for i in range(network.num_outputs):
        out = network.get_output(i)
        print(f"  TRT output [{i}]: {out.name}  {tuple(out.shape)}  {out.dtype}")

    prec = "INT8" + (" + FP16 fallback" if fp16_fallback else " + FP32 fallback")
    print(f"Building TRT engine ({prec}, explicit quantization) ...")

    build_fn = builder.build_serialized_network if is_trt10 else builder.build_engine  # type: ignore[attr-defined]
    with build_fn(network, config) as engine:
        if engine is None:
            raise RuntimeError("TRT engine build failed — check log output above.")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            if metadata is not None:
                meta = json.dumps(metadata)
                f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                f.write(meta.encode())
            f.write(engine if is_trt10 else engine.serialize())

    print(f"TRT engine saved: {engine_path}")
    return str(engine_path)

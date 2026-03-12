"""
TensorRT engine export utilities.

Single entry point: ``onnx2engine()``.

Precision modes
---------------
"float32"
    Plain FP32 engine.  No precision flags set beyond EXPLICIT_BATCH.

"float16"
    FP16 engine.  Sets BuilderFlag.FP16.  TRT picks FP16 kernels where
    available and falls back to FP32 for ops that have no FP16 path.

"int8"
    Explicit INT8 engine.  The ONNX *must* already contain
    QuantizeLinear / DequantizeLinear (QDQ) nodes produced by
    ``wrap_nodes_in_int8_qdq()``.  TRT reads the embedded per-tensor
    scales directly ("explicit quantization" mode) — no calibration
    dataset is required.  Setting BuilderFlag.INT8 tells TRT that INT8
    kernels are allowed; setting BuilderFlag.FP16 additionally lets
    layers without Q/DQ coverage fall back to FP16 instead of FP32.

Compared with ultralytics onnx2engine
--------------------------------------
* INT8 path: uses embedded Q/DQ scales instead of an IInt8Calibrator.
* Always writes a plain engine file (no JSON metadata prefix) so the
  file can be deserialised with a plain ``file.read()`` call.
* Detailed profiling verbosity enabled for per-layer TRT timing via
  trtexec --profilingVerbosity.
"""

from __future__ import annotations

from pathlib import Path


def onnx2engine(
    onnx_file: str,
    engine_file: str | None = None,
    workspace_gb: float = 4.0,
    precision: str = "float32",
    fp16_fallback: bool = True,
    verbose: bool = False,
    sparse_weights: bool = False,
) -> str:
    """Build a TensorRT engine from an ONNX file.

    Steps (mirrors ultralytics onnx2engine structure):
        1. Create TRT Logger / Builder / BuilderConfig.
        2. Set workspace memory pool limit.
        3. Create network with EXPLICIT_BATCH flag (required for ONNX).
        4. Set precision flags:
             float32 : no extra flags.
             float16 : BuilderFlag.FP16.
             int8    : BuilderFlag.INT8 (reads Q/DQ scales from ONNX —
                       explicit quantization, no calibrator).  Also sets
                       BuilderFlag.FP16 when fp16_fallback=True so layers
                       without Q/DQ coverage fall back to FP16.
        5. Enable DETAILED profiling verbosity.
        6. Parse ONNX; raise on parse errors.
        7. Build and serialise engine to disk.

    Args:
        onnx_file: Path to the ONNX model.
        engine_file: Destination path.  Defaults to onnx_file with .engine
            suffix.
        workspace_gb: GPU memory for TRT optimisation workspace (default 4 GB).
        precision: One of "float32", "float16", "int8".
        fp16_fallback: For precision="int8", allow FP16 for layers without
            INT8 Q/DQ coverage.  Ignored for other precisions.
        verbose: Enable TRT VERBOSE logging.
        sparse_weights: Set BuilderFlag.SPARSE_WEIGHTS so TRT selects sparse
            Tensor Core kernels for layers whose weight initializers satisfy
            the 2:4 structured-sparsity pattern.  Layers that do not satisfy
            the pattern fall back to dense kernels silently.  Requires Ampere
            or later hardware for any speedup; on Turing the engine builds
            successfully but no sparse kernels are selected.

    Returns:
        Absolute path of the written engine file.

    Raises:
        ValueError: For unknown precision.
        RuntimeError: If ONNX parsing or engine build fails.
    """
    import tensorrt as trt

    if precision not in ("float32", "float16", "int8"):
        raise ValueError(f"Unknown precision '{precision}'. Use 'float32', 'float16', or 'int8'.")

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

    # Network — EXPLICIT_BATCH required for ONNX models
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)

    # Precision flags
    if precision == "float16":
        if not builder.platform_has_fast_fp16:
            print("  WARNING: platform_has_fast_fp16 is False; FP16 engine may be slow.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        # INT8 flag + Q/DQ nodes in the ONNX → TRT enters explicit-quantization
        # mode.  No IInt8Calibrator needed; embedded scales are used directly.
        config.set_flag(trt.BuilderFlag.INT8)
        if fp16_fallback and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

    # 2:4 structured sparsity — TRT selects sparse Tensor Core kernels for
    # Conv/Linear layers whose weights satisfy the 2:4 pattern.  No-op on
    # Turing (CC 7.5); effective on Ampere (CC 8.0+) and later.
    if sparse_weights:
        if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS"):
            config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
            print("  Sparse weights enabled (BuilderFlag.SPARSE_WEIGHTS).")
        else:
            print("  WARNING: BuilderFlag.SPARSE_WEIGHTS not available in this "
                  "TRT version — sparse flag ignored.")

    # Detailed per-layer profiling (accessible via trtexec --profilingVerbosity)
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

    prec_label = {
        "float32": "FP32",
        "float16": "FP16",
        "int8": f"INT8 explicit + {'FP16' if fp16_fallback else 'FP32'} fallback",
    }[precision]
    print(f"Building TRT engine ({prec_label}) ...")

    build_fn = builder.build_serialized_network if is_trt10 else builder.build_engine  # type: ignore[attr-defined]
    with build_fn(network, config) as engine:
        if engine is None:
            raise RuntimeError("TRT engine build failed — check log output above.")
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(engine if is_trt10 else engine.serialize())

    print(f"TRT engine saved: {engine_path}")
    return str(engine_path)


def onnx2engine_explicit(
    onnx_file: str,
    engine_file: str | None = None,
    workspace_gb: float = 4.0,
    fp16_fallback: bool = True,
    verbose: bool = False,
    metadata: dict | None = None,
) -> str:
    """Build a TRT engine from an ONNX model with embedded QDQ nodes.

    Convenience wrapper around ``onnx2engine(precision='int8')``.
    The ``metadata`` parameter is accepted for backward compatibility but
    ignored — the engine is always written without a metadata prefix.
    """
    return onnx2engine(
        onnx_file=onnx_file,
        engine_file=engine_file,
        workspace_gb=workspace_gb,
        precision="int8",
        fp16_fallback=fp16_fallback,
        verbose=verbose,
    )

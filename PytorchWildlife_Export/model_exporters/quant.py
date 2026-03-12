"""
Quantization / mixed-precision utilities for ONNX graphs.
"""
import logging

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import onnx.shape_inference
from onnx import TensorProto

LOGGER = logging.getLogger(__name__)

FLOAT = TensorProto.FLOAT
FLOAT16 = TensorProto.FLOAT16
INT8 = TensorProto.INT8


def _find_silu_muls(model: onnx.ModelProto) -> dict:
    """
    Detect all SiLU activation Mul nodes in the graph.

    SiLU(x) = x * sigmoid(x) is represented in ONNX as:
        Sigmoid(x) → sig_out
        Mul(x, sig_out) or Mul(sig_out, x) → silu_out

    Returns:
        {mul_node_name: x_tensor_name}  — maps each SiLU Mul to the x tensor
        (the non-Sigmoid input that should be calibrated and quantized).
    """
    output_producer = {out: n for n in model.graph.node for out in n.output}
    result = {}
    for node in model.graph.node:
        if node.op_type != "Mul" or len(node.input) != 2:
            continue
        a, b = node.input[0], node.input[1]
        prod_a = output_producer.get(a)
        prod_b = output_producer.get(b)
        # Pattern: Mul(x, Sigmoid(x)) or Mul(Sigmoid(x), x)
        if prod_b and prod_b.op_type == "Sigmoid" and prod_b.input[0] == a:
            result[node.name] = a  # x is input a
        elif prod_a and prod_a.op_type == "Sigmoid" and prod_a.input[0] == b:
            result[node.name] = b  # x is input b
    return result


def _get_node_input_tensors_for_calibration(
    node, silu_map: dict, init_names: set
) -> list:
    """
    Returns the ordered list of input tensor names that need calibration for a node.

    Initializers (weights / biases) are skipped — they are not runtime activations.
    For SiLU Mul nodes the sigmoid(x) input is skipped because it lives in (0, 1)
    and must stay float32.
    """
    if node.op_type == "Conv":
        return [node.input[0]]  # X only; W and B are initializers
    elif node.op_type == "MatMul":
        return [inp for inp in node.input if inp not in init_names]
    elif node.op_type == "Mul" and node.name in silu_map:
        return [silu_map[node.name]]  # x only, not sigmoid(x)
    elif node.op_type in ("Add", "Concat", "MaxPool"):
        return [inp for inp in node.input if inp not in init_names]
    else:
        return [node.input[0]]


def _make_scalar_init(name: str, value, dtype) -> onnx.TensorProto:
    """Create a scalar (shape=[]) initializer."""
    arr = np.array(value, dtype=dtype)
    return onh.from_array(arr, name=name)


def _qdq_pair(
    tensor_name: str,
    scale_name: str,
    zp_name: str,
    new_nodes: list,
    node_prefix: str = "",
) -> str:
    """
    Insert QuantizeLinear → DequantizeLinear for a runtime (activation) tensor.
    Returns the name of the DequantizeLinear output (float32).

    ``node_prefix`` should be set to the wrapping node's name so that when
    multiple Conv nodes share the same activation input tensor, each produces
    uniquely named Q/DQ intermediate tensors (ONNX SSA requirement).
    """
    scope = (node_prefix + "__") if node_prefix else ""
    q_out = scope + tensor_name + "__q_int8"
    dq_out = scope + tensor_name + "__dq_fp32"
    new_nodes.append(oh.make_node(
        "QuantizeLinear",
        inputs=[tensor_name, scale_name, zp_name],
        outputs=[q_out],
        name=scope + tensor_name + "__QuantizeLinear",
    ))
    new_nodes.append(oh.make_node(
        "DequantizeLinear",
        inputs=[q_out, scale_name, zp_name],
        outputs=[dq_out],
        name=scope + tensor_name + "__DequantizeLinear",
    ))
    return dq_out


def _apply_fp16_wraps(
    model: onnx.ModelProto, target_names: set
) -> onnx.ModelProto:
    """
    Single-pass graph rewrite that wraps all nodes whose ``name`` is in
    ``target_names`` in fp16 Cast nodes.  The surrounding graph stays float32.

    This is the shared implementation used by both ``wrap_node_in_fp16`` and
    ``wrap_conv_nodes_in_fp16``.
    """
    graph = model.graph
    init_map = {init.name: init for init in graph.initializer}
    # Track fp16 initializers we've already created to avoid duplicates when
    # the same weight is shared by multiple target nodes.
    created_fp16_inits: set = set()

    new_nodes = []
    new_inits = list(graph.initializer)

    for node in graph.node:
        if node.name not in target_names:
            new_nodes.append(node)
            continue

        LOGGER.info(f"Wrapping node '{node.name}' ({node.op_type}) in fp16 casts.")

        # --- cast inputs float32 → float16 ---
        new_inputs = []
        for inp in node.input:
            if inp == "":
                new_inputs.append(inp)
                continue

            if inp in init_map:
                fp16_init_name = inp + "__fp16"
                if fp16_init_name not in created_fp16_inits:
                    arr = onh.to_array(init_map[inp]).astype(np.float16)
                    new_inits.append(onh.from_array(arr, name=fp16_init_name))
                    created_fp16_inits.add(fp16_init_name)
                    LOGGER.debug(f"  Converted initializer '{inp}' → '{fp16_init_name}'")
                new_inputs.append(fp16_init_name)
            else:
                cast_out = inp + "__cast_fp16"
                cast_node = oh.make_node(
                    "Cast",
                    inputs=[inp],
                    outputs=[cast_out],
                    name=inp + "__cast_to_fp16",
                    to=FLOAT16,
                )
                new_nodes.append(cast_node)
                new_inputs.append(cast_out)
                LOGGER.debug(f"  Inserted Cast fp32→fp16 for input '{inp}'")

        # --- rewired node operating in fp16 ---
        fp16_outputs = [o + "__fp16_out" if o else o for o in node.output]
        new_node = oh.make_node(
            node.op_type,
            inputs=new_inputs,
            outputs=fp16_outputs,
            name=node.name,
        )
        new_node.attribute.extend(node.attribute)
        new_nodes.append(new_node)

        # --- cast outputs float16 → float32, restoring original names ---
        for orig_out, fp16_out in zip(node.output, fp16_outputs):
            if orig_out == "":
                continue
            cast_node = oh.make_node(
                "Cast",
                inputs=[fp16_out],
                outputs=[orig_out],
                name=orig_out + "__cast_to_fp32",
                to=FLOAT,
            )
            new_nodes.append(cast_node)
            LOGGER.debug(f"  Inserted Cast fp16→fp32 for output '{orig_out}'")

    new_graph = oh.make_graph(
        new_nodes,
        graph.name,
        list(graph.input),
        list(graph.output),
        new_inits,
    )
    new_model = oh.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    new_model = onnx.shape_inference.infer_shapes(new_model, data_prop=True)
    onnx.checker.check_model(new_model)
    return new_model


def wrap_node_in_fp16(model: onnx.ModelProto, target_node_name: str) -> onnx.ModelProto:
    """
    Wraps a single ONNX node (identified by its ``name`` attribute) in float16
    Cast nodes so that the node itself runs in fp16 while the rest of the graph
    remains in float32.

    Args:
        model: A loaded ONNX ModelProto (float32 baseline).
        target_node_name: The ``name`` field of the node to wrap.

    Returns:
        A new ModelProto with the targeted node wrapped in fp16 casts.

    Raises:
        ValueError: If ``target_node_name`` is not found in the graph.
    """
    if not any(n.name == target_node_name for n in model.graph.node):
        raise ValueError(
            f"Node '{target_node_name}' not found in graph. "
            f"Available names: {[n.name for n in model.graph.node if n.name]}"
        )
    result = _apply_fp16_wraps(model, {target_node_name})
    LOGGER.info(f"Node '{target_node_name}' successfully wrapped in fp16 casts.")
    return result


def wrap_conv_nodes_in_fp16(
    model: onnx.ModelProto, n: int = -1
) -> onnx.ModelProto:
    """
    Wraps the first ``n`` Conv nodes in the graph in float16 Cast nodes.
    Pass ``n=-1`` to wrap every Conv node.

    Prints the total number of Conv nodes in the model before wrapping.

    Args:
        model: A loaded ONNX ModelProto (float32 baseline).
        n: Number of Conv nodes to wrap, in graph order.  -1 means all.

    Returns:
        A new ModelProto with the selected Conv nodes wrapped in fp16 casts.
    """
    all_convs = [node for node in model.graph.node if node.op_type == "Conv"]
    total = len(all_convs)
    print(f"Total Conv nodes in model: {total}")

    targets = all_convs if n == -1 else all_convs[:n]
    target_names = {node.name for node in targets}
    print(f"Wrapping {len(targets)} Conv node(s) in fp16.")

    result = _apply_fp16_wraps(model, target_names)
    LOGGER.info(f"Wrapped {len(targets)}/{total} Conv nodes in fp16 casts.")
    return result


def calibrate_node_scales(
    model: onnx.ModelProto,
    target_node_name: str,
    calibration_loader,
) -> tuple:
    """
    Computes symmetric per-tensor INT8 activation scales for the input and
    output of a single Conv node by running calibration images through the
    base float32 model.

    The model is temporarily modified to expose the Conv's runtime activation
    input and its output as extra graph outputs so ORT can return those tensors.
    Calibration images from ``calibration_loader`` are expected to be uint8
    NCHW tensors (the default TRTCalibrationDataLoader format); they are
    converted to float32 [0, 1] before being fed to the model.

    The scale is computed as ``max(abs(tensor)) / 127`` taken over all
    calibration batches (symmetric INT8, zero_point=0).

    Args:
        model: The base float32 ONNX ModelProto.
        target_node_name: Name of the Conv node to calibrate.
        calibration_loader: A TRTCalibrationDataLoader instance.

    Returns:
        (activation_scale, output_scale) as floats.
    """
    import os
    import tempfile

    import onnxruntime as ort

    node_map = {n.name: n for n in model.graph.node}
    if target_node_name not in node_map:
        raise ValueError(f"Node '{target_node_name}' not found.")
    node = node_map[target_node_name]

    # index 0 = activation input, index 1 = weight (initializer, skip)
    act_input_name = node.input[0]
    act_output_name = node.output[0]

    # Build a temporary model that adds those two tensors to graph outputs
    existing_out_names = {o.name for o in model.graph.output}
    extra_outputs = []
    for name in (act_input_name, act_output_name):
        if name not in existing_out_names:
            extra_outputs.append(
                oh.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )

    calib_graph = oh.make_graph(
        list(model.graph.node),
        model.graph.name,
        list(model.graph.input),
        list(model.graph.output) + extra_outputs,
        list(model.graph.initializer),
    )
    calib_model = oh.make_model(calib_graph, opset_imports=model.opset_import)
    calib_model.ir_version = model.ir_version

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        onnx.save(calib_model, tmp_path)
        session = ort.InferenceSession(
            tmp_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    finally:
        os.unlink(tmp_path)

    ort_input_name = session.get_inputs()[0].name
    num_images = calibration_loader.num_images
    print(f"Calibrating '{target_node_name}' over {num_images} images...")

    act_input_max = 0.0
    act_output_max = 0.0
    n = 0

    for batch in calibration_loader:
        # TRTCalibrationDataLoader default: (1, 3, H, W) uint8
        img_np = batch["img"].numpy().astype(np.float32) / 255.0
        act_in, act_out = session.run(
            [act_input_name, act_output_name], {ort_input_name: img_np}
        )
        act_input_max = max(act_input_max, float(np.max(np.abs(act_in))))
        act_output_max = max(act_output_max, float(np.max(np.abs(act_out))))
        n += 1
        if n % 50 == 0 or n == num_images:
            print(f"  [{n}/{num_images}]  in_max={act_input_max:.4f}  out_max={act_output_max:.4f}")

    activation_scale = act_input_max / 127.0
    output_scale = act_output_max / 127.0
    print(
        f"Calibrated scales — activation: {activation_scale:.6f}, "
        f"output: {output_scale:.6f}"
    )
    return activation_scale, output_scale


def wrap_node_in_int8_qdq(
    model: onnx.ModelProto,
    target_node_name: str,
    input_scales: list = None,
    output_scale: float = 1.0 / 127.0,
) -> onnx.ModelProto:
    """
    Wraps a single node in INT8 QuantizeLinear / DequantizeLinear (QDQ) pairs.

    Supported op types: Conv, MatMul, Mul (SiLU), Add, Concat, MaxPool.

    ``input_scales`` is an ordered list of calibrated scales, one per dynamic
    activation input (initializers and the sigmoid(x) branch of SiLU are not
    counted).  Defaults to ``[1/127]`` when not supplied.

    Weight scales for Conv are computed from the stored initializer values
    using symmetric per-tensor quantisation: scale = max(abs(W)) / 127.
    Conv bias is kept as the original float32 initializer and passed directly
    into the Conv node (no Q/DQ wrapper, no separate Add).  TRT's ONNX parser
    rejects INT32 bias; float32 is valid because X_dq and W_dq are both float32
    DQ outputs.  TRT fuses the bias into the INT8 Conv epilogue alongside the
    output DQ and any following activation (e.g. SiLU).

    Args:
        model: A loaded float32 ONNX ModelProto.
        target_node_name: The ``name`` field of the node to quantise.
        input_scales: Calibrated activation scales (one per dynamic input).
        output_scale: Scale for the node output tensor.

    Returns:
        A new ModelProto with QDQ nodes inserted around the target node.

    Raises:
        ValueError: If the node is not found or its op type is unsupported.
    """
    if input_scales is None:
        input_scales = [1.0 / 127.0]

    graph = model.graph

    node_map = {n.name: n for n in graph.node}
    if target_node_name not in node_map:
        raise ValueError(
            f"Node '{target_node_name}' not found in graph."
        )
    target = node_map[target_node_name]
    _SUPPORTED = ("Conv", "MatMul", "Mul", "Add", "Concat", "MaxPool")
    if target.op_type not in _SUPPORTED:
        raise ValueError(
            f"Node '{target_node_name}' is a {target.op_type}, "
            f"which is not in the supported set {_SUPPORTED}."
        )

    init_map = {init.name: init for init in graph.initializer}

    new_nodes = []
    new_inits = list(graph.initializer)

    # ── shared zero-point scalar (int8, value=0) ──────────────────────────
    zp_name = target_node_name + "__zp_int8"
    new_inits.append(_make_scalar_init(zp_name, 0, np.int8))

    # ── output scale ──────────────────────────────────────────────────────
    out_scale_name = target_node_name + "__out_scale"
    new_inits.append(_make_scalar_init(out_scale_name, np.float32(output_scale), np.float32))

    for node in graph.node:
        if node.name != target_node_name:
            new_nodes.append(node)
            continue

        raw_out = node.output[0] + "__raw"

        if node.op_type == "Conv":
            # Conv inputs: [X, W, B?]
            inp_x = node.input[0]   # activation
            inp_w = node.input[1]   # weight initializer
            inp_b = node.input[2] if len(node.input) > 2 else ""  # bias (optional)

            # ── quantise activation ───────────────────────────────────────────
            act_scale_name = target_node_name + "__act_scale"
            new_inits.append(_make_scalar_init(act_scale_name, np.float32(input_scales[0]), np.float32))
            inp_x_dq = _qdq_pair(inp_x, act_scale_name, zp_name, new_nodes,
                                  node_prefix=target_node_name)
            LOGGER.debug(f"  Activation Q/DQ inserted for '{inp_x}'")

            # ── quantise weight: compute scale, store INT8 init, add DQ only ──
            weight_arr = onh.to_array(init_map[inp_w]).astype(np.float32)
            w_max = np.max(np.abs(weight_arr))
            w_scale = float(w_max / 127.0) if w_max > 0 else 1.0 / 127.0
            w_int8 = np.clip(np.round(weight_arr / w_scale), -128, 127).astype(np.int8)

            w_int8_name = inp_w + "__int8"
            w_scale_name = inp_w + "__scale"
            new_inits.append(onh.from_array(w_int8, name=w_int8_name))
            new_inits.append(_make_scalar_init(w_scale_name, np.float32(w_scale), np.float32))

            w_dq_out = inp_w + "__dq_fp32"
            new_nodes.append(oh.make_node(
                "DequantizeLinear",
                inputs=[w_int8_name, w_scale_name, zp_name],
                outputs=[w_dq_out],
                name=inp_w + "__DequantizeLinear",
            ))
            LOGGER.debug(f"  Weight pre-quantised to INT8, scale={w_scale:.6f}")

            # ── bias: keep as original float32 initializer ────────────────────
            # TRT's ONNX parser rejects INT32 bias at parse time.  Passing the
            # original float32 bias directly is valid (X_dq and W_dq are both
            # float32 DQ outputs, so this is a well-typed ONNX Conv).  TRT fuses
            # it into the INT8 Conv epilogue: INT32 accum → DQ → +bias → SiLU,
            # all in one kernel — no separate Add node to break fusion.
            new_op_inputs = [inp_x_dq, w_dq_out]
            if inp_b:
                new_op_inputs.append(inp_b)
                LOGGER.debug(f"  Bias '{inp_b}' kept as float32 inside Conv")

        elif node.op_type == "MatMul":
            # MatMul inputs: [A, B] — both are dynamic activations in the attention block
            inp_a = node.input[0]
            inp_b = node.input[1]

            act_scale_a = target_node_name + "__act_scale_A"
            act_scale_b = target_node_name + "__act_scale_B"
            new_inits.append(_make_scalar_init(act_scale_a, np.float32(input_scales[0]), np.float32))
            new_inits.append(_make_scalar_init(act_scale_b, np.float32(
                input_scales[1] if len(input_scales) > 1 else input_scales[0]), np.float32))

            inp_a_dq = _qdq_pair(inp_a, act_scale_a, zp_name, new_nodes,
                                  node_prefix=target_node_name + "__A")
            inp_b_dq = _qdq_pair(inp_b, act_scale_b, zp_name, new_nodes,
                                  node_prefix=target_node_name + "__B")
            LOGGER.debug(f"  MatMul activation Q/DQ inserted for '{inp_a}' and '{inp_b}'")

            new_op_inputs = [inp_a_dq, inp_b_dq]

        elif node.op_type == "Mul":
            # SiLU: x * sigmoid(x).
            # Only quantize x — sigmoid(x) is in (0, 1) and must stay float32.
            act_scale_name = target_node_name + "__act_scale"
            new_inits.append(_make_scalar_init(act_scale_name, np.float32(input_scales[0]), np.float32))
            output_producer = {out: n for n in graph.node for out in n.output}
            new_op_inputs = []
            for inp in node.input:
                prod = output_producer.get(inp)
                if prod and prod.op_type == "Sigmoid":
                    new_op_inputs.append(inp)   # sigmoid(x) — pass through unchanged
                else:
                    dq = _qdq_pair(inp, act_scale_name, zp_name, new_nodes,
                                   node_prefix=target_node_name)
                    new_op_inputs.append(dq)    # x — quantized
            LOGGER.debug(f"  SiLU Mul: x quantized, sigmoid(x) kept float32")

        elif node.op_type in ("Add", "Concat"):
            # All inputs are dynamic activations; each gets its own scale.
            new_op_inputs = []
            for i, inp in enumerate(node.input):
                scale_i = input_scales[i] if i < len(input_scales) else input_scales[-1]
                in_scale_name = f"{target_node_name}__in{i}_scale"
                new_inits.append(_make_scalar_init(in_scale_name, np.float32(scale_i), np.float32))
                dq = _qdq_pair(inp, in_scale_name, zp_name, new_nodes,
                               node_prefix=f"{target_node_name}__in{i}")
                new_op_inputs.append(dq)
            LOGGER.debug(f"  {node.op_type}: {len(node.input)} input(s) quantized")

        else:  # MaxPool
            act_scale_name = target_node_name + "__act_scale"
            new_inits.append(_make_scalar_init(act_scale_name, np.float32(input_scales[0]), np.float32))
            inp_dq = _qdq_pair(node.input[0], act_scale_name, zp_name, new_nodes,
                               node_prefix=target_node_name)
            new_op_inputs = [inp_dq]
            LOGGER.debug(f"  MaxPool input Q/DQ inserted")

        # ── op node (unchanged op, rewired inputs) ─────────────────────────
        new_op = oh.make_node(
            node.op_type,
            inputs=new_op_inputs,
            outputs=[raw_out],
            name=node.name,
        )
        new_op.attribute.extend(node.attribute)
        new_nodes.append(new_op)

        # ── quantise output: float → Q → DQ → float ───────────────────────
        q_out = raw_out + "__q_int8"
        new_nodes.append(oh.make_node(
            "QuantizeLinear",
            inputs=[raw_out, out_scale_name, zp_name],
            outputs=[q_out],
            name=node.output[0] + "__out_QuantizeLinear",
        ))
        new_nodes.append(oh.make_node(
            "DequantizeLinear",
            inputs=[q_out, out_scale_name, zp_name],
            outputs=[node.output[0]],
            name=node.output[0] + "__out_DequantizeLinear",
        ))
        LOGGER.info(f"  INT8 QDQ pattern inserted around '{node.name}'")

    new_graph = oh.make_graph(
        new_nodes,
        graph.name,
        list(graph.input),
        list(graph.output),
        new_inits,
    )
    new_model = oh.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version
    new_model = onnx.shape_inference.infer_shapes(new_model, data_prop=True)
    onnx.checker.check_model(new_model)
    LOGGER.info(f"Node '{target_node_name}' successfully wrapped in INT8 QDQ.")
    return new_model


def calibrate_conv_nodes_scales(
    model: onnx.ModelProto,
    target_node_names: list,
    calibration_loader,
    silu_map: dict = None,
) -> dict:
    """
    Single-pass calibration for multiple nodes (Conv, MatMul, SiLU Mul,
    Add, Concat, MaxPool, …).

    Exposes every target node's dynamic activation inputs and output as extra
    graph outputs, runs all calibration images through ORT once, and returns a
    dict mapping node name → ([input_scale, ...], output_scale).

    The number of input scales equals the number of dynamic activation inputs
    for that node (determined by ``_get_node_input_tensors_for_calibration``):
    Conv and MaxPool → 1, MatMul and Add → 2, Concat → N inputs.

    Args:
        model: The base float32 ONNX ModelProto.
        target_node_names: Ordered list of node names to calibrate.
        calibration_loader: A TRTCalibrationDataLoader instance.
        silu_map: Optional dict ``{mul_node_name: x_tensor_name}`` from
            ``_find_silu_muls``.  Used to identify the correct x input for
            SiLU Mul nodes.  Computed internally if not supplied.

    Returns:
        {node_name: ([input_scale_0, ...], output_scale)}
    """
    import os
    import tempfile

    import onnxruntime as ort

    if silu_map is None:
        silu_map = _find_silu_muls(model)

    init_names = {init.name for init in model.graph.initializer}
    node_map = {n.name: n for n in model.graph.node}
    existing_out_names = {o.name for o in model.graph.output}

    # observe: node_name → (list_of_input_tensor_names, output_tensor_name)
    observe: dict = {}
    extra_outputs = []
    seen_tensors: set = set()
    for name in target_node_names:
        node = node_map[name]
        input_tensors = _get_node_input_tensors_for_calibration(node, silu_map, init_names)
        act_out = node.output[0]
        observe[name] = (input_tensors, act_out)
        for t in input_tensors + [act_out]:
            if t not in existing_out_names and t not in seen_tensors:
                extra_outputs.append(
                    oh.make_tensor_value_info(t, TensorProto.FLOAT, None)
                )
                seen_tensors.add(t)

    calib_graph = oh.make_graph(
        list(model.graph.node),
        model.graph.name,
        list(model.graph.input),
        list(model.graph.output) + extra_outputs,
        list(model.graph.initializer),
    )
    calib_model = oh.make_model(calib_graph, opset_imports=model.opset_import)
    calib_model.ir_version = model.ir_version

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp_path = f.name
    try:
        onnx.save(calib_model, tmp_path)
        session = ort.InferenceSession(
            tmp_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    finally:
        os.unlink(tmp_path)

    ort_input_name = session.get_inputs()[0].name
    fetch_names = list(seen_tensors)
    num_images = calibration_loader.num_images
    print(
        f"Calibrating {len(target_node_names)} nodes "
        f"over {num_images} images (single pass)..."
    )

    running_max: dict = {t: 0.0 for t in seen_tensors}
    n = 0
    for batch in calibration_loader:
        img_np = batch["img"].numpy().astype(np.float32) / 255.0
        results = session.run(fetch_names, {ort_input_name: img_np})
        for tensor_name, arr in zip(fetch_names, results):
            running_max[tensor_name] = max(
                running_max[tensor_name], float(np.max(np.abs(arr)))
            )
        n += 1
        if n % 50 == 0 or n == num_images:
            print(f"  [{n}/{num_images}] calibration images processed")

    scales = {}
    for node_name, (input_tensors, act_out) in observe.items():
        in_scales = [running_max[t] / 127.0 for t in input_tensors]
        out_scale = running_max[act_out] / 127.0
        scales[node_name] = (in_scales, out_scale)
        LOGGER.debug(
            f"  {node_name}: in_scales={[f'{s:.6f}' for s in in_scales]}, out={out_scale:.6f}"
        )

    print(f"Calibration complete for {len(scales)} nodes.")
    return scales


def wrap_nodes_in_int8_qdq(
    model: onnx.ModelProto,
    calibration_loader,
    node_types: list = None,
    exclude: list = None,
    max_index: dict = None,
    node_names: list = None,
) -> onnx.ModelProto:
    """
    Calibrates and wraps nodes in INT8 QDQ pairs.

    Args:
        model: A loaded float32 ONNX ModelProto.
        calibration_loader: A TRTCalibrationDataLoader instance.
        node_types: Op types to quantize (e.g. ``['Conv']``).
            Defaults to ``['Conv']`` when neither node_types nor node_names
            is provided. Pass an empty list ``[]`` to suppress type-based
            selection entirely and rely solely on node_names.
        exclude: List of substrings or exact node names. Any node whose name
            contains at least one of these strings is skipped during
            type-based selection. Does not affect node_names entries.
            Defaults to no exclusions.
        max_index: Per-op-type topological index cap (e.g. ``{'Conv': 60}``
            to consider only the first 60 Conv nodes in graph order before
            applying the exclude filter).  Op types absent from the dict are
            uncapped.  Defaults to no cap.
        node_names: Explicit list of node names to quantize, regardless of
            op type, exclude list, or max_index. Added after type-based
            selection; duplicates are silently dropped. Use this to target
            specific high-cost nodes by name (e.g. from profile_analysis
            output) without enabling bulk type-based quantization.

    Returns:
        A new ModelProto with the selected nodes wrapped in INT8 QDQ.
    """
    if node_names is None:
        node_names = []
    if exclude is None:
        exclude = []
    if max_index is None:
        max_index = {}
    # Only default to Conv when neither selector is specified
    if node_types is None:
        node_types = [] if node_names else ["Conv"]

    # Pre-build SiLU map once (used for "SiLU" pseudo-type and for calibration overrides)
    silu_map = _find_silu_muls(model)  # {mul_node_name: x_tensor_name}

    targets = []
    for op_type in node_types:
        # "SiLU" is a pseudo-type: match Mul nodes that implement SiLU(x) = x * sigmoid(x)
        if op_type == "SiLU":
            all_of_type = [n for n in model.graph.node
                           if n.op_type == "Mul" and n.name in silu_map]
            label = "SiLU(Mul)"
        else:
            all_of_type = [n for n in model.graph.node if n.op_type == op_type]
            label = op_type

        cap = max_index.get(op_type, len(all_of_type))
        capped = all_of_type[:cap]
        cap_excluded = all_of_type[cap:]
        excluded = [n for n in capped if any(ex in n.name for ex in exclude)]
        selected = [n for n in capped if n not in excluded]
        print(
            f"  {label}: {len(all_of_type)} total, "
            f"capped at {cap}, "
            f"excluded {len(excluded)}, "
            f"quantizing {len(selected)}"
        )
        if cap_excluded:
            print(f"    Excluded by cap ({len(cap_excluded)}):")
            for n in cap_excluded:
                print(f"      {n.name}")
        targets.extend(selected)

    # Resolve explicitly named nodes and append (deduplicating against type-selected set)
    if node_names:
        node_map = {n.name: n for n in model.graph.node}
        already = {n.name for n in targets}
        named_targets = []
        for name in node_names:
            if name in already:
                print(f"  [node_names] '{name}' already selected via node_types, skipping duplicate")
                continue
            if name not in node_map:
                raise ValueError(f"node_names: node '{name}' not found in graph")
            named_targets.append(node_map[name])
        print(f"  Explicit node_names: {len(named_targets)} node(s)")
        for n in named_targets:
            print(f"    {n.op_type} '{n.name}'")
        targets.extend(named_targets)

    target_names = [n.name for n in targets]
    print(f"Total nodes to quantize: {len(target_names)}")

    scales = calibrate_conv_nodes_scales(
        model, target_names, calibration_loader, silu_map=silu_map
    )

    # Build op-type map from the original model so we can apply Concat-specific
    # shared-scale logic before any nodes have been rewritten.
    node_op_map = {n.name: n.op_type for n in model.graph.node}

    result = model
    for node_name in target_names:
        in_scales, out_scale = scales[node_name]

        if node_op_map.get(node_name) == "Concat":
            # Shared-scale INT8 Concat:
            #
            # With per-input scales (heterogeneous), TRT cannot merge all inputs
            # into one INT8 layout and falls back to: DQ each input to FP16 →
            # Concat in FP16 → clone Q per downstream consumer.  That creates a
            # separate Q kernel for every consumer of the Concat output.
            #
            # A single shared scale avoids the heterogeneity.  The pattern
            # Q(s) → Concat → DQ(s) is a first-class INT8 Concat in TRT's
            # explicit-quantization mode: all inputs arrive as INT8 at the same
            # scale, Concat runs INT8, the output DQ is a single shared buffer.
            # Inputs whose preceding DQ had a different scale undergo a DQ→Q
            # requantise, which TRT may fuse into the preceding Conv epilogue.
            s_shared = max(max(in_scales, default=0.0), out_scale)
            LOGGER.info(
                f"  Concat '{node_name}': using shared scale {s_shared:.6f} "
                f"(inputs max={max(in_scales, default=0.0):.6f}, "
                f"output={out_scale:.6f})"
            )
            in_scales = [s_shared] * len(in_scales)
            out_scale = s_shared

        result = wrap_node_in_int8_qdq(
            result, node_name,
            input_scales=in_scales,
            output_scale=out_scale,
        )

    LOGGER.info(f"Wrapped {len(targets)} nodes in INT8 QDQ.")
    return result

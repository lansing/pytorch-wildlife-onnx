"""
naming.py
---------
Canonical output filename builder for exported models.

Kept as a standalone module (no heavy deps) so it can be imported by both
export_tool.py and sweep_export.py without pulling in torch/onnx/tensorrt.
"""


def build_output_filename(
    model_version: str,
    format: str,
    input_img_size: int,
    model_type: str = "yolov10",
    denormalized_input: bool = False,
    nhwc_input: bool = False,
    uint8_input: bool = False,
    runtime: str = "onnx",
) -> str:
    """Return the canonical output filename for an export configuration.

    This is the single source of truth for output filenames — used by the TUI,
    export_tool, and sweep_export to ensure consistent naming.

    Example
    -------
    >>> build_output_filename("MDV6-yolov10-e", "float16", 640,
    ...     denormalized_input=True, nhwc_input=True, uint8_input=True,
    ...     runtime="tensorrt")
    'MDV6-yolov10-e_float16_640_denorm_nhwc_uint8input.engine'
    """
    name = f"{model_version}_{format}_{input_img_size}"
    if model_type == "yolov10_v9_compatible":
        name += "_v9_compat"
    if denormalized_input:
        name += "_denorm"
    if nhwc_input:
        name += "_nhwc"
    if uint8_input:
        name += "_uint8input"
    ext = ".engine" if runtime == "tensorrt" else ".onnx"
    return name + ext

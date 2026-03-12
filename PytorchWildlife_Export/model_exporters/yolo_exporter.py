import logging
import os
from abc import ABC
from pathlib import Path
from typing import Literal

import onnx
import torch
import torch.nn as nn

from .input_preprocessing_wrapper import InputPreprocessingWrapper
from .trt_calibration_dataset import TRTCalibrationDataLoader
from .trt_export import onnx2engine
from .util import merge_onnx_models

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# INT8 exclusion rules per model architecture
#
# These substrings are matched against node names when selecting Conv nodes
# for INT8 QDQ wrapping.  Any Conv whose name contains one of these strings
# is left at full precision to preserve accuracy.
#
# Add new entries when INT8 exclusion rules have been characterised for a
# new model family.  Raise NotImplementedError for uncharacterised families.
# ---------------------------------------------------------------------------
_INT8_EXCLUDES: dict[str, list[str]] = {
    # YOLOv10 detection head: the one2one output branches that directly
    # produce class scores and box coordinates.  Quantizing these collapses
    # confidence scores (same issue as DFL Conv in YOLOv9).
    "yolov10": ["model.23"],
    "yolov10_v9_compatible": ["model.23"],
    # YOLOv9: exclusion rules not yet characterised.
    # Add entry once sensitivity analysis is complete.
}


def _get_int8_excludes(model_type: str) -> list[str]:
    """Return the INT8 exclusion list for *model_type*, or raise if unknown."""
    if model_type not in _INT8_EXCLUDES:
        raise NotImplementedError(
            f"INT8 quantization is not yet supported for model type '{model_type}'. "
            f"The per-layer sensitivity analysis has not been performed. "
            f"Supported model types: {list(_INT8_EXCLUDES)}"
        )
    return _INT8_EXCLUDES[model_type]


class YOLOExporter(ABC):
    """
    Abstract base class for YOLO model exporters.

    Export workflow
    ---------------
    1. Export the base YOLO model to a float32 ONNX file via the ultralytics
       exporter (``_export_base_onnx``).
    2. Apply precision conversion to the base model *only* — not to any
       pre/post-processing wrappers that are merged in later steps:
         - float32 : no-op.
         - float16 : onnxruntime float16 transform (ONNX export only; for
                     TRT the FP16 flag at engine build time is sufficient).
         - int8    : explicit QDQ quantization via ``wrap_nodes_in_int8_qdq``
                     with model-type-specific exclusion rules.
    3. ``do_your_merges`` (overridable): merge subclass-specific output
       converters (e.g. YOLOv10→v9 output format adapter) on top of the
       precision-converted base model.
    4. ``add_preprocessing`` (shared): prepend the input-preprocessing
       wrapper (uint8 cast, NHWC transpose, /255 normalisation) if any of
       the corresponding flags are set.
    5a. ONNX runtime: apply float16 conversion to the fully merged model
        (if requested) and save to *output_path*.
    5b. TRT runtime: serialise the merged ONNX to a temp file and call
        ``onnx2engine`` with the appropriate precision flag.

    Note: the pre/post-processing wrappers are intentionally *not* quantized.
    They perform trivial element-wise ops whose compute cost is negligible,
    and keeping them at float32 avoids input-casting complications.
    """

    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: tuple,
        opset_version: int = 18,
        do_simplify: bool = True,
        export_format: Literal["float32", "float16", "int8"] = "float32",
        num_classes: int = 3,
        uint8_input: bool = False,
        nhwc_input: bool = False,
        denormalized_input: bool = False,
        runtime: str = "onnx",
        num_calibration_images: int = 300,
        model_type: str = "yolov10",
        **kwargs,
    ) -> None:
        """Run the full export pipeline.

        Args:
            model: Loaded ultralytics YOLO model (PyTorch).
            output_path: Destination file path (.onnx or .engine).
            input_shape: NCHW input shape for the *base* model, e.g.
                ``(1, 3, 640, 640)``.
            opset_version: ONNX opset version.
            do_simplify: Run onnxslim on the base export.
            export_format: "float32", "float16", or "int8".
            num_classes: Number of output classes.
            uint8_input: Prepend uint8→float32 cast to the model.
            nhwc_input: Prepend NHWC→NCHW transpose to the model.
            denormalized_input: Prepend /255 normalisation to the model.
            runtime: "onnx" or "tensorrt".
            num_calibration_images: Images used for INT8 calibration.
            model_type: Model architecture family ("yolov9", "yolov10",
                "yolov10_v9_compatible").  Required for INT8 to look up the
                correct exclusion rules.
        """
        if runtime not in ("onnx", "tensorrt"):
            raise ValueError(f"Unsupported runtime '{runtime}'. Use 'onnx' or 'tensorrt'.")

        # For TRT we need tensorrt importable; check early.
        if runtime == "tensorrt":
            self._ensure_tensorrt()

        # ── Step 1: Export base YOLO to float32 ONNX ─────────────────────
        base_onnx_path = self._export_base_onnx(
            model, output_path, input_shape, opset_version, do_simplify
        )
        LOGGER.info(f"Base ONNX exported to: {base_onnx_path}")

        # ── Step 2: Precision conversion on base model ────────────────────
        base_model = onnx.load(base_onnx_path)

        if export_format == "int8":
            base_model = self._apply_int8_qdq(
                base_model,
                model_type=model_type,
                input_size=input_shape[2],
                num_calibration_images=num_calibration_images,
            )

        # ── Step 3: Subclass-specific output merges ───────────────────────
        yolo_output_shape = model.model(torch.zeros(input_shape))[0].shape
        merged_model = self.do_your_merges(
            yolo_output_shape, base_model, num_classes, opset_version
        )

        # ── Step 4: Input preprocessing wrapper ──────────────────────────
        merged_model, final_input_shape = self.add_preprocessing(
            merged_model,
            input_shape,
            opset_version,
            uint8_input,
            nhwc_input,
            denormalized_input,
        )

        # ── Step 5: Save or build engine ──────────────────────────────────
        if runtime == "onnx":
            if export_format == "float16":
                merged_model = self._convert_float16(merged_model)
            onnx.save_model(merged_model, output_path)

        elif runtime == "tensorrt":
            tmp_onnx = "/tmp/merged_for_trt.onnx"
            onnx.save_model(merged_model, tmp_onnx)
            LOGGER.info(f"Merged ONNX for TRT saved to: {tmp_onnx}")

            engine_file = str(Path(output_path).with_suffix(".engine"))
            onnx2engine(
                onnx_file=tmp_onnx,
                engine_file=engine_file,
                workspace_gb=4.0,
                precision=export_format if export_format != "uint8" else "float32",
                fp16_fallback=True,
                verbose=False,
            )
            # Ensure output_path points to the engine (rename if needed)
            if engine_file != output_path:
                import shutil
                shutil.copy(engine_file, output_path)

    # -----------------------------------------------------------------------
    # Overridable hook — subclasses merge their output converters here
    # -----------------------------------------------------------------------

    def do_your_merges(
        self,
        yolo_output_shape: tuple,
        base_model: onnx.ModelProto,
        num_classes: int,
        opset_version: int,
    ) -> onnx.ModelProto:
        """Return the (possibly merged) base model.

        The default implementation is a pass-through.  Subclasses override
        this to merge an output-format converter on top of the base model.

        Args:
            yolo_output_shape: Shape of the raw YOLO output tensor, obtained
                by running a forward pass through the PyTorch model.
            base_model: Loaded (and optionally quantized) base ONNX model.
            num_classes: Number of detection classes.
            opset_version: ONNX opset used when exporting sub-modules.

        Returns:
            Merged ModelProto (or the unchanged base_model).
        """
        return base_model

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    def add_preprocessing(
        self,
        model: onnx.ModelProto,
        input_shape: tuple,
        opset_version: int,
        allow_uint8: bool,
        allow_nhwc: bool,
        allow_denormalized: bool,
    ) -> tuple[onnx.ModelProto, tuple]:
        """Prepend an input-preprocessing wrapper to *model* if requested.

        Returns:
            (merged_model, preprocessor_input_shape)
        """
        onnx_preprocessor_tmp_path = "/tmp/preprocessor.onnx"

        preprocessor = InputPreprocessingWrapper(
            allow_uint8=allow_uint8,
            allow_nhwc=allow_nhwc,
            allow_denormalized=allow_denormalized,
        )
        pre_processor_input = preprocessor.make_dummy_input(input_shape)
        torch.onnx.export(
            preprocessor,
            args=(pre_processor_input,),
            opset_version=opset_version,
            f=onnx_preprocessor_tmp_path,
            dynamo=False,
        )
        merged_model = merge_onnx_models(
            onnx_preprocessor_tmp_path,
            model,
            prefix1="Preprocessor",
            prefix2="YOLO",
        )
        return merged_model, pre_processor_input.shape

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _export_base_onnx(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: tuple,
        opset_version: int,
        do_simplify: bool,
    ) -> str:
        """Export the ultralytics model to a float32 ONNX file.

        Always exports float32 regardless of the final requested format.
        Precision conversion is applied as a subsequent step.

        Returns:
            Path to the exported ONNX file (may differ from output_path
            because ultralytics writes to its own cache directory).
        """
        export_kwargs = {
            "format": "onnx",
            "imgsz": input_shape[2],
            "batch": input_shape[0],
            "simplify": do_simplify,
            "opset": opset_version,
            "workspace": 4,
            "half": False,   # always float32; precision applied separately
            "int8": False,   # always float32; quantization applied separately
            "name": os.path.basename(output_path),
            "exist_ok": True,
            "nms": False,
        }
        return model.export(**export_kwargs)

    def _apply_int8_qdq(
        self,
        base_model: onnx.ModelProto,
        model_type: str,
        input_size: int,
        num_calibration_images: int,
    ) -> onnx.ModelProto:
        """Calibrate and wrap Conv nodes with INT8 QDQ pairs.

        Uses model-type-specific exclusion rules so that accuracy-sensitive
        layers (detection head, DFL, etc.) are kept at float32.

        Args:
            base_model: Float32 base YOLO ModelProto.
            model_type: Architecture family — used to look up exclusion list.
            input_size: Square input resolution (e.g. 640).
            num_calibration_images: Number of images for activation calibration.

        Returns:
            ModelProto with QDQ nodes inserted around the selected Conv layers.

        Raises:
            NotImplementedError: If model_type has no characterised exclusion
                rules yet.
        """
        from .quant import wrap_nodes_in_int8_qdq

        excludes = _get_int8_excludes(model_type)
        calib_loader = TRTCalibrationDataLoader(
            input_size=input_size,
            num_images=num_calibration_images,
        )
        LOGGER.info(
            f"INT8 calibration: {num_calibration_images} images, "
            f"excluding substrings {excludes}"
        )
        return wrap_nodes_in_int8_qdq(
            base_model,
            calib_loader,
            node_types=["Conv"],
            exclude=excludes,
        )

    @staticmethod
    def _convert_float16(model: onnx.ModelProto) -> onnx.ModelProto:
        """Convert model internals to float16, keeping I/O as float32."""
        from onnxruntime.transformers import float16

        return float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            node_block_list=["GatherElements", "TopK", "ArgMax", "Sigmoid"],
        )

    @staticmethod
    def _ensure_tensorrt() -> None:
        """Initialise PyTorch CUDA context and verify TensorRT is importable.

        PyTorch CUDA must be initialised *before* importing TensorRT to avoid
        cudaErrorNoDevice (error 100) when TRT's builder tries to enumerate
        CUDA devices.  This mirrors the ``select_device("0")`` call that the
        original export_tensorrt implementation performed.
        """
        import torch

        if torch.cuda.is_available():
            torch.zeros(1).cuda()
            LOGGER.info(
                f"CUDA context initialised on device: {torch.cuda.get_device_name(0)}"
            )
        else:
            LOGGER.warning("No CUDA device found; TRT export may fail.")

        try:
            import tensorrt as trt  # noqa: F401
            LOGGER.info(f"TensorRT version: {trt.__version__}")
        except ImportError:
            raise RuntimeError(
                "TensorRT is not installed. Cannot perform TensorRT export. "
                "Install tensorrt or use --runtime onnx."
            )

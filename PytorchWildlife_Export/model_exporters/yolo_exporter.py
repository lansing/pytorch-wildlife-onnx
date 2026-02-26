import logging
import os
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

LOGGER = logging.getLogger(__name__)

import onnx
import torch
import torch.nn as nn
from onnx.onnx_pb import OperatorSetIdProto
from ultralytics.nn.modules.head import v10Detect

from PytorchWildlife_Export.model_exporters.calibration_data_reader import (
    WildlifeCalibrationDataReader,
)

from .input_preprocessing_wrapper import InputPreprocessingWrapper
from .util import merge_onnx_models


@contextmanager
def _preprocessing_calibration_patch():
    """
    Context manager that monkey-patches ``tensorrt.Builder`` so that when
    ``onnx2engine`` creates its local ``EngineCalibrator`` and assigns it to
    ``config.int8_calibrator``, we intercept that assignment and replace the
    ``get_batch`` method on the calibrator instance.

    The patched ``get_batch`` forwards the tensor from our dataloader directly
    to TRT without the hard-coded ``/ 255.0`` division that ultralytics' local
    ``EngineCalibrator`` applies.  This allows calibration data to be delivered
    in whatever dtype/layout/range the merged ONNX model's input expects
    (uint8, float32 0-255, NHWC, etc.).

    Approach B from PREPROCESS_PLAN.md.
    """
    import types

    import tensorrt as trt

    _orig_builder_cls = trt.Builder

    class _ConfigProxy:
        """Wraps IBuilderConfig and intercepts ``int8_calibrator =``."""

        def __init__(self, real_cfg):
            object.__setattr__(self, "_real", real_cfg)

        def __setattr__(self, name, value):
            real = object.__getattribute__(self, "_real")
            if name == "int8_calibrator":
                # Patch get_batch on the calibrator instance so that it
                # forwards data as-is (no /255 division).
                def _patched_get_batch(self_cal, names):
                    try:
                        img = next(self_cal.data_iter)["img"]
                        img = img.contiguous()
                        if img.device.type == "cpu":
                            img = img.cuda()
                        return [int(img.data_ptr())]
                    except StopIteration:
                        return None

                value.get_batch = types.MethodType(_patched_get_batch, value)
                real.int8_calibrator = value
            else:
                setattr(real, name, value)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

    class _PatchedBuilder:
        """Wraps trt.Builder and returns _ConfigProxy from create_builder_config."""

        def __init__(self, logger):
            object.__setattr__(self, "_real", _orig_builder_cls(logger))

        def _unwrap(self, config):
            """Return the real IBuilderConfig, unwrapping _ConfigProxy if needed."""
            if isinstance(config, _ConfigProxy):
                return object.__getattribute__(config, "_real")
            return config

        def create_builder_config(self):
            real_cfg = object.__getattribute__(self, "_real").create_builder_config()
            return _ConfigProxy(real_cfg)

        def build_serialized_network(self, network, config):
            # Unwrap before the C-level type check in pybind11.
            return object.__getattribute__(self, "_real").build_serialized_network(
                network, self._unwrap(config)
            )

        def build_engine(self, network, config):
            # Older TRT API path.
            return object.__getattribute__(self, "_real").build_engine(
                network, self._unwrap(config)
            )

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_real"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_real"), name, value)

    trt.Builder = _PatchedBuilder
    LOGGER.debug("trt.Builder patched for preprocessing-aware INT8 calibration.")
    try:
        yield
    finally:
        trt.Builder = _orig_builder_cls
        LOGGER.debug("trt.Builder restored.")


class YOLOExporter(ABC):
    """
    An abstract base class for ONNX model exporters.
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
        **kwargs,
    ) -> None:
        if runtime == "onnx":
            export_fn = self.export_onnx
        elif runtime == "tensorrt":
            export_fn = self.export_tensorrt
        else:
            raise Exception(f"Unsupported runtime: {runtime}")

        export_fn(
            model,
            output_path,
            input_shape,
            opset_version,
            do_simplify,
            export_format,
            num_classes,
            uint8_input,
            nhwc_input,
            denormalized_input,
            num_calibration_images=num_calibration_images,
        )

    def export_tensorrt(
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
        num_calibration_images: int = 300,
        **kwargs,
    ) -> None:
        # Initialize PyTorch CUDA context BEFORE importing TensorRT.
        # This ordering is critical — importing tensorrt before select_device can
        # cause a cudaErrorNoDevice / CUDA initialization failure.
        from ultralytics.utils.torch_utils import select_device

        device = select_device("0", verbose=True)
        LOGGER.info(f"Active device for TensorRT export: {device}")

        try:
            import tensorrt as trt  # noqa: F401

            LOGGER.info(f"TensorRT imported successfully. Version: {trt.__version__}")
        except ImportError:
            raise RuntimeError(
                "TensorRT is not installed. Cannot perform TensorRT export."
            )
        except Exception as e:
            raise RuntimeError(f"Unexpected error importing TensorRT: {e}") from e

        onnx_base_model_path = self.export_base_onnx(
            model,
            output_path,
            input_shape,
            opset_version=opset_version,
            do_simplify=do_simplify,
        )
        LOGGER.info(f"Intermediate ONNX model at: {onnx_base_model_path}")

        yolo_output_shape = model.model(torch.zeros(input_shape))[0].shape

        # do model-specific merges (i.e. output converter)
        merged_model = self.do_your_merges(
            yolo_output_shape, onnx_base_model_path, num_classes, opset_version
        )

        # add preprocessing if needed
        merged_model, final_input_shape = self.add_preprocessing(
            merged_model,
            input_shape,
            opset_version,
            uint8_input,
            nhwc_input,
            denormalized_input,
        )

        # save merged model to a temp file for TRT conversion
        merged_onnx_tmp_path = "/tmp/merged_for_trt.onnx"
        onnx.save_model(merged_model, merged_onnx_tmp_path)
        LOGGER.info(f"Merged ONNX model saved to: {merged_onnx_tmp_path}")

        from ultralytics.utils.export.engine import (
            onnx2engine as ultralytics_onnx2engine,
        )

        from PytorchWildlife_Export.model_exporters.trt_calibration_dataset import (
            TRTCalibrationDataLoader,
        )

        engine_file = str(Path(output_path).with_suffix(".engine"))

        calibration_dataloader = None
        if export_format == "int8":
            calibration_dataloader = TRTCalibrationDataLoader(
                input_size=tuple(final_input_shape)[-1],
                num_images=num_calibration_images,
                nhwc_input=nhwc_input,
                uint8_input=uint8_input,
                denormalized_input=denormalized_input,
            )
            LOGGER.info(
                f"INT8 calibration: will stream {num_calibration_images} images "
                f"from '{TRTCalibrationDataLoader.__module__}'."
            )

        # The compensating-data trick handles nhwc_input alone: the loader
        # yields uint8 NHWC tensors and EngineCalibrator's /255 produces float32
        # NHWC 0-1, which is exactly what the merged model expects.
        #
        # For uint8_input and denormalized_input the /255 produces the wrong
        # dtype or range, so we must patch trt.Builder so the local
        # EngineCalibrator inside onnx2engine delivers data as-is.
        needs_patch = export_format == "int8" and (uint8_input or denormalized_input)
        patch_ctx = (
            _preprocessing_calibration_patch()
            if needs_patch
            else contextmanager(lambda: (yield))()
        )

        with patch_ctx:
            ultralytics_onnx2engine(
                onnx_file=merged_onnx_tmp_path,
                engine_file=engine_file,
                workspace=4,
                half=(export_format == "float16" or export_format == "int8"),
                int8=(export_format == "int8"),
                dynamic=False,
                shape=tuple(final_input_shape),
                dla=None,
                dataset=calibration_dataloader,
                metadata=None,  # prevents Ultralytics metadata header
                verbose=False,
                prefix="TRT Export: ",
            )
        LOGGER.info(f"TensorRT engine saved to: {engine_file}")
        if engine_file != output_path:
            shutil.copy(engine_file, output_path)

    def export_onnx(
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
        **kwargs,
    ) -> None:
        """
        Exports a PyTorch model to ONNX format.

        Args:
            model (nn.Module): The PyTorch model to export.
            output_path (str): The path where the ONNX model will be saved.
            input_shape (tuple): The shape of input to the ORIGINAL model (e.g., (1, 3, 640, 640)).
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph using onnx-simplifier.
            export_format (Literal["float32", "float16", "int8"]): The numeric format for export.
            num_classes:
            **kwargs: Additional arguments to pass to torch.onnx.export.
        """

        # export base model
        onnx_base_model_path = self.export_base_onnx(
            model, output_path, input_shape, opset_version, do_simplify
        )

        yolo_output_shape = model.model(torch.zeros(input_shape))[0].shape

        # do model-specific merges (i.e. output converter)
        merged_model = self.do_your_merges(
            yolo_output_shape, onnx_base_model_path, num_classes, opset_version
        )

        # add preprocessing if needed
        merged_model, final_input_shape = self.add_preprocessing(
            merged_model,
            input_shape,
            opset_version,
            uint8_input,
            nhwc_input,
            denormalized_input,
        )

        # convert/quantize
        if export_format == "float16":
            from onnxruntime.transformers import float16

            converted_model = float16.convert_float_to_float16(
                merged_model,
                keep_io_types=True,
                node_block_list=["GatherElements", "TopK", "ArgMax", "Sigmoid"],
            )
            onnx.save_model(converted_model, output_path)

        elif export_format == "int8":
            # PLACEHOLDER: ONNX INT8 static quantization (e.g. via onnxruntime quantization API).
            # Ultralytics does not support native INT8 at the ONNX export stage.
            raise NotImplementedError(
                "INT8 ONNX export is not yet implemented. "
                "Use --runtime tensorrt for INT8 quantization."
            )

        else:
            onnx.save_model(merged_model, output_path)

    def add_preprocessing(
        self,
        model,
        input_shape,
        opset_version,
        allow_uint8,
        allow_nhwc,
        allow_denormalized,
    ):
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
            dynamo=False,  # match ultralytics, otherwise IR changes
        )
        merged_model = merge_onnx_models(
            onnx_preprocessor_tmp_path,
            model,
            prefix1="Preprocessor",
            prefix2="YOLO",
        )
        return merged_model, pre_processor_input.shape

    def export_base_onnx(
        self,
        model: nn.Module,
        output_path,
        input_shape,
        opset_version,
        do_simplify: bool,
        **kwargs,
    ):
        export_kwargs = {
            "format": "onnx",
            "imgsz": input_shape[2],
            "batch": input_shape[0],
            "simplify": do_simplify,
            "opset": opset_version,
            "workspace": 4,
            "half": False,  # do the 16 bit conversion later after we merge
            # "half": True if export_format == "float16" else False,
            "int8": False,  # Force False here, as we will do static quantization separately if requested
            "name": os.path.basename(output_path),
            "exist_ok": True,
            "nms": False,  # Force NMS to False for the raw output export
            **kwargs,
        }

        onnx_base_model_path = model.export(**export_kwargs)
        return onnx_base_model_path

    def do_your_merges(
        self, yolo_output_shape, onnx_base_model_path, num_classes, opset_version
    ):
        # "null" merge... just load the base model
        return onnx.load(onnx_base_model_path)

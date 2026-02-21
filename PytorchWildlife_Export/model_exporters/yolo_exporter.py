import os
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Literal

import onnx
import torch


from .input_preprocessing_wrapper import InputPreprocessingWrapper
from .util import merge_onnx_models


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
            **kwargs
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
        onnx_base_model_path = self.export_base(model, output_path, input_shape, opset_version, do_simplify)

        yolo_output_shape = model.model(torch.zeros(input_shape))[0].shape

        # do model-specific merges (i.e. output converter)
        merged_model = self.do_your_merges(yolo_output_shape, onnx_base_model_path, num_classes, opset_version)

        # add preprocessing if needed
        merged_model = self.add_preprocessing(merged_model, input_shape, opset_version, uint8_input, nhwc_input, denormalized_input)

        # convert/quantize
        if export_format == "float16":
            from onnxruntime.transformers import float16

            converted_model = float16.convert_float_to_float16(
                merged_model, keep_io_types=True
            )
        else:
            converted_model = merged_model

        onnx.save_model(converted_model, output_path)
        return output_path


    def add_preprocessing(self, model, input_shape, opset_version, allow_uint8, allow_nhwc, allow_denormalized):
        onnx_preprocessor_tmp_path = "/tmp/preprocessor.onnx"

        preprocessor = InputPreprocessingWrapper(
            allow_uint8=allow_uint8,
            allow_nhwc=allow_nhwc,
            allow_denormalized=allow_denormalized
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
        return merged_model



    def export_base(self, model: nn.Module, output_path, input_shape, opset_version, do_simplify: bool, **kwargs):
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

    def do_your_merges(self, yolo_output_shape, onnx_base_model_path, num_classes, opset_version):
        # "null" merge... just load the base model
        return onnx.load(onnx_base_model_path)



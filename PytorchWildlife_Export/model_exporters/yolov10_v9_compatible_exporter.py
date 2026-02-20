import os
import shutil
from typing import Literal, Optional

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLO

from .input_preprocessing_wrapper import InputPreprocessingWrapper
from .onnx_exporter import ONNXExporter
from .yolov10_v9_output_converter import YOLOv10ToYOLOv9OutputConverter


def merge_onnx_models(m_1, m_2, prefix1, prefix2):
    if isinstance(m_1, str):
        m_1 = onnx.load(m_1)
    if isinstance(m_2, str):
        m_2 = onnx.load(m_2)

    m_1_output_name = [node.name for node in m_1.graph.output][0]
    m_2_input_name = [node.name for node in m_2.graph.input][0]

    for m in [m_1, m_2]:
        input_all = [node.name for node in m.graph.input]
        output_all = [node.name for node in m.graph.output]
        print("Inputs:", input_all)
        print("Outputs:", output_all)
        ir_version = getattr(m, "ir_version", 0)
        print(f"ir_version: {ir_version}")

    merged_model = onnx.compose.merge_models(
        m_1,
        m_2,
        io_map=[(m_1_output_name, m_2_input_name)],
        prefix1=prefix1,
        prefix2=prefix2,
    )
    return merged_model


class YOLOv10V9CompatibleONNXExporter(ONNXExporter):
    """
    An ONNX exporter specifically for YOLOv10 models, that outputs a tensor
    compatible with YOLOv9 raw output format.
    """

    def export(
        self,
        model: YOLO,
        output_path: str,
        preprocessor: Optional[InputPreprocessingWrapper] = None,
        input_shape: tuple = (1, 3, 1280, 1280),
        opset_version: int = 18,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16", "int8", "uint8"] = "float32",
        num_classes: int = 3,
        **kwargs,
    ) -> str:
        """
        Exports a YOLOv10 PyTorch model (ultralytics.YOLO object) to ONNX format,
        with an output layer that converts its native output to a YOLOv9-compatible format.

        Args:
            model (YOLO): The ultralytics.YOLO model to export.
            output_path (str): The desired path/filename for the ONNX model.
            input_shape (tuple): The shape of the dummy input to the model (e.g., (1, 3, 1280, 1280)).
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph.
            export_format (Literal["float32", "float16"]): The numeric format for export.
            num_classes (int): The number of classes the YOLOv10 model is trained for.
            **kwargs: Additional arguments to pass to model.export().

        Returns:
            str: The actual path to the exported ONNX model.
        """
        if not isinstance(model, YOLO):
            raise TypeError("model must be an instance of ultralytics.YOLO")

        original_yolov10_nn_module = model.model

        converter_module = YOLOv10ToYOLOv9OutputConverter(num_classes=num_classes)
        converter_module.eval()

        try:
            # Assign the patched forward method
            # TODO we try onnx compose
            # original_yolov10_nn_module.forward = new_forward.__get__(
            #     original_yolov10_nn_module, type(original_yolov10_nn_module)
            # )
            #
            #
            yolov10_output = model.model(torch.zeros(input_shape))[0]
            print(f"yolov10_output shape: {yolov10_output.shape}")

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

            print(
                f"Exporting YOLOv10 (v9 compatible output) model to ONNX using ultralytics.YOLO.export (format: {export_kwargs['format']}, opset: {export_kwargs['opset']}, nms={export_kwargs['nms']})..."
            )

            exported_model_path_from_ultralytics = model.export(**export_kwargs)
            print(
                f"Ultralytics exported model to: {exported_model_path_from_ultralytics}"
            )

            output_path_post_processing = output_path.replace(
                ".onnx", "_post_processing.onnx"
            )

            torch.onnx.export(
                converter_module,
                args=(yolov10_output,),
                opset_version=opset_version,
                f=output_path_post_processing,
                dynamo=False,  # match ultralytics, otherwise IR changes
            )

            # print(f"Loading ylo from {exported_model_path_from_ultralytics}")
            # onnx_yolov10 = onnx.load(exported_model_path_from_ultralytics)
            # onnx_post_processing = onnx.load(output_path_post_processing)

            # yolo_output_name = [node.name for node in onnx_yolov10.graph.output][0]
            # post_processing_input_name = [
            #     node.name for node in onnx_post_processing.graph.input
            # ][0]

            # for exported_model in [onnx_yolov10, onnx_post_processing]:
            #     input_all = [node.name for node in exported_model.graph.input]
            #     output_all = [node.name for node in exported_model.graph.output]
            #     print("Inputs:", input_all)
            #     print("Outputs:", output_all)
            #     ir_version = getattr(exported_model, "ir_version", 0)
            #     print(f"ir_version: {ir_version}")

            # merged_model = onnx.compose.merge_models(
            #     onnx_yolov10,
            #     onnx_post_processing,
            #     io_map=[(yolo_output_name, post_processing_input_name)],
            # )

            merged_model = merge_onnx_models(
                exported_model_path_from_ultralytics,
                output_path_post_processing,
                prefix1="yolov10",
                prefix2="v9compat",
            )

            if preprocessor:
                print("Prepending pre-processor model")
                output_path_pre_processor = output_path.replace(
                    ".onnx", "_pre_processor.onnx"
                )
                pre_processor_input = preprocessor.make_dummy_input(input_shape)
                torch.onnx.export(
                    preprocessor,
                    args=(pre_processor_input,),
                    opset_version=opset_version,
                    f=output_path_pre_processor,
                    dynamo=False,  # match ultralytics, otherwise IR changes
                )
                merged_model = merge_onnx_models(
                    output_path_pre_processor,
                    merged_model,
                    prefix1="preprocess",
                    prefix2="yolov10v9compat",
                )
                # TODO update output path

            if export_format == "float16":
                from onnxruntime.transformers import float16

                converted_model = float16.convert_float_to_float16(
                    merged_model, keep_io_types=True
                )
                # TODO update output_path
            else:
                converted_model = merged_model

            onnx.save_model(converted_model, output_path)
            return output_path

        except Exception as e:
            print(f"Error exporting YOLOv10 (v9 compatible output) model to ONNX: {e}")
            raise

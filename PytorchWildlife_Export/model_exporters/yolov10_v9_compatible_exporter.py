import torch
import torch.nn as nn
from ultralytics import YOLO
import shutil
import os
from .onnx_exporter import ONNXExporter
from typing import Literal

from .yolov10_v9_output_converter import YOLOv10ToYOLOv9OutputConverter

class YOLOv10V9CompatibleONNXExporter(ONNXExporter):
    """
    An ONNX exporter specifically for YOLOv10 models, that outputs a tensor
    compatible with YOLOv9 raw output format.
    """
    def export(
        self,
        model: YOLO,
        output_path: str,
        input_shape: tuple = (1, 3, 1280, 1280),
        opset_version: int = 18,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16", "int8", "uint8"] = "float32",
        num_classes: int = 3,
        **kwargs
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

        # Temporarily patch the forward method of the original YOLOv10 nn.Module
        original_forward = original_yolov10_nn_module.forward

        def new_forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            yolov10_native_output = original_forward(x, *args, **kwargs)

            # Ensure output is a single tensor if it comes as a list/tuple
            if isinstance(yolov10_native_output, (list, tuple)):
                yolov10_native_output = yolov10_native_output[0]
            
            # Now pass through our converter
            converted_output = converter_module(yolov10_native_output)
            return converted_output
        
        try:
            # Assign the patched forward method
            original_yolov10_nn_module.forward = new_forward.__get__(original_yolov10_nn_module, type(original_yolov10_nn_module))

            export_kwargs = {
                'format': 'onnx',
                'imgsz': input_shape[2],
                'batch': input_shape[0],
                'simplify': do_simplify,
                'opset': opset_version,
                'workspace': 4,
                'half': True if export_format == "float16" else False,
                'int8': False, # Force False here, as we will do static quantization separately if requested
                'name': os.path.basename(output_path),
                'exist_ok': True,
                'nms': False, # Force NMS to False for the raw output export
                **kwargs
            }

            # Determine temporary path for the float model from ultralytics export
            # Our ONNXExporter's _quantize_and_simplify_model expects this temporary float model.
            if export_format in ["int8", "uint8"]:
                temp_output_path_ultralytics = output_path.replace(".onnx", "_float.onnx")
                export_kwargs['name'] = os.path.basename(temp_output_path_ultralytics)
            else:
                temp_output_path_ultralytics = output_path

            print(f"Exporting YOLOv10 (v9 compatible output) model to ONNX using ultralytics.YOLO.export (format: {export_kwargs['format']}, opset: {export_kwargs['opset']}, nms={export_kwargs['nms']})...")
            
            exported_model_path_from_ultralytics = model.export(**export_kwargs)
            print(f"Ultralytics exported model to: {exported_model_path_from_ultralytics}")

            # Move the file from where ultralytics saved it to our designated temporary path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(exported_model_path_from_ultralytics, temp_output_path_ultralytics)
            print(f"Moved ultralytics exported float model to temporary path: {temp_output_path_ultralytics}")

            # Now, call our ONNXExporter's helper to handle subsequent quantization and simplification
            final_exported_path = self._quantize_and_simplify_model(
                input_onnx_path=temp_output_path_ultralytics,
                output_final_onnx_path=output_path,
                export_format=export_format,
                do_simplify=False, # not necessary because we simplified at export
                input_shape=input_shape # Pass input_shape for CalibrationDataReader
            )
            return final_exported_path
        except Exception as e:
            print(f"Error exporting YOLOv10 (v9 compatible output) model to ONNX: {e}")
            raise
        finally:
            # Restore the original forward method
            original_yolov10_nn_module.forward = original_forward

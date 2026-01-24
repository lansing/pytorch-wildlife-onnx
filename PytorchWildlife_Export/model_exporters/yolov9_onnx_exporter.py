import torch.nn as nn
from ultralytics import YOLO # Import YOLO object
import shutil # Import shutil for moving files
from .onnx_exporter import ONNXExporter
from typing import Literal

class YoloV9ONNXExporter(ONNXExporter):
    """
    An ONNX exporter specifically for YOLOv9 models, utilizing the ultralytics built-in export.
    """
    def export(
        self,
        model: YOLO, # Changed type hint to YOLO
        output_path: str, # This will now be the desired name, not directly passed to ultralytics
        input_shape: tuple = (1, 3, 1280, 1280), # Default for MDV6-yolov9-c
        opset_version: int = 18, # Use opset 18
        do_simplify: bool = False, # Simplified will be handled by ultralytics export
        export_format: Literal["float32", "float16"] = "float32",
        **kwargs
    ) -> str: # Changed return type to str
        """
        Exports a YOLOv9 PyTorch model (ultralytics.YOLO object) to ONNX format using its built-in export method.

        Args:
            model (YOLO): The ultralytics.YOLO model to export.
            output_path (str): The desired path/filename for the ONNX model.
            input_shape (tuple): The shape of the dummy input to the model (e.g., (1, 3, 1280, 1280)).
                                 Defaults to (1, 3, 1280, 1280) for MegaDetectorV6 YOLOv9 compact.
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph. This will be passed to ultralytics export.
            export_format (Literal["float32", "float16"]): The numeric format for export.
            **kwargs: Additional arguments to pass to model.export().
        
        Returns:
            str: The actual path to the exported ONNX model.
        """
        if not isinstance(model, YOLO):
            raise TypeError("model must be an instance of ultralytics.YOLO")
        
        # Prepare arguments for ultralytics export method
        export_kwargs = {
            'format': 'onnx',
            'imgsz': input_shape[2], # Assuming square input, takes height/width from input_shape
            'batch': input_shape[0], # Take batch size from input_shape
            'simplify': do_simplify,
            'opset': opset_version,
            'workspace': 4, # Default from ultralytics docs
            'half': True if export_format == "float16" else False,
            'int8': True if export_format == "int8" else False, # Add int8 here, but raise error in base for now
            **kwargs
        }

        # Check for int8 support
        if export_format == "int8":
            raise NotImplementedError("Int8 export is not supported directly through ultralytics.YOLO.export without further quantization steps.")

        print(f"Exporting YOLO model to ONNX using ultralytics.YOLO.export (format: {export_format}, opset: {opset_version})...")
        try:
            exported_model_path = model.export(**export_kwargs)
            print(f"Ultralytics exported model to: {exported_model_path}")
            
            # Ultralytics exports to a default location (usually same dir as input .pt file or runs/detect/exp/weights)
            # We want to move/rename it to the desired output_path
            shutil.move(exported_model_path, output_path)
            print(f"Moved exported ONNX model to {output_path}")
            return output_path
        except Exception as e:
            print(f"Error exporting YOLO model to ONNX: {e}")
            raise
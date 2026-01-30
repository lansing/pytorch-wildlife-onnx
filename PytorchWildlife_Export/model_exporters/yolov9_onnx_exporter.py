import torch.nn as nn
from ultralytics import YOLO # Import YOLO object
import shutil # Import shutil for moving files
import os # Import os for path manipulation
from .onnx_exporter import ONNXExporter
from typing import Literal

class YoloV9ONNXExporter(ONNXExporter):
    """
    An ONNX exporter specifically for YOLOv9 models.
    """
    def export(
        self,
        model: YOLO, # The raw ultralytics YOLO model
        output_path: str,
        input_shape: tuple = (1, 3, 1280, 1280),
        opset_version: int = 18,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16", "int8", "uint8"] = "float32", # Added int8, uint8
        **kwargs
    ) -> str:
        """
        Exports a YOLOv9 PyTorch model (ultralytics.YOLO object) to ONNX format.

        Args:
            model (YOLO): The ultralytics.YOLO model to export.
            output_path (str): The desired path/filename for the ONNX model.
            input_shape (tuple): The shape of the dummy input to the model (e.g., (1, 3, 1280, 1280)).
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph.
            export_format (Literal["float32", "float16", "int8", "uint8"]): The numeric format for export.
            **kwargs: Additional arguments to pass to model.export().

        Returns:
            str: The actual path to the exported ONNX model.
        """
        if not isinstance(model, YOLO):
            raise TypeError("model must be an instance of ultralytics.YOLO")
        
        # Prepare arguments for ultralytics.YOLO.export method for initial float32 export
        export_kwargs_ultralytics = {
            'format': 'onnx',
            'imgsz': input_shape[2],
            'batch': input_shape[0],
            'simplify': do_simplify, # Ultralytics can simplify during its export (optional)
            'opset': opset_version,
            'workspace': 4,
            'half': True if export_format == "float16" else False, # Ultralytics can export to float16
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
            export_kwargs_ultralytics['name'] = os.path.basename(temp_output_path_ultralytics)
        else:
            temp_output_path_ultralytics = output_path

        print(f"Exporting YOLO model to ONNX using ultralytics.YOLO.export (float32, opset: {export_kwargs_ultralytics['opset']}, nms={export_kwargs_ultralytics['nms']})...")
        try:
            exported_float_model_path = model.export(**export_kwargs_ultralytics)
            print(f"Ultralytics exported float model to: {exported_float_model_path}")
            
            # Move the file from where ultralytics saved it to our designated temporary path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(exported_float_model_path, temp_output_path_ultralytics)
            print(f"Moved ultralytics exported float model to temporary path: {temp_output_path_ultralytics}")

            # Now, call our ONNXExporter's helper to handle subsequent quantization and simplification
            final_exported_path = self._quantize_and_simplify_model(
                input_onnx_path=temp_output_path_ultralytics,
                output_final_onnx_path=output_path,
                export_format=export_format,
                do_simplify=do_simplify,
                input_shape=input_shape # Pass input_shape for CalibrationDataReader
            )
            return final_exported_path

        except Exception as e:
            print(f"Error exporting YOLO model to ONNX: {e}")
            raise
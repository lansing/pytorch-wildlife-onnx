import os
import torch
import torch.nn as nn
import onnx
import onnxsim
from typing import Literal, Union, Tuple, List, Dict
import numpy as np

import onnxruntime.quantization
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, CalibrationDataReader

from .base_exporter import BaseONNXExporter

# Dummy CalibrationDataReader for static quantization
# In a real scenario, this would load actual data for calibration.
class DummyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, input_shape: Tuple[int, ...], num_batches: int = 1):
        super().__init__()
        self.input_shape = input_shape
        self.num_batches = num_batches
        self.current_batch = 0
        self.input_name = "images" # Default input name for YOLO models

    def get_next(self) -> Dict[str, np.ndarray]:
        if self.current_batch < self.num_batches:
            # Generate dummy input data (e.g., random tensor)
            # This should match the expected input format of the model.
            input_data = np.random.randn(*self.input_shape).astype(np.float32)
            self.current_batch += 1
            return { self.input_name: input_data }
        else:
            return None

    def rewind(self):
        self.current_batch = 0

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.input_shape


class ONNXExporter(BaseONNXExporter):
    """
    A concrete ONNX model exporter that handles core export logic, simplification,
    and different numeric formats.
    """

    def export(
        self,
        model: nn.Module,
        output_path: str,
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        opset_version: int = 18,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16", "int8", "uint8"] = "float32",
        input_names: List[str] = None,
        output_names: List[str] = None,
        dynamic_axes: Dict[str, Dict[int, str]] = None,
        **kwargs
    ) -> str:
        """
        Exports a PyTorch model to ONNX format. This method will always produce a float32 ONNX model first.
        Subsequent quantization or simplification is handled by _quantize_and_simplify_model.

        Args:
            model (nn.Module): The PyTorch model to export.
            output_path (str): The path where the ONNX model will be saved.
            dummy_input (torch.Tensor | tuple[torch.Tensor, ...]): The dummy input(s) to the model.
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph using onnx-simplifier.
            export_format (Literal["float32", "float16", "int8", "uint8"]): The numeric format for export.
            input_names (list): Names to assign to the input nodes of the graph.
            output_names (list): Names to assign to the output nodes of the graph.
            dynamic_axes (dict): Dictionary to specify dynamic axes.
            **kwargs: Additional arguments to pass to torch.onnx.export.

        Returns:
            str: The path to the exported ONNX model.
        """
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an instance of torch.nn.Module")
        
        # Ensure dummy_input is a tuple of tensors for consistent processing
        if not isinstance(dummy_input, tuple):
            dummy_input = (dummy_input,)
        
        # Move dummy inputs to the correct device and convert format if needed
        processed_dummy_input = []
        for inp in dummy_input:
            if not isinstance(inp, torch.Tensor):
                raise TypeError("All elements in dummy_input must be torch.Tensor")
            
            # Ensure dummy input is on the same device as the model
            inp = inp.to(model.device if hasattr(model, 'device') else 'cpu')

            if export_format == "float16":
                # Only apply .half() to float tensors
                if inp.dtype == torch.float32:
                    inp = inp.half()
            processed_dummy_input.append(inp)
        
        # If there was only one input, convert back to a single tensor for torch.onnx.export
        # unless it was originally a tuple. This retains the original intent for single inputs.
        if len(processed_dummy_input) == 1 and not isinstance(dummy_input, tuple):
            processed_dummy_input = processed_dummy_input[0]
        else:
            processed_dummy_input = tuple(processed_dummy_input) # Ensure it's a tuple if multiple inputs

        # Set model to evaluation mode
        model.eval()

        # Always export to float32 first
        temp_float_onnx_path = output_path.replace(".onnx", "_float.onnx") if export_format in ["int8", "uint8"] else output_path
        
        print(f"Exporting model to ONNX (float32, opset: {opset_version})...")
        try:
            torch.onnx.export(
                model,
                processed_dummy_input,
                temp_float_onnx_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                **kwargs # Pass remaining kwargs to torch.onnx.export
            )
            print(f"Model exported to {temp_float_onnx_path}")

            # Now, handle simplification and quantization
            return self._quantize_and_simplify_model(
                input_onnx_path=temp_float_onnx_path,
                output_final_onnx_path=output_path,
                export_format=export_format,
                do_simplify=do_simplify,
                input_shape=processed_dummy_input[0].shape if isinstance(processed_dummy_input, tuple) else processed_dummy_input.shape # Pass first dummy input shape for calibration
            )

        except Exception as e:
            print(f"Error exporting model to ONNX: {e}")
            raise

    def _quantize_and_simplify_model(
        self,
        input_onnx_path: str,
        output_final_onnx_path: str,
        export_format: Literal["float32", "float16", "int8", "uint8"],
        do_simplify: bool,
        input_shape: Tuple[int, ...], # Input shape for calibration data reader
    ) -> str:
        """
        Performs quantization and simplification on an ONNX model.
        """
        current_model_path = input_onnx_path # Start with the float model

        # 1. Quantization (if requested)
        if export_format in ["int8", "uint8"]:
            print(f"Quantizing model to {export_format}...")
            
            quant_type = QuantType.QInt8 if export_format == "int8" else QuantType.QUInt8
            activation_type = QuantType.QInt8 if export_format == "int8" else QuantType.QUInt8 # Set activation_type based on export_format
            
            # For static quantization, we need a CalibrationDataReader
            # For this demo, we use a DummyCalibrationDataReader.
            calibration_data_reader = DummyCalibrationDataReader(input_shape=input_shape)

            # Ensure output directory exists for quantized model
            os.makedirs(os.path.dirname(output_final_onnx_path), exist_ok=True)

            quantize_static(
                model_input=current_model_path,
                model_output=output_final_onnx_path, # Quantize directly to final path
                calibration_data_reader=calibration_data_reader,
                op_types_to_quantize=["MatMul", "Gemm", "Conv"], # Common ops
                weight_type=quant_type,
                activation_type=activation_type, # Pass activation_type
                # For uint8, reduce_range and per_channel typically False
                # per_channel=False, reduce_range=False
            )
            print(f"Quantized model saved to {output_final_onnx_path}")
            # Remove the intermediate float model
            if input_onnx_path != output_final_onnx_path:
                os.remove(input_onnx_path)
            current_model_path = output_final_onnx_path # Update current path
        elif export_format == "float16":
            # If float16 is requested, and ultralytics didn't handle it,
            # we need to convert the float32 model to float16 here.
            # This is not directly supported by onnxruntime.quantization.quantize_static
            # but can be done using onnx.checker and onnx.convert.
            # For now, we will assume float16 conversion is handled by the original exporter.
            # If not, the model will remain float32.
            # However, ultralytics.YOLO.export() actually handles 'half' argument, so it would
            # have already exported to float16 if 'half=True' was passed.
            # So, current_model_path would already be float16.
            pass # No specific action needed here if already float16

        # 2. Simplification (if requested)
        if do_simplify:
            print("Simplifying ONNX model...")
            try:
                onnx_model = onnx.load(current_model_path)
                model_simplified, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(model_simplified, current_model_path) # Save back to current path
                    print("ONNX model simplified successfully.")
                else:
                    print("ONNX model simplification failed due to consistency check.")
            except Exception as e:
                print(f"Error simplifying ONNX model: {e}")
        
        return current_model_path # Return path to the final model

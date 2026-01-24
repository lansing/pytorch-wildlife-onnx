import os
import torch
import torch.nn as nn
import onnx
import onnxsim
from typing import Literal

from .base_exporter import BaseONNXExporter

class ONNXExporter(BaseONNXExporter):
    """
    A concrete ONNX model exporter that handles core export logic, simplification,
    and different numeric formats.
    """

    def export(
        self,
        model: nn.Module,
        output_path: str,
        dummy_input: torch.Tensor | tuple[torch.Tensor, ...], # Modified to accept single or multiple inputs
        opset_version: int = 18,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16", "int8"] = "float32",
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        **kwargs
    ) -> None:
        """
        Exports a PyTorch model to ONNX format.

        Args:
            model (nn.Module): The PyTorch model to export.
            output_path (str): The path where the ONNX model will be saved.
            dummy_input (torch.Tensor | tuple[torch.Tensor, ...]): The dummy input(s) to the model.
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph using onnx-simplifier.
            export_format (Literal["float32", "float16", "int8"]): The numeric format for export.
            input_names (list): Names to assign to the input nodes of the graph.
            output_names (list): Names to assign to the output nodes of the graph.
            dynamic_axes (dict): Dictionary to specify dynamic axes.
            **kwargs: Additional arguments to pass to torch.onnx.export.
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
            elif export_format == "int8":
                raise NotImplementedError("Int8 export is not supported in this version without further quantization steps.")
            processed_dummy_input.append(inp)
        
        # If there was only one input, convert back to a single tensor for torch.onnx.export
        # unless it was originally a tuple. This retains the original intent for single inputs.
        if len(processed_dummy_input) == 1 and not isinstance(dummy_input, tuple):
            processed_dummy_input = processed_dummy_input[0]
        else:
            processed_dummy_input = tuple(processed_dummy_input) # Ensure it's a tuple if multiple inputs

        # Set model to evaluation mode
        model.eval()

        print(f"Exporting model to ONNX (format: {export_format}, opset: {opset_version})...")
        try:
            torch.onnx.export(
                model,
                processed_dummy_input,
                output_path,
                opset_version=opset_version,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                **kwargs
            )
            print(f"Model exported to {output_path}")

            if do_simplify:
                print("Simplifying ONNX model...")
                try:
                    onnx_model = onnx.load(output_path)
                    model_simplified, check = onnxsim.simplify(onnx_model)
                    if check:
                        onnx.save(model_simplified, output_path)
                        print("ONNX model simplified successfully.")
                    else:
                        print("ONNX model simplification failed due to consistency check.")
                except Exception as e:
                    print(f"Error simplifying ONNX model: {e}")

        except Exception as e:
            print(f"Error exporting model to ONNX: {e}")
            raise

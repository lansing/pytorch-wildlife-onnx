import torch
import torch.nn as nn
from .onnx_exporter import ONNXExporter
from typing import Literal

class RTDETRONNXExporter(ONNXExporter):
    """
    An ONNX exporter specifically for RT-DETR models.
    """
    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: tuple = (1, 3, 640, 640), # Default for MDV6-apa-rtdetr-c
        orig_target_sizes_shape: tuple = (1, 2), # (batch_size, 2) for original image (width, height)
        opset_version: int = 17,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16"] = "float32",
        input_names=None,
        output_names=None,
        dynamic_axes=None,
        **kwargs
    ) -> None:
        """
        Exports an RT-DETR PyTorch model to ONNX format.

        Args:
            model (nn.Module): The PyTorch model to export.
            output_path (str): The path where the ONNX model will be saved.
            input_shape (tuple): The shape of the dummy input for images (e.g., (1, 3, 640, 640)).
                                 Defaults to (1, 3, 640, 640) for MegaDetectorV6 RT-DETR compact.
            orig_target_sizes_shape (tuple): The shape of the dummy input for original image sizes (e.g., (1, 2)).
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph using onnx-simplifier.
            export_format (Literal["float32", "float16"]): The numeric format for export.
            input_names (list): Names to assign to the input nodes of the graph.
            output_names (list): Names to assign to the output nodes of the graph.
            dynamic_axes (dict): Dictionary to specify dynamic axes.
            **kwargs: Additional arguments to pass to torch.onnx.export.
        """
        # Set default input/output names if not provided
        if input_names is None:
            input_names = ['images', 'orig_target_sizes']
        if output_names is None:
            output_names = ['labels', 'boxes', 'scores'] # RT-DETR outputs

        # Default dynamic axes for RT-DETR models
        if dynamic_axes is None:
            dynamic_axes = {
                'images': {0: 'batch_size', 2: 'height', 3: 'width'},
                'orig_target_sizes': {0: 'batch_size'},
                'labels': {0: 'batch_size'},
                'boxes': {0: 'batch_size'},
                'scores': {0: 'batch_size'}
            }
        
        # Create dummy inputs
        dummy_input_images = torch.randn(input_shape).to(model.device if hasattr(model, 'device') else 'cpu')
        dummy_input_orig_sizes = torch.tensor([[input_shape[3], input_shape[2]]]).to(model.device if hasattr(model, 'device') else 'cpu')

        # Handle float16 conversion for the model and dummy input
        if export_format == "float16":
            model.half()
            dummy_input_images = dummy_input_images.half()
            # orig_target_sizes usually remains int/long type, not float16

        super().export(
            model=model,
            output_path=output_path,
            dummy_input=(dummy_input_images, dummy_input_orig_sizes),
            opset_version=18,
            do_simplify=False,
            export_format=export_format,
            # input_names=input_names, # Removed for debugging
            # output_names=output_names, # Removed for debugging
            # dynamic_axes={}, # Removed for debugging
            **kwargs
        )

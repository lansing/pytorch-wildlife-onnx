from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Literal

class BaseONNXExporter(ABC):
    """
    An abstract base class for ONNX model exporters.
    """
    @abstractmethod
    def export(
        self,
        model: nn.Module,
        output_path: str,
        input_shape: tuple,
        opset_version: int = 17,
        do_simplify: bool = False,
        export_format: Literal["float32", "float16", "int8"] = "float32",
        **kwargs
    ) -> None:
        """
        Exports a PyTorch model to ONNX format.

        Args:
            model (nn.Module): The PyTorch model to export.
            output_path (str): The path where the ONNX model will be saved.
            input_shape (tuple): The shape of the dummy input to the model (e.g., (1, 3, 640, 640)).
            opset_version (int): The ONNX opset version to use.
            do_simplify (bool): Whether to simplify the ONNX graph using onnx-simplifier.
            export_format (Literal["float32", "float16", "int8"]): The numeric format for export.
            **kwargs: Additional arguments to pass to torch.onnx.export.
        """
        pass

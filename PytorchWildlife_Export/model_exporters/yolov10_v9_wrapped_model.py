import torch
import torch.nn as nn
from typing import Tuple, Union, List, Dict

from .yolov10_v9_output_converter import YOLOv10ToYOLOv9OutputConverter

class YOLOv10V9WrappedModel(nn.Module):
    """
    A wrapper PyTorch module that combines a YOLOv10 model's core `nn.Module`
    with our `YOLOv10ToYOLOv9OutputConverter`.

    This allows us to export the entire pipeline (YOLOv10 inference + output conversion)
    as a single ONNX model, with the final output structure compatible with YOLOv9.
    """
    def __init__(self, yolov10_model_nn: nn.Module, num_classes: int = 3):
        super().__init__()
        self.yolov10_model = yolov10_model_nn
        self.converter = YOLOv10ToYOLOv9OutputConverter(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The underlying ultralytics model's forward method
        # This is where the core YOLOv10 inference happens.
        # It might return a tuple or list of outputs depending on tracing context.
        yolov10_native_output = self.yolov10_model(x)

        # Handle potential tuple/list outputs from ultralytics's forward
        if isinstance(yolov10_native_output, (list, tuple)):
            # Assuming the main detection tensor is the first element
            # This is crucial for torch.export to correctly trace the graph.
            yolov10_native_output = yolov10_native_output[0]
        
        # Ensure the YOLOv10 output is the expected (B, N, 6) format
        if yolov10_native_output.ndim == 3 and yolov10_native_output.shape[-1] == 6:
            # Pass through our converter to get YOLOv9 compatible output
            converted_output = self.converter(yolov10_native_output)
            return converted_output
        else:
            # If the output format is unexpected, we might have issues.
            # For robustness, we could try to handle other ultralytics output formats here
            # or raise an informative error.
            raise ValueError(f"Unexpected YOLOv10 native output shape for conversion: {yolov10_native_output.shape}")

if __name__ == '__main__':
    # This example demonstrates how to use the WrappedModel
    # You would typically load a pretrained YOLOv10 model here.
    # For a standalone test, we'll create a dummy YOLOv10-like output.

    num_classes = 3
    
    # Create a dummy YOLOv10 model's nn.Module (just for forward pass test)
    # In reality, this would be model_pt.model from YoloV9Loader
    class DummyYOLOv10Model(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate an output layer that produces (B, N, 6)
            self.output_layer = nn.Linear(10, num_classes + 5) # Dummy
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Simulate some processing and then output (B, N, 6)
            # Input x shape: (B, C, H, W)
            # Flatten or pool x to get (B, features)
            dummy_features = torch.randn(x.shape[0], 10) # Dummy features
            # Simulate detections: (B, num_detections, 6)
            num_detections = 5
            # Generate dummy (x1, y1, x2, y2, confidence, class_id)
            dummy_output = torch.rand(x.shape[0], num_detections, 6)
            dummy_output[..., 4] = dummy_output[..., 4] * 0.9 + 0.1 # Confidence [0.1, 1.0]
            dummy_output[..., 5] = torch.randint(0, num_classes, (x.shape[0], num_detections, 1), dtype=torch.float32)
            return dummy_output

    dummy_yolov10_nn_module = DummyYOLOv10Model()
    
    # Create an instance of our wrapper
    wrapped_model = YOLOv10V9WrappedModel(dummy_yolov10_nn_module, num_classes=num_classes)

    # Create a dummy input for the wrapped model
    dummy_input = torch.randn(1, 3, 1280, 1280) # Simulate image input

    # Run forward pass
    output_yolov9_compatible = wrapped_model(dummy_input)

    print(f"Wrapped model output shape: {output_yolov9_compatible.shape}")
    # Expected output shape: (1, 7, num_detections) (e.g., (1, 7, 5))
    # yseq[0] = xc, yseq[1] = yc, yseq[2] = w, yseq[3] = h
    # yseq[4] = c0_score, yseq[5] = c1_score, yseq[6] = c2_score
    print("\nConverted YOLOv9 output (transposed for readability):")
    print(output_yolov9_compatible[0].transpose(0,1))
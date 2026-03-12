import torch
import onnx

from .util import merge_onnx_models
from .yolo_exporter import YOLOExporter
from .yolov10_v9_output_converter import YOLOv10ToYOLOv9OutputConverter


class YOLOv10V9CompatibleONNXExporter(YOLOExporter):
    """
    YOLO exporter that appends a YOLOv10→YOLOv9 output-format converter.

    The base YOLOv10 model outputs ``(B, 300, 6)`` one2one predictions.
    This subclass merges an output-converter module that reshapes the raw
    detections into the YOLOv9 ``(B, 84, N)`` layout expected by downstream
    YOLOv9-compatible post-processors.
    """

    def do_your_merges(
        self,
        yolo_output_shape: tuple,
        base_model: onnx.ModelProto,
        num_classes: int,
        opset_version: int,
    ) -> onnx.ModelProto:
        converter_module = YOLOv10ToYOLOv9OutputConverter(num_classes=num_classes)
        converter_module.eval()
        tmp_converter_path = "/tmp/yolo_v10v9_converter.onnx"

        yolov10_output = torch.zeros(yolo_output_shape)
        torch.onnx.export(
            converter_module,
            args=(yolov10_output,),
            opset_version=opset_version,
            f=tmp_converter_path,
            dynamo=False,
        )

        merged_model = merge_onnx_models(
            base_model,
            tmp_converter_path,
            prefix1="YOLOV10",
            prefix2="YOLOv10ToYOLOv9OutputConverter",
        )
        return merged_model

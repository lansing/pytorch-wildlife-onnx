import os
import shutil
from typing import Literal, Optional

import onnx
import torch
import torch.nn as nn
from ultralytics import YOLO

from .input_preprocessing_wrapper import InputPreprocessingWrapper
from .util import merge_onnx_models
from .yolo_exporter import YOLOExporter
from .yolov10_v9_output_converter import YOLOv10ToYOLOv9OutputConverter



class YOLOv10V9CompatibleONNXExporter(YOLOExporter):
    """
    An ONNX exporter specifically for YOLOv10 models, that outputs a tensor
    compatible with YOLOv9 raw output format.
    """

    def do_your_merges(self, yolo_output_shape, onnx_base_model_path, num_classes, opset_version):
        converter_module = YOLOv10ToYOLOv9OutputConverter(num_classes=num_classes)
        converter_module.eval()
        tmp_output_path = "/tmp/yolo_v10v9_merged.onnx"

        yolov10_output = torch.zeros(yolo_output_shape)

        torch.onnx.export(
            converter_module,
            args=(yolov10_output,),
            opset_version=opset_version,
            f=tmp_output_path,
            dynamo=False,  # match ultralytics, otherwise IR changes
        )

        merged_model = merge_onnx_models(
            onnx_base_model_path,
            tmp_output_path,
            prefix1="YOLOV10",
            prefix2="YOLOv10ToYOLOv9OutputConverter",
        )

        return merged_model

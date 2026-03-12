import torch.nn as nn
import shutil
import os
from typing import Literal

from .yolo_exporter import YOLOExporter


class YoloV9ONNXExporter(YOLOExporter):
    """
    An ONNX exporter specifically for YOLOv9 models.
    """

    pass
from typing import Tuple

import torch
import torch.nn as nn


class YOLOv10ToYOLOv9OutputConverter(nn.Module):
    """
    A PyTorch module that transforms the raw YOLOv10 model output (post-NMS-like)
    into a format compatible with YOLOv9 raw (pre-NMS) output structure.

    YOLOv10 output: (batch_size, num_detections, 6) where 6 is [x1, y1, x2, y2, confidence, class_id]
    YOLOv9 target output: (batch_size, 7, num_detections_from_v10) where 7 is [xc, yc, w, h, c0_score, c1_score, c2_score]

    This module will convert the 6 attributes into 7 attributes and transpose the output.
    The num_proposals will be num_detections from YOLOv10 output.
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        if self.num_classes != 3:
            # The current YOLOv9 postprocessor assumes 3 classes
            raise NotImplementedError(
                "Currently only supports 3 classes for YOLOv9 output compatibility."
            )

    def forward(self, yolov10_output: torch.Tensor) -> torch.Tensor:
        """
        Transforms YOLOv10 output to a YOLOv9-compatible output.

        Args:
            yolov10_output (torch.Tensor): Output from YOLOv10 model, shape (B, N, 6)
                                           where 6 is [x1, y1, x2, y2, confidence, class_id].

        Returns:
            torch.Tensor: Transformed output, shape (B, 7, N)
                          where 7 is [xc, yc, w, h, c0_score, c1_score, c2_score].
        """
        batch_size, num_detections, _ = yolov10_output.shape

        # Extract components
        # Using slicing instead of split to maintain tensor dimensions and avoid potential tracing issues
        x1 = yolov10_output[..., 0:1]
        y1 = yolov10_output[..., 1:2]
        x2 = yolov10_output[..., 2:3]
        y2 = yolov10_output[..., 3:4]
        confidence = yolov10_output[..., 4:5]
        class_id = yolov10_output[..., 5:6]

        # Convert xyxy to xywh
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Create class score vectors
        # Initialize with zeros (B, N, num_classes)
        class_scores = torch.zeros(
            batch_size, num_detections, self.num_classes, device=yolov10_output.device
        )

        # Populate class score for the detected class with its confidence
        # Clamp to ensure indices are ALWAYS valid [0, num_classes - 1]
        # TODO clampo if we have precision issues
        class_id_long = class_id.detach().long()
        class_id_long = torch.clamp(class_id_long, 0, self.num_classes - 1)
        # Use scatter_ to place confidence at the class_id index
        # class_id needs to be long for scatter_
        # scatter_ expects source and index to have compatible shapes
        class_scores.scatter_(dim=-1, index=class_id_long, src=confidence)

        # Concatenate boxes (xywh) and class scores
        # Resulting shape: (B, N, 4 + num_classes) -> (B, N, 7)
        yolov9_compatible_output_flat = torch.cat([xc, yc, w, h, class_scores], dim=-1)

        # Transpose to (B, 7, N)
        # This matches the (batch_size, num_attributes, num_predictions) format of YOLOv9
        yolov9_compatible_output = yolov9_compatible_output_flat.transpose(1, 2)

        return yolov9_compatible_output

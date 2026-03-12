import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Any

from ultralytics.utils.ops import xywh2xyxy, scale_boxes, clip_boxes # Used for reference, not directly called
from ultralytics.utils.nms import non_max_suppression # Used for reference, not directly called

from .util import scale_boxes_np


class YOLOv10PostProcessor():
    """
    Custom post-processor for YOLOv10 models exported to ONNX (raw output, pre-NMS).
    This class handles converting raw ONNX output (typically [1, 300, 6]) into
    structured detection results, including NMS and scaling, matching ultralytics's methodology.

    Assumes raw_output: [1, num_detections, 6] where 6:
    [x1, y1, x2, y2, confidence_score, class_id] - Already NMS-ready
    """

    def postprocess(
        self,
        raw_output: np.ndarray,
        original_dims: Tuple[int, int], # (width, height)
        input_shape: List[int], # [batch_size, channels, height, width]
        confidence_threshold: float,
        iou_threshold: float, # Not directly used for NMS as NMS is assumed to be in ONNX graph
        class_names: Dict[int, str],
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]] # ((gain_w, gain_h), (pad_x, pad_y))
    ) -> List[Dict]:
        """
        Converts raw ONNX output tensors into a list of structured detection results.

        Args:
            raw_output (np.ndarray): The raw output tensor from the ONNX model.
                                     Expected to be in format [1, num_detections, 6]
            original_dims (Tuple[int, int]): Original (width, height) of the image.
            input_shape (List[int]): Expected input shape of the ONNX model [batch_size, channels, height, width].
            confidence_threshold (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS). (Not used here)
            class_names (Dict[int, str]): Mapping of class IDs to class names.
            ratio_pad (Tuple[Tuple[float, float], Tuple[float, float]]): Scaling ratios and padding values
                                                                         from preprocessing.

        Returns:
            List[Dict]: A list of dictionaries, each representing a detected object.
                        Each dict contains 'box' (xyxy), 'confidence', 'class_id', 'class_name'.
        """
        # Output is expected to be [batch_size, num_detections, 6]
        # where 6 is [x1, y1, x2, y2, confidence, class_id]
        detections_raw = raw_output[0] # Get detections for the first image in batch
        
        # Filter by confidence threshold
        confidences = detections_raw[:, 4]
        class_ids_raw = detections_raw[:, 5].astype(int) # Ensure class IDs are integers

        keep_indices = confidences > confidence_threshold
        detections_filtered = detections_raw[keep_indices]

        if len(detections_filtered) == 0:
            return []

        boxes_xyxy_normalized = detections_filtered[:, :4]
        confidences_filtered = detections_filtered[:, 4]
        class_ids_filtered = detections_filtered[:, 5].astype(int)

        # Scale bounding boxes back to original image dimensions
        input_h, input_w = input_shape[2], input_shape[3]
        original_h, original_w = original_dims[1], original_dims[0] # Note: original_dims is (width, height)

        scaled_boxes_np_array = scale_boxes_np(
            img1_shape=(input_h, input_w), 
            boxes=boxes_xyxy_normalized, # already xyxy
            img0_shape=(original_h, original_w),
            ratio_pad=ratio_pad,
            padding=True # Always True for LetterBox
        )
        
        final_detections = []
        for i in range(len(scaled_boxes_np_array)):
            box = scaled_boxes_np_array[i]
            score = confidences_filtered[i]
            class_id = class_ids_filtered[i]
            final_detections.append({
                "box": [int(b) for b in box], # x1, y1, x2, y2
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), "unknown")
            })
        return final_detections

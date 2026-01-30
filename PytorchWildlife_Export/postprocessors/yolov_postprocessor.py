import numpy as np
import cv2
import torch # Import torch
from typing import List, Dict, Tuple, Any

# Import ultralytics NMS and ops
from ultralytics.utils.ops import xywh2xyxy, scale_boxes, clip_boxes # Import ultralytics's ops functions
from ultralytics.utils.nms import non_max_suppression

from PytorchWildlife_Export.postprocessors.base_postprocessor import BasePostProcessor

class YOLOvPostProcessor(BasePostProcessor):
    """
    Custom post-processor for YOLOv8/YOLOv9/YOLOv10 models exported to ONNX (raw output, pre-NMS).
    This class handles converting raw ONNX output (typically [1, 7, N]) into
    structured detection results, including NMS and scaling, matching ultralytics's methodology.

    Assumes raw_output: [1, num_attributes, num_predictions] where num_attributes = 7:
    [x_center, y_center, width, height, class_0_score, class_1_score, class_2_score]
    """

    def postprocess(
        self,
        raw_output: np.ndarray,
        original_dims: Tuple[int, int], # (width, height)
        input_shape: List[int], # [batch_size, channels, height, width]
        confidence_threshold: float,
        iou_threshold: float,
        class_names: Dict[int, str],
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]] # ((gain, gain), (pad_x, pad_y))
    ) -> List[Dict]:
        """
        Converts raw ONNX output tensors into a list of structured detection results.

        Args:
            raw_output (np.ndarray): The raw output tensor from the ONNX model.
                                     Expected to be in format [1, num_attributes, num_predictions]
            original_dims (Tuple[int, int]): Original (width, height) of the image.
            input_shape (List[int]): Expected input shape of the ONNX model [batch_size, channels, height, width].
            confidence_threshold (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
            class_names (Dict[int, str]): Mapping of class IDs to class names.
            ratio_pad (Tuple[Tuple[float, float], Tuple[float, float]]): Scaling ratios and padding values
                                                                         from preprocessing.

        Returns:
            List[Dict]: A list of dictionaries, each representing a detected object.
                        Each dict contains 'box' (xyxy), 'confidence', 'class_id', 'class_name'.
        """
        # 1. Convert numpy raw_output to torch.Tensor for ultralytics NMS
        # raw_output is (batch_size, num_attributes, num_predictions) -> (1, 7, 33600)
        torch_prediction = torch.from_numpy(raw_output)

        # 2. Call ultralytics's non_max_suppression function
        # nc will be derived as prediction.shape[1] - 4 = 7 - 4 = 3
        detections_torch_xyxy_scaled = non_max_suppression(
            prediction=torch_prediction,
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
            nc=len(class_names), # Explicitly pass number of classes (3 for MDV6)
            max_det=300 # Default max_det
        )

        # detections_torch_xyxy_scaled is a list of tensors, one per image in batch.
        # For batch_size=1, it will contain one tensor of shape (num_detections, 6)
        # where 6 is (x1, y1, x2, y2, confidence, class_id)
        
        detections_np = []
        if detections_torch_xyxy_scaled and len(detections_torch_xyxy_scaled[0]) > 0: # Check if there are detections
            detections_np = detections_torch_xyxy_scaled[0].cpu().numpy() # Convert back to numpy

        if len(detections_np) == 0:
            return []

        # 3. Scale bounding boxes back to original image dimensions
        # The detections from ultralytics NMS are already in xyxy format, but scaled to model input size (e.g., 1280x1280)
        # We need to use ultralytics.utils.ops.scale_boxes for accurate unscaling.
        
        input_h, input_w = input_shape[2], input_shape[3]
        original_h, original_w = original_dims[1], original_dims[0] # Note: original_dims is (width, height)

        # boxes are xyxy format from ultralytics NMS output
        boxes = detections_np[:, :4] 
        scores = detections_np[:, 4]
        class_ids = detections_np[:, 5]

        # Use ultralytics's scaling function
        # ops.scale_boxes signature: (img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False)
        # img1_shape: (height, width) of the image *after* letterbox, which is (input_h, input_w)
        # img0_shape: (height, width) of the original image
        # ratio_pad: ((gain, gain), (pad_x, pad_y))
        
        # Convert boxes to torch tensor for scale_boxes
        boxes_torch = torch.from_numpy(boxes).float()
        
        scaled_boxes_torch = scale_boxes(
            img1_shape=(input_h, input_w), 
            boxes=boxes_torch, 
            img0_shape=(original_h, original_w),
            ratio_pad=ratio_pad, # Pass ratio_pad
            padding=True # Assuming padding was applied during LetterBox
        )
        # clip_boxes is called internally by scale_boxes
        final_boxes_np = scaled_boxes_torch.cpu().numpy()

        final_detections = []
        for i in range(len(final_boxes_np)):
            box = final_boxes_np[i]
            score = scores[i]
            class_id = class_ids[i]
            final_detections.append({
                "box": [int(b) for b in box], # x1, y1, x2, y2
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), "unknown")
            })

        return final_detections

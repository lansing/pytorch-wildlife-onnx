import numpy as np
import cv2
from typing import List, Dict, Tuple

from PytorchWildlife_Export.postprocessors.base_postprocessor import BasePostProcessor

# Helper function inspired by ultralytics.utils.ops.xywh2xyxy
def xywh2xyxy_np(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format.
    Numpy version of ultralytics.utils.ops.xywh2xyxy.
    """
    y = np.empty_like(x)
    xy = x[..., :2]
    wh_half = x[..., 2:] / 2
    y[..., 0] = xy[..., 0] - wh_half[..., 0] # x1
    y[..., 1] = xy[..., 1] - wh_half[..., 1] # y1
    y[..., 2] = xy[..., 0] + wh_half[..., 0] # x2
    y[..., 3] = xy[..., 1] + wh_half[..., 1] # y2
    return y

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
        class_names: Dict[int, str]
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

        Returns:
            List[Dict]: A list of dictionaries, each representing a detected object.
                        Each dict contains 'box' (xyxy), 'confidence', 'class_id', 'class_name'.
        """
        # Raw output is (batch_size, num_attributes, num_predictions) -> (1, 7, 33600)
        # Ultralytics: num_attributes = 4 (bbox) + nc (3 classes) = 7
        
        # We process a single image at a time (batch_size = 1)
        predictions_single_image = raw_output[0] # Shape (7, 33600)

        # 1. Extract class scores (ultralytics 'xc' logic)
        # Class scores are at indices 4, 5, 6
        class_scores_raw = predictions_single_image[4:7, :] # Shape (3, 33600)
        
        # Max score across classes for each prediction
        max_class_scores = np.max(class_scores_raw, axis=0) # Shape (33600,)
        
        # Filter by confidence threshold (like ultralytics 'xc')
        candidates_mask = max_class_scores > confidence_threshold
        
        # Apply mask to filter predictions
        predictions_filtered = predictions_single_image[:, candidates_mask] # Shape (7, num_candidates)
        max_class_scores_filtered = max_class_scores[candidates_mask] # Shape (num_candidates,)

        if predictions_filtered.shape[1] == 0: # No candidates left
            return []

        # 2. Transpose to (num_candidates, num_attributes) for easier processing
        # This makes it (num_candidates, 7)
        predictions_transposed = predictions_filtered.transpose(1, 0)

        # 3. Extract components from filtered, transposed predictions
        boxes_xywh = predictions_transposed[:, :4] # (num_candidates, 4) in xywh
        class_scores_filtered = predictions_transposed[:, 4:7] # (num_candidates, 3)

        # 4. Convert xywh to xyxy
        boxes_xyxy_unscaled = xywh2xyxy_np(boxes_xywh) # (num_candidates, 4) in xyxy

        # 5. Get final confidence and class IDs
        # max_class_scores_filtered is already the final confidence
        final_scores = max_class_scores_filtered
        class_ids = np.argmax(class_scores_filtered, axis=1) # (num_candidates,)

        # 6. Scale bounding boxes to original image dimensions
        input_h, input_w = input_shape[2], input_shape[3]
        scale_w = original_dims[0] / input_w
        scale_h = original_dims[1] / input_h

        final_boxes_scaled = np.copy(boxes_xyxy_unscaled)
        final_boxes_scaled[:, 0] = np.clip(final_boxes_scaled[:, 0] * scale_w, 0, original_dims[0])
        final_boxes_scaled[:, 1] = np.clip(final_boxes_scaled[:, 1] * scale_h, 0, original_dims[1])
        final_boxes_scaled[:, 2] = np.clip(final_boxes_scaled[:, 2] * scale_w, 0, original_dims[0])
        final_boxes_scaled[:, 3] = np.clip(final_boxes_scaled[:, 3] * scale_h, 0, original_dims[1])

        # 7. Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(
            bboxes=final_boxes_scaled.tolist(), 
            scores=final_scores.tolist(), 
            score_threshold=confidence_threshold, # NMS uses score_threshold to filter boxes
            nms_threshold=iou_threshold
        )
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = final_boxes_scaled[i]
                score = final_scores[i]
                class_id = class_ids[i]
                detections.append({
                    "box": [int(b) for b in box], # x1, y1, x2, y2
                    "confidence": float(score),
                    "class_id": int(class_id),
                    "class_name": class_names.get(int(class_id), "unknown")
                })
        return detections
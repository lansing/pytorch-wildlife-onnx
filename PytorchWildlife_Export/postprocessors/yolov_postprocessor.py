import numpy as np
import cv2
import torch # Keep torch for now to convert numpy array to torch tensor for non_max_suppression_np
from typing import List, Dict, Tuple, Any

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

def clip_boxes_np(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Clip bounding boxes to image boundaries (Numpy version)."""
    h, w = shape[:2]
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, w)  # x1, x2
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, h)  # y1, y2
    return boxes

def scale_boxes_np(
    img1_shape: Tuple[int, int], # (height, width) of the image *after* letterbox
    boxes: np.ndarray, # Bounding boxes in xyxy format, scaled to img1_shape
    img0_shape: Tuple[int, int], # (height, width) of the original image
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]], # ((gain_w, gain_h), (pad_x, pad_y))
    padding: bool = True # Whether boxes are based on YOLO-style augmented images with padding.
) -> np.ndarray:
    """Rescale bounding boxes from img1_shape to img0_shape (Numpy version)."""
    gain, (pad_x, pad_y) = ratio_pad

    if padding:
        boxes[..., 0] -= pad_x  # x padding
        boxes[..., 1] -= pad_y  # y padding
        boxes[..., 2] -= pad_x  # x padding
        boxes[..., 3] -= pad_y  # y padding
    
    # Apply gain (ultralytics applies gain to all 4 coords, this is correct for xyxy)
    boxes[..., :4] /= gain[0] # Assuming gain_w == gain_h

    return clip_boxes_np(boxes, img0_shape)

def non_max_suppression_np(
    prediction: np.ndarray, # (num_predictions, num_attributes) -> (33600, 7)
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    nc: int = 3, # number of classes
    agnostic: bool = False, # Whether to perform class-agnostic NMS.
) -> List[np.ndarray]: # Returns a list of detections, each np.ndarray (N, 6)
    
    # 1. Split prediction into boxes and class scores
    # prediction: (num_predictions, 7)
    # 4 (bbox) + 3 (class scores)
    boxes_xywh = prediction[:, :4] # (num_predictions, 4)
    class_scores = prediction[:, 4:] # (num_predictions, nc) -> (33600, 3)

    # 2. Convert boxes from xywh to xyxy
    boxes_xyxy = xywh2xyxy_np(boxes_xywh)

    # 3. Get best class score and class ID for each box
    # conf: (num_predictions,) maximum class score
    # j: (num_predictions,) class ID
    conf = np.max(class_scores, axis=1)
    j = np.argmax(class_scores, axis=1)

    # 4. Filter by confidence threshold
    # filt: boolean array (num_predictions,)
    filt = conf > conf_thres
    
    boxes_xyxy = boxes_xyxy[filt]
    conf = conf[filt]
    j = j[filt]

    # If no boxes remain after filtering, return empty
    if len(boxes_xyxy) == 0:
        return [np.empty((0, 6))] # Return empty list conforming to ultralytics output structure

    # 5. Apply batched NMS trick for class-aware NMS (similar to torchvision.ops.nms)
    # The idea is to offset boxes of different classes so that NMS doesn't suppress boxes
    # from different classes if they overlap.
    max_coordinate = np.max(boxes_xyxy)
    offsets = j * (max_coordinate + 1)
    boxes_for_nms = boxes_xyxy + offsets[:, None]

    # 6. Perform NMS using cv2.dnn.NMSBoxes (or another numpy NMS if needed)
    # cv2.dnn.NMSBoxes is efficient but works on a list of boxes.
    # We need to adapt it for batched NMS.

    # This is the tricky part to get exactly right as ultralytics uses TorchNMS or torchvision.ops.nms
    # cv2.dnn.NMSBoxes is class-agnostic by default, but the offsetting trick makes it class-aware.
    
    # Prepare inputs for cv2.dnn.NMSBoxes
    # boxes_for_nms.tolist() needs to be float32 for NMSBoxes
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_for_nms.astype(np.float32).tolist(), 
        scores=conf.astype(np.float32).tolist(), 
        score_threshold=conf_thres, # NMS uses score_threshold to filter boxes, even if already filtered
        nms_threshold=iou_thres,
        top_k=max_det # Limit detections to max_det
    )

    if len(indices) == 0:
        return [np.empty((0, 6))]

    # Extract results
    keep_indices = indices.flatten()
    final_boxes = boxes_xyxy[keep_indices]
    final_conf = conf[keep_indices]
    final_j = j[keep_indices]

    # Combine into (x1, y1, x2, y2, confidence, class_id) format
    # detections: (num_final_detections, 6)
    detections = np.concatenate((final_boxes, final_conf[:, None], final_j[:, None]), axis=1)
    
    # max_det filtering (ultralytics does this after NMS)
    if len(detections) > max_det:
        detections = detections[np.argsort(detections[:, 4])[::-1]][:max_det] # Sort by confidence

    return [detections] # Wrap in a list for batch_size = 1


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
        ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]] # ((gain_w, gain_h), (pad_x, pad_y))
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
        # raw_output is (batch_size, num_attributes, num_predictions) -> (1, 7, 33600)
        
        # 1. Initial Filtering and NMS using ultralytics-like logic
        # Convert to (num_predictions, num_attributes) -> (33600, 7) for single image processing
        predictions_raw = raw_output[0].transpose(1, 0)

        # Assuming 3 classes (animal, person, vehicle)
        num_classes = len(class_names) # Should be 3

        # Call our custom NMS (reimplemented from ultralytics's logic)
        detections_final_np = non_max_suppression_np(
            prediction=predictions_raw,
            conf_thres=confidence_threshold,
            iou_thres=iou_threshold,
            nc=num_classes,
            max_det=300 # Default for ultralytics
        )
        
        if not detections_final_np or len(detections_final_np[0]) == 0: # non_max_suppression_np returns list of arrays
            return []
        
        detections_final_np = detections_final_np[0] # Get the detections array for the first image
        
        # detections_final_np is (num_final_detections, 6) -> (x1, y1, x2, y2, confidence, class_id)
        
        # 2. Scale bounding boxes back to original image dimensions
        input_h, input_w = input_shape[2], input_shape[3]
        original_h, original_w = original_dims[1], original_dims[0] # Note: original_dims is (width, height)

        boxes_xyxy = detections_final_np[:, :4] 
        scores = detections_final_np[:, 4]
        class_ids = detections_final_np[:, 5]

        # Use our custom scaling function
        scaled_boxes_np_array = scale_boxes_np(
            img1_shape=(input_h, input_w), 
            boxes=boxes_xyxy, # already xyxy
            img0_shape=(original_h, original_w),
            ratio_pad=ratio_pad,
            padding=True # Always True for LetterBox
        )
        
        final_detections = []
        for i in range(len(scaled_boxes_np_array)):
            box = scaled_boxes_np_array[i]
            score = scores[i]
            class_id = class_ids[i]
            final_detections.append({
                "box": [int(b) for b in box], # x1, y1, x2, y2
                "confidence": float(score),
                "class_id": int(class_id),
                "class_name": class_names.get(int(class_id), "unknown")
            })

        return final_detections

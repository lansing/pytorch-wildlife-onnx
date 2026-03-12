from typing import Tuple

import numpy as np


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
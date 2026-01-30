from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Tuple

class BasePostProcessor(ABC):
    """
    Abstract base class for ONNX model post-processors.
    Defines the interface for converting raw ONNX output tensors into structured detection results.
    """

    @abstractmethod
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
            raw_output (np.ndarray): The raw output tensor(s) from the ONNX model.
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
        pass

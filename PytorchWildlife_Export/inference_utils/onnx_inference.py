import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os
import math
from typing import List, Dict, Tuple, Union

# Try to import ultralytics, but don't fail if not present (e.g., for non-YOLO ONNX models)
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    YOLO = None
    Results = None
    print("Ultralytics not found. YOLO-specific post-processing will not be available.")

from PytorchWildlife_Export.postprocessors.base_postprocessor import BasePostProcessor

class ONNXInferenceSession:
    """
    Manages ONNX model loading, inference execution, and post-processing for object detection.
    This class always uses onnxruntime for raw inference. Post-processing is delegated to a
    provided PostProcessor instance.
    """

    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        self.session: ort.InferenceSession = None
        self.input_name = None
        self.input_shape = None # [batch_size, channels, height, width]
        self.output_name = None
        
        self._load_model()

    def _load_model(self):
        """Loads the ONNX model using onnxruntime."""
        self.session = ort.InferenceSession(self.onnx_model_path, providers=ort.get_available_providers())
        
        # Get input/output names and shapes for onnxruntime session
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]

        self.input_name = input_meta.name
        self.input_shape = input_meta.shape # e.g., [1, 3, 1280, 1280]
        self.output_name = output_meta.name

        print(f"ONNX Model loaded via onnxruntime: {self.onnx_model_path}")
        print(f"Input Name: {self.input_name}, Input Shape: {self.input_shape}")
        print(f"Output Name: {self.output_name}")

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses a single image to the format expected by the ONNX model for onnxruntime.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]:
                - resized_image_chw: The preprocessed image as a NumPy array (CHW).
                - original_dims: Original (width, height) of the image.
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        original_dims = (original_image.shape[1], original_image.shape[0]) # (width, height)

        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        input_h, input_w = self.input_shape[2], self.input_shape[3]
        resized_image = cv2.resize(img_rgb, (input_w, input_h), interpolation=cv2.INTER_AREA)

        resized_image_chw = resized_image.astype(np.float32) / 255.0
        resized_image_chw = np.transpose(resized_image_chw, (2, 0, 1))  # HWC to CHW
        resized_image_chw = np.expand_dims(resized_image_chw, axis=0) # Add batch dimension

        return resized_image_chw, original_dims

    def run_inference(
        self,
        image_path: str,
        post_processor: BasePostProcessor,
        confidence_threshold: float,
        iou_threshold: float,
        class_names: Dict[int, str]
    ) -> List[Dict]:
        """
        Runs inference on a single image using onnxruntime and delegates post-processing.

        Args:
            image_path (str): Path to the input image.
            post_processor (BasePostProcessor): An instance of a post-processor.
            confidence_threshold (float): Confidence threshold for filtering detections.
            iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
            class_names (Dict[int, str]): Mapping of class IDs to class names.

        Returns:
            List[Dict]: A list of detected objects with bounding boxes, confidence, and class info.
        """
        preprocessed_image, original_dims = self.preprocess_image(image_path)
        
        # Run inference with onnxruntime
        raw_output = self.session.run([self.output_name], {self.input_name: preprocessed_image})[0]
        
        # Delegate post-processing to the provided post_processor
        detections = post_processor.postprocess(
            raw_output=raw_output,
            original_dims=original_dims,
            input_shape=self.input_shape,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names
        )
        
        return detections
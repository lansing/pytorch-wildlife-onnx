import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Try to import ultralytics, but don't fail if not present (e.g., for non-YOLO ONNX models)
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    from ultralytics.utils.ops import (
        clip_boxes,  # Import ultralytics's clip_boxes
        scale_boxes,  # Import ultralytics's scale_boxes
    )
except ImportError:
    YOLO = None
    Results = None
    scale_boxes = None
    clip_boxes = None
    print("Ultralytics not found. YOLO-specific features might be limited.")


class ONNXInferenceSession:
    """
    Manages ONNX model loading, inference execution, and post-processing for object detection.
    This class always uses onnxruntime for raw inference. Post-processing is delegated to a
    provided PostProcessor instance.
    """

    def __init__(self, onnx_model_path: str, normalize: bool = True):
        self.onnx_model_path = onnx_model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[list] = None  # [batch_size, channels, height, width]
        self.input_type: Optional[str] = None
        self.output_name: Optional[str] = None
        self.normalize = normalize

        self._load_model()

    def _load_model(self):
        """Loads the ONNX model using onnxruntime."""
        options = ort.SessionOptions()
        options.enable_profiling = True
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            providers=ort.get_available_providers(),
            sess_options=options,
        )

        # Get input/output names and shapes for onnxruntime session
        input_meta = self.session.get_inputs()[0]
        output_meta = self.session.get_outputs()[0]

        self.input_name = input_meta.name
        self.input_shape = input_meta.shape  # e.g., [1, 3, 1280, 1280]
        self.input_type = input_meta.type
        self.output_name = output_meta.name

        if self.input_shape[1] > self.input_shape[3]:
            self.tensor_format = "nhwc"
        else:
            self.tensor_format = "nchw"

        print(f"ONNX Model loaded via onnxruntime: {self.onnx_model_path}")
        print(
            f"Input Name: {self.input_name}, Input Shape: {self.input_shape}, Tensor format: {self.tensor_format}, Dtype: {self.input_type}"
        )
        print(f"Output Name: {self.output_name}")

    def preprocess_image(
        self, image_path: str
    ) -> Tuple[
        np.ndarray, Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]
    ]:
        """
        Preprocesses a single image to the format expected by the ONNX model for onnxruntime.
        Implements ultralytics LetterBox functionality to maintain aspect ratio and pad.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[np.ndarray, Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
                - preprocessed_image_bchw: The preprocessed image as a NumPy array (BCHW).
                - original_dims: Original (width, height) of the image.
                - ratio_pad: ((gain, gain), (pad_x, pad_y)) for scaling boxes back.
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        original_dims = (
            original_image.shape[1],
            original_image.shape[0],
        )  # (width, height)

        # 1. LetterBox Resizing and Padding
        img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        shape = img_rgb.shape[:2]  # current shape [height, width]

        if self.tensor_format == "nchw":
            new_shape = (
                self.input_shape[2],
                self.input_shape[3],
            )  # target shape [height, width]
        else:
            new_shape = (self.input_shape[1], self.input_shape[2])

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # Allow scaling up (default in ultralytics for LetterBox is scaleup=True)
        # r = min(r, 1.0) # only scale down (original is commented out, so we follow ultralytics default)

        # Compute padding
        new_unpad = (
            int(round(shape[1] * r)),
            int(round(shape[0] * r)),
        )  # new_unpad width, height
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        # Auto padding (ultralytics uses this if auto=True, which is not default for LetterBox directly)
        # For simplicity, assuming auto=False and center padding for now.
        # if auto: # minimum rectangle
        #     dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride) # wh padding

        # Center padding (default in ultralytics LetterBox)
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize if needed
            img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img_rgb

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # padding_value = 114 (default in ultralytics LetterBox)
        padded_image = cv2.copyMakeBorder(
            img_resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Store ratio_pad for post-processing bounding boxes
        gain = r
        pad_x, pad_y = left, top  # These are the actual left and top padding amounts
        ratio_pad = ((gain, gain), (pad_x, pad_y))

        # 2. Normalize pixel values to [0, 1]
        if self.normalize:
            preprocessed_image = padded_image.astype(np.float32) / 255.0
        else:
            if self.input_type and "uint8" in self.input_type:
                preprocessed_image = padded_image
            else:
                preprocessed_image = padded_image.astype(np.float32)

        # 3. Transpose to BCHW
        if self.tensor_format == "nchw":
            preprocessed_image = np.transpose(
                preprocessed_image, (2, 0, 1)
            )  # HWC to CHW

        preprocessed_image = np.expand_dims(
            preprocessed_image, axis=0
        )  # Add batch dimension

        return preprocessed_image, original_dims, ratio_pad

    def run_inference(
        self,
        image_path: str,
        post_processor: Any,  # post processor
        confidence_threshold: float,
        iou_threshold: float,
        class_names: Dict[int, str],
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
        preprocessed_image, original_dims, ratio_pad = self.preprocess_image(image_path)

        # Run inference with onnxruntime
        raw_output = self.session.run(
            [self.output_name], {self.input_name: preprocessed_image}
        )[0]

        # Delegate post-processing to the provided post_processor
        detections = post_processor.postprocess(
            raw_output=raw_output,
            original_dims=original_dims,
            input_shape=self.input_shape,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            class_names=class_names,
            ratio_pad=ratio_pad,  # Pass ratio_pad for correct bbox scaling
        )

        return detections

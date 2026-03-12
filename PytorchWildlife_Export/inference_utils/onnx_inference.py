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


def preprocess_image(
    image_path: str,
    input_shape: list,
    tensor_format: str = "nchw",
    normalize: bool = True,
    uint8_input: bool = False,
) -> Tuple[
    np.ndarray, Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]
]:
    """
    Preprocess a single image into the tensor format expected by the model.

    Performs letterbox resize + pad, then converts to the requested dtype/layout.

    Args:
        image_path: Path to the input image.
        input_shape: Model input shape.
            NCHW → [batch, C, H, W]; NHWC → [batch, H, W, C].
        tensor_format: "nchw" or "nhwc".
        normalize: Divide pixel values by 255 to produce float32 in [0, 1].
            Ignored when uint8_input=True.
        uint8_input: Return raw uint8 pixels without any normalization or cast.

    Returns:
        preprocessed: np.ndarray with a batch dimension prepended.
        original_dims: (width, height) of the source image.
        ratio_pad: ((gain, gain), (pad_left, pad_top)) for box rescaling.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_dims = (original_image.shape[1], original_image.shape[0])  # (w, h)
    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    shape = img_rgb.shape[:2]  # (h, w)
    if tensor_format == "nchw":
        new_shape = (input_shape[2], input_shape[3])  # (H, W)
    else:
        new_shape = (input_shape[1], input_shape[2])  # (H, W)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    img_resized = cv2.resize(img_rgb, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        img_resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    ratio_pad = ((r, r), (left, top))

    if uint8_input:
        img_out = padded  # uint8
    elif normalize:
        img_out = padded.astype(np.float32) / 255.0
    else:
        img_out = padded.astype(np.float32)  # denormalized float, range 0-255

    if tensor_format == "nchw":
        img_out = np.transpose(img_out, (2, 0, 1))  # HWC → CHW

    img_out = np.expand_dims(img_out, 0)  # add batch dim
    return img_out, original_dims, ratio_pad


class ONNXInferenceSession:
    """
    Manages ONNX model loading, inference execution, and post-processing for object detection.
    This class always uses onnxruntime for raw inference. Post-processing is delegated to a
    provided PostProcessor instance.
    """

    def __init__(
        self,
        onnx_model_path: str,
        normalize: bool = True,
        preferred_provider: Optional[str] = None,
        provider_options: Optional[Dict[str, Dict]] = None,
    ):
        self.onnx_model_path = onnx_model_path
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[list] = None  # [batch_size, channels, height, width]
        self.input_type: Optional[str] = None
        self.output_name: Optional[str] = None
        self.normalize = normalize
        self.preferred_provider = preferred_provider
        self.provider_options = provider_options or {}

        print("b4 load_model")
        self._load_model()
        print("after load_model")

    def _load_model(self):
        """Loads the ONNX model using onnxruntime."""
        available = ort.get_available_providers()
        if self.preferred_provider is not None and self.preferred_provider in available:
            idx = available.index(self.preferred_provider)
            provider_names = available[idx:]
        else:
            provider_names = available

        # Build provider list: wrap with options dict where provided
        providers = [
            (name, self.provider_options[name]) if name in self.provider_options else name
            for name in provider_names
        ]
        print(f"ORT providers: {provider_names}")

        options = ort.SessionOptions()
        options.enable_profiling = True
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            providers=providers,
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
        """Delegates to the module-level preprocess_image utility."""
        uint8 = bool(self.input_type and "uint8" in self.input_type)
        return preprocess_image(
            image_path,
            self.input_shape,
            tensor_format=self.tensor_format,
            normalize=self.normalize,
            uint8_input=uint8,
        )

    def benchmark(
        self,
        image_path: str,
        warmup: int = 20,
        iterations: int = 100,
    ) -> dict:
        """
        Run warmup + timed inference iterations and flush the ORT profiling file.

        ORT profiling cannot be paused mid-session, so warmup events are recorded
        alongside measurement events. The returned metadata (warmup_runs, total_runs)
        lets the profile_analysis script skip warmup events by timestamp order.

        Args:
            image_path: Path to a representative input image (preprocessed once).
            warmup: Number of warmup iterations (hardware stabilisation + JIT caching).
            iterations: Number of timed measurement iterations.

        Returns:
            dict with keys:
                profile_path  – absolute path to the ORT JSON profiling file
                warmup_runs   – number of warmup runs recorded before measurement
                total_runs    – warmup + iterations (total runs in the profile file)
                latencies_ms  – list of per-iteration wall-clock latencies (measurement only)
                mean_ms       – mean latency over measurement iterations
                p50_ms        – median latency
                p99_ms        – 99th-percentile latency
        """
        import time

        preprocessed, _, _ = self.preprocess_image(image_path)

        print(f"Benchmarking: {warmup} warmup + {iterations} measurement runs...")

        # Warmup — profiling is live but these events will be skipped in analysis
        for i in range(warmup):
            self.session.run([self.output_name], {self.input_name: preprocessed})
        print(f"  Warmup complete ({warmup} runs).")

        # Measurement
        latencies_ms = []
        for i in range(iterations):
            t0 = time.perf_counter()
            self.session.run([self.output_name], {self.input_name: preprocessed})
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{iterations}] avg so far: {sum(latencies_ms)/len(latencies_ms):.2f} ms")

        profile_path = self.session.end_profiling()
        print(f"  Profile saved: {profile_path}")

        latencies_ms.sort()
        n = len(latencies_ms)
        return {
            "profile_path": profile_path,
            "warmup_runs": warmup,
            "total_runs": warmup + iterations,
            "latencies_ms": latencies_ms,
            "mean_ms": sum(latencies_ms) / n,
            "p50_ms": latencies_ms[n // 2],
            "p99_ms": latencies_ms[int(n * 0.99)],
        }

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

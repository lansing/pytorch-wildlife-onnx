import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import os
import math
from typing import List, Tuple, Dict, Union

# Try to import ultralytics, but don't fail if not present (e.g., for non-YOLO ONNX models)
try:
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
except ImportError:
    YOLO = None
    Results = None
    print("Ultralytics not found. YOLO-specific post-processing will not be available.")


class ONNXInferenceSession:
    """
    Manages ONNX model loading, inference execution, and post-processing for object detection.
    Supports both generic ONNX models via onnxruntime and YOLO models via ultralytics's predict method.
    """

    def __init__(self, onnx_model_path: str, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        self.onnx_model_path = onnx_model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        # Class names for MegaDetectorV6 (animal, person, vehicle)
        self.class_names = {0: "animal", 1: "person", 2: "vehicle"} 

        self.session: Union[ort.InferenceSession, YOLO, None] = None
        self.is_yolo_model = False
        
        self._load_model()

    def _load_model(self):
        """Loads the ONNX model. If it's a YOLO model, loads via ultralytics, otherwise via onnxruntime."""
        if YOLO and "yolov" in self.onnx_model_path.lower() or "mdv6" in self.onnx_model_path.lower():
            try:
                self.session = YOLO(self.onnx_model_path)
                self.is_yolo_model = True
                print(f"ONNX YOLO Model loaded via ultralytics: {self.onnx_model_path}")
                # Extract input shape from ultralytics model if possible
                # The input shape here refers to the expected image size (height, width)
                # ultralytics typically expects square images, so input_shape[2] for both h and w
                # We can try to infer it from the model.metadata or a dummy predict call
                self.input_shape = [1, 3, 1280, 1280] # Default for MDV6-yolov9-c, will be updated if possible
                print(f"Inferred Input Shape (YOLO): {self.input_shape}")
                return
            except Exception as e:
                print(f"Warning: Could not load YOLO model with ultralytics. Falling back to onnxruntime. Error: {e}")
                self.is_yolo_model = False

        self.session = ort.InferenceSession(self.onnx_model_path, providers=ort.get_available_providers())
        self.is_yolo_model = False
        
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

    def postprocess_output_onnxruntime(self, output: np.ndarray, original_dims: Tuple[int, int]) -> List[Dict]:
        """
        Post-processes the raw ONNX model output (from onnxruntime) to extract detection results.
        Expected output format: [1, num_attributes, num_predictions] where num_attributes = 7 (x,y,w,h,obj_conf,class_0_score,class_1_score)

        Args:
            output (np.ndarray): Raw output from the ONNX model (e.g., [1, 7, 33600]).
            original_dims (Tuple[int, int]): Original (width, height) of the image.

        Returns:
            List[Dict]: A list of dictionaries, each representing a detected object.
                        Each dict contains 'box' (xyxy), 'confidence', 'class_id', 'class_name'.
        """
        # Transpose the output to (batch_size, num_predictions, num_attributes)
        # e.g., from [1, 7, 33600] to [1, 33600, 7]
        output = output.transpose(0, 2, 1) # Assuming batch size is 1

        predictions = output[0] # Get predictions for the first image in batch (shape: [33600, 7])

        # Extract components: bbox, objectness, class probabilities
        boxes_raw = predictions[:, :4] # x, y, w, h
        objectness_scores = predictions[:, 4] # objectness score
        class_probs = predictions[:, 5:] # class probabilities (shape: [num_predictions, num_classes_output])

        # Filter by objectness confidence threshold first
        conf_mask = objectness_scores > self.confidence_threshold
        predictions = predictions[conf_mask]
        boxes_raw = boxes_raw[conf_mask]
        objectness_scores = objectness_scores[conf_mask]
        class_probs = class_probs[conf_mask]
        
        if len(predictions) == 0:
            return []

        # Combine objectness confidence with class confidence
        # For each prediction, get the max class score and its index
        max_class_scores = np.max(class_probs, axis=1)
        max_class_ids = np.argmax(class_probs, axis=1)
        
        # Final confidence is objectness score * max_class_score
        final_scores = objectness_scores * max_class_scores

        # Apply a combined confidence threshold (objectness * class_score)
        keep_indices_final = final_scores > self.confidence_threshold
        boxes_raw = boxes_raw[keep_indices_final]
        final_scores = final_scores[keep_indices_final]
        class_ids = max_class_ids[keep_indices_final]
        
        if len(boxes_raw) == 0:
            return []

        # Convert bounding boxes from (center_x, center_y, width, height) to (x1, y1, x2, y2)
        # Coordinates are normalized (0-1) and relative to the input_shape (e.g., 1280x1280)
        
        # Denormalize to input_shape dimensions first
        input_h, input_w = self.input_shape[2], self.input_shape[3]
        
        # x_center, y_center, width, height are relative to input_w, input_h
        boxes_xywh = boxes_raw
        boxes_xyxy = np.copy(boxes_xywh) 

        boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 # x1
        boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 # y1
        boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 # x2
        boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 # y2
        
        # Scale to original image dimensions
        scale_w = original_dims[0] / input_w
        scale_h = original_dims[1] / input_h

        final_boxes_scaled = np.copy(boxes_xyxy)
        final_boxes_scaled[:, 0] = np.clip(final_boxes_scaled[:, 0] * scale_w, 0, original_dims[0])
        final_boxes_scaled[:, 1] = np.clip(final_boxes_scaled[:, 1] * scale_h, 0, original_dims[1])
        final_boxes_scaled[:, 2] = np.clip(final_boxes_scaled[:, 2] * scale_w, 0, original_dims[0])
        final_boxes_scaled[:, 3] = np.clip(final_boxes_scaled[:, 3] * scale_h, 0, original_dims[1])

        # Apply Non-Maximum Suppression (NMS)
        # NMSBoxes expects boxes as [x, y, w, h] or [x1, y1, x2, y2]
        # Our final_boxes_scaled are [x1, y1, x2, y2]
        indices = cv2.dnn.NMSBoxes(
            bboxes=final_boxes_scaled.tolist(), 
            scores=final_scores.tolist(), 
            score_threshold=self.confidence_threshold, # NMS uses score_threshold to filter boxes
            nms_threshold=self.iou_threshold
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
                    "class_name": self.class_names.get(int(class_id), "unknown")
                })
        return detections

    def run_inference(self, image_path: str) -> List[Dict]:
        """
        Runs inference on a single image and returns detection results.
        If the model is a YOLO model and ultralytics is available, it uses ultralytics's predict method.
        Otherwise, it uses onnxruntime directly.
        """
        if self.is_yolo_model and isinstance(self.session, YOLO):
            # Use ultralytics's predict method for YOLO models
            print("Using ultralytics.YOLO.predict for inference and post-processing.")
            results_list: List[Results] = self.session.predict(
                source=image_path,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_shape[2], # Assuming square input
                verbose=False
            )
            
            detections = []
            if results_list:
                for res in results_list: # Should be one result for one image
                    for box_data in res.boxes:
                        x1, y1, x2, y2 = [int(x) for x in box_data.xyxy[0].tolist()]
                        confidence = float(box_data.conf[0])
                        class_id = int(box_data.cls[0])
                        detections.append({
                            "box": [x1, y1, x2, y2],
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": self.class_names.get(class_id, "unknown")
                        })
            return detections
        else:
            # Fallback to onnxruntime direct inference
            print("Using onnxruntime direct inference and custom post-processing.")
            preprocessed_image, original_dims = self.preprocess_image(image_path)
            
            # Run inference
            raw_output = self.session.run([self.output_name], {self.input_name: preprocessed_image})[0]
            
            detections = self.postprocess_output_onnxruntime(raw_output, original_dims)
            
            return detections

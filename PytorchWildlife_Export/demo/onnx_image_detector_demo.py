import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Using PIL for drawing bounding boxes
from typing import List, Dict

# Add the project's new top-level directory to the Python path
# This assumes the script is run from the main project root.
# If running directly from PytorchWildlife_Export/demo/, need to go up two levels.
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

from PytorchWildlife_Export.inference_utils.onnx_inference import ONNXInferenceSession
from PytorchWildlife_Export.postprocessors.yolov_postprocessor import YOLOvPostProcessor
from PytorchWildlife_Export.postprocessors.ultralytics_baseline_utils import get_ultralytics_baseline_detections

# --- Configuration ---
ONNX_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'exported_models_test', 'MDV6-yolov9-c_1280x1280_raw.onnx'
))
SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'sample_image.jpg'
))
OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'demo_output'
))
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMAGE_CUSTOM_PATH = os.path.join(OUTPUT_DIR, 'detected_sample_image_custom_pp.jpg')
OUTPUT_IMAGE_BASELINE_PATH = os.path.join(OUTPUT_DIR, 'detected_sample_image_ultralytics_baseline.jpg')


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Hardcoded for MegaDetectorV6, adjust if needed
# Use the full 3-class mapping for both custom and baseline for consistency
GLOBAL_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}


# --- Helper for Visualization ---
def visualize_detections(image_path: str, detections: List[Dict], output_path: str, class_names: Dict[int, str]):
    if not os.path.exists(image_path):
        print(f"Original image not found for visualization: {image_path}")
        return

    img_pil = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = f"{class_names.get(det['class_id'], 'unknown')}: {det['confidence']:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
    
    img_pil.save(output_path)
    print(f"Annotated image saved to: {output_path}")


# --- Main Demo Logic ---
def run_demo():
    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    inference_session = ONNXInferenceSession(onnx_model_path=ONNX_MODEL_PATH)
    
    # Custom Post-Processor
    custom_post_processor = YOLOvPostProcessor()
    
    print("\n--- Running Inference with Custom Post-Processor ---")
    custom_detections = inference_session.run_inference(
        image_path=SAMPLE_IMAGE_PATH,
        post_processor=custom_post_processor,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_names=GLOBAL_CLASS_NAMES
    )

    print(f"Custom Post-Processor found {len(custom_detections)} detections.")
    for det in custom_detections:
        print(f"  Custom - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}")
    visualize_detections(SAMPLE_IMAGE_PATH, custom_detections, OUTPUT_IMAGE_CUSTOM_PATH, GLOBAL_CLASS_NAMES)

    # Ultralytics Baseline Post-Processor
    print("\n--- Running Inference with Ultralytics Baseline ---")
    ultralytics_detections = get_ultralytics_baseline_detections(
        onnx_model_path=ONNX_MODEL_PATH,
        image_path=SAMPLE_IMAGE_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        input_img_size=inference_session.input_shape[2], # Use input size from ONNXInferenceSession
        class_names=GLOBAL_CLASS_NAMES
    )

    print(f"Ultralytics Baseline found {len(ultralytics_detections)} detections.")
    for det in ultralytics_detections:
        print(f"  Baseline - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}")
    
    # print(f"DEBUG: Ultralytics Baseline raw output (first detection): {ultralytics_detections[0] if ultralytics_detections else 'N/A'}")

    visualize_detections(SAMPLE_IMAGE_PATH, ultralytics_detections, OUTPUT_IMAGE_BASELINE_PATH, GLOBAL_CLASS_NAMES)

    # Optional: Compare results
    print("\n--- Comparison ---")
    print(f"Custom Post-Processor Detections: {len(custom_detections)}")
    print(f"Ultralytics Baseline Detections: {len(ultralytics_detections)}")
    # Add more sophisticated comparison logic if needed (e.g., IoU matching, etc.)

if __name__ == "__main__":
    run_demo()

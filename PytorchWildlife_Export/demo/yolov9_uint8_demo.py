import os
import sys
import cv2
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict

# Add the project's top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

# Import necessary components
from PytorchWildlife_Export.inference_utils.onnx_inference import ONNXInferenceSession
from PytorchWildlife_Export.postprocessors.yolov_postprocessor import YOLOvPostProcessor # For YOLOv9 output
from PytorchWildlife_Export.postprocessors.ultralytics_baseline_utils import get_ultralytics_baseline_detections
from PytorchWildlife_Export.export_tool import main as export_tool_main, parse_args as export_parse_args

# --- Configuration ---
SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'sample_image.jpg'
))
OUTPUT_DIR = "/exported_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
GLOBAL_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

# Model parameters for YOLOv9 UINT8 export
YOLOV9_MODEL_VERSION = "MDV6-yolov9-e"
YOLOV9_INPUT_IMG_SIZE = 640 # Let's use 640x640 for this demo
YOLOV9_UINT8_ONNX_PATH = os.path.join(OUTPUT_DIR, f"{YOLOV9_MODEL_VERSION}_{YOLOV9_INPUT_IMG_SIZE}x{YOLOV9_INPUT_IMG_SIZE}_uint8.onnx")
YOLOV9_FLOAT32_ONNX_PATH = os.path.join(OUTPUT_DIR, f"{YOLOV9_MODEL_VERSION}_{YOLOV9_INPUT_IMG_SIZE}x{YOLOV9_INPUT_IMG_SIZE}_float32.onnx")


# --- Helper for Visualization ---
def visualize_detections(image_path: str, detections: List[Dict], output_path: str, class_names: Dict[int, str], title: str = ""):
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
    
    if title:
        title_bbox = draw.textbbox((0,0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text((img_pil.width - title_width - 10, 10), title, fill="yellow", font=font)

    img_pil.save(output_path)
    print(f"Annotated image saved to: {output_path}")


# --- Main Demo Logic ---
def run_demo():
    print("\n--- Step 1: Export YOLOv9 Model in UINT8 ---")
    
    # Export the YOLOv9 model in UINT8 format
    export_tool_main(export_parse_args([
        "--model_type", "yolov9",
        "--model_version", YOLOV9_MODEL_VERSION,
        "--output_path", YOLOV9_UINT8_ONNX_PATH,
        "--format", "uint8",
        "--opset", "18",
        "--simplify",
        "--input_img_size", str(YOLOV9_INPUT_IMG_SIZE),
    ]))

    print("\n--- Step 2: Run Inference on the YOLOv9 UINT8 Model ---")
    inference_session_uint8 = ONNXInferenceSession(onnx_model_path=YOLOV9_UINT8_ONNX_PATH)
    custom_post_processor = YOLOvPostProcessor() # Use YOLOv9 post-processor

    custom_detections_uint8 = inference_session_uint8.run_inference(
        image_path=SAMPLE_IMAGE_PATH,
        post_processor=custom_post_processor,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_names=GLOBAL_CLASS_NAMES
    )

    output_image_custom_uint8_path = os.path.join(OUTPUT_DIR, f'detected_sample_image_{YOLOV9_MODEL_VERSION}_{YOLOV9_INPUT_IMG_SIZE}x{YOLOV9_INPUT_IMG_SIZE}_uint8_custom_pp.jpg')
    print(f"Custom Post-Processor (UINT8) found {len(custom_detections_uint8)} detections.")
    for det in custom_detections_uint8:
        print(f"  Custom UINT8 - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}")
    visualize_detections(SAMPLE_IMAGE_PATH, custom_detections_uint8, output_image_custom_uint8_path, GLOBAL_CLASS_NAMES, title=f"YOLOv9 UINT8 Custom PP")

    print("\n--- Step 3: Export Original YOLOv9 FLOAT32 Model for Comparison (if not exists) ---")
    # Export the original YOLOv9 raw float32 model for baseline if it doesn't exist
    if not os.path.exists(YOLOV9_FLOAT32_ONNX_PATH):
        print(f"Exporting original YOLOv9 raw FLOAT32 model to: {YOLOV9_FLOAT32_ONNX_PATH}")
        export_tool_main(export_parse_args([
            "--model_type", "yolov9",
            "--model_version", YOLOV9_MODEL_VERSION,
            "--output_path", YOLOV9_FLOAT32_ONNX_PATH,
            "--format", "float32",
            "--opset", "18",
            "--simplify",
            "--input_img_size", str(YOLOV9_INPUT_IMG_SIZE),
        ]))
    else:
        print(f"Using existing original YOLOv9 raw FLOAT32 model: {YOLOV9_FLOAT32_ONNX_PATH}")

    print("\n--- Step 4: Run Ultralytics Baseline on Original YOLOv9 FLOAT32 Model ---")
    ultralytics_detections_float32 = get_ultralytics_baseline_detections(
        onnx_model_path=YOLOV9_FLOAT32_ONNX_PATH,
        image_path=SAMPLE_IMAGE_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        input_img_size=YOLOV9_INPUT_IMG_SIZE, # Match input size
        class_names=GLOBAL_CLASS_NAMES
    )

    output_image_baseline_float32_path = os.path.join(OUTPUT_DIR, f'detected_sample_image_{YOLOV9_MODEL_VERSION}_{YOLOV9_INPUT_IMG_SIZE}x{YOLOV9_INPUT_IMG_SIZE}_float32_ultralytics_baseline.jpg')
    print(f"Ultralytics Baseline (Float32 Original) found {len(ultralytics_detections_float32)} detections.")
    for det in ultralytics_detections_float32:
        print(f"  Baseline Float32 - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}")
    visualize_detections(SAMPLE_IMAGE_PATH, ultralytics_detections_float32, output_image_baseline_float32_path, GLOBAL_CLASS_NAMES, title=f"UL Baseline: {YOLOV9_MODEL_VERSION} Float32")

    print("\n--- Comparison ---")
    print(f"Custom PP (YOLOv9 UINT8) Detections: {len(custom_detections_uint8)}")
    print(f"Ultralytics Baseline (YOLOv9 Float32) Detections: {len(ultralytics_detections_float32)}")


if __name__ == "__main__":
    run_demo()

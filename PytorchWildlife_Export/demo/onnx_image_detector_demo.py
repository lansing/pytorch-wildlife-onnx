import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Using PIL for drawing bounding boxes

# Add the project's new top-level directory to the Python path
# This assumes the script is run from the main project root.
# If running directly from PytorchWildlife_Export/demo/, need to go up two levels.
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

# Correctly import from our new package
from PytorchWildlife_Export.inference_utils.onnx_inference import ONNXInferenceSession

# --- Configuration ---
ONNX_MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', 'exported_models_test', 'MDV6-yolov9-c_1280x1280.onnx'
))
SAMPLE_IMAGE_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'sample_image.jpg'
))
OUTPUT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'demo_output'
))
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, 'detected_sample_image.jpg')

CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.45

# --- Main Demo Logic ---
def run_demo():
    print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
    inference_session = ONNXInferenceSession(
        onnx_model_path=ONNX_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )

    print(f"Running inference on image: {SAMPLE_IMAGE_PATH}")
    detections = inference_session.run_inference(SAMPLE_IMAGE_PATH)

    print(f"Found {len(detections)} detections.")
    for det in detections:
        print(f"  Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}")

    # --- Visualize Detections ---
    if os.path.exists(SAMPLE_IMAGE_PATH):
        img_pil = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
        draw = ImageDraw.Draw(img_pil)

        # Try to load a font, fall back to default if not found
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()
        
        for det in detections:
            x1, y1, x2, y2 = det['box']
            label = f"{det['class_name']}: {det['confidence']:.2f}"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            
            # Draw label background
            # text_width, text_height = draw.textsize(label, font=font) # Deprecated
            # Use textbbox for newer Pillow versions
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
            draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
        
        img_pil.save(OUTPUT_IMAGE_PATH)
        print(f"Detected image saved to: {OUTPUT_IMAGE_PATH}")
    else:
        print(f"Original image not found for visualization: {SAMPLE_IMAGE_PATH}")

if __name__ == "__main__":
    run_demo()
import argparse
import os
import sys
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # Using PIL for drawing bounding boxes

# Add the project's new top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

from PytorchWildlife_Export.inference_utils.onnx_inference import ONNXInferenceSession
from PytorchWildlife_Export.postprocessors.ultralytics_baseline_utils import (
    get_ultralytics_baseline_detections,
)
from PytorchWildlife_Export.postprocessors.yolov10_postprocessor import (
    YOLOv10PostProcessor,  # For YOLOv10
)
from PytorchWildlife_Export.postprocessors.yolov_postprocessor import (
    YOLOvPostProcessor,  # For YOLOv9
)

# --- Configuration ---
SAMPLE_IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "sample_image.jpg")
)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Hardcoded for MegaDetectorV6, adjust if needed
GLOBAL_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}


# --- Helper for Visualization ---
def visualize_detections(
    image_path: str,
    detections: List[Dict],
    output_path: str,
    class_names: Dict[int, str],
    title: str = "",
):
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
        x1, y1, x2, y2 = det["box"]
        label = (
            f"{class_names.get(det['class_id'], 'unknown')}: {det['confidence']:.2f}"
        )

        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)

    # Add title if provided
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(
            (img_pil.width - title_width - 10, 10), title, fill="yellow", font=font
        )

    img_pil.save(output_path)
    print(f"Annotated image saved to: {output_path}")


# --- Main Demo Logic ---
def run_demo(model_paths: List[str]):

    for model_path in model_paths:
        model_filename = os.path.basename(model_path)
        print(f"\n--- Running Inference for model: {model_filename} ---")

        print(f"Loading ONNX model from: {model_path}")
        inference_session = ONNXInferenceSession(
            onnx_model_path=model_path, normalize=True
        )

        # Determine which post-processor to use based on model filename
        if "yolov10" in model_filename.lower():
            if "v9_compatible" in model_filename.lower():
                custom_post_processor = YOLOvPostProcessor()
            else:
                custom_post_processor = YOLOv10PostProcessor()
        else:  # Default to YOLOv9 post-processor for other ultralytics models
            custom_post_processor = YOLOvPostProcessor()

        # --- Run with Custom Post-Processor ---
        output_image_custom_path = os.path.join(
            OUTPUT_DIR,
            f"detected_sample_image_{os.path.splitext(model_filename)[0]}_custom_pp.jpg",
        )

        print("\n--- Running Inference with Custom Post-Processor ---")
        custom_detections = inference_session.run_inference(
            image_path=SAMPLE_IMAGE_PATH,
            post_processor=custom_post_processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            class_names=GLOBAL_CLASS_NAMES,
        )

        print(f"Custom Post-Processor found {len(custom_detections)} detections.")
        for det in custom_detections:
            print(
                f"  Custom - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}"
            )
        visualize_detections(
            SAMPLE_IMAGE_PATH,
            custom_detections,
            output_image_custom_path,
            GLOBAL_CLASS_NAMES,
            title=f"Custom PP: {model_filename}",
        )

        # --- Run with Ultralytics Baseline (only for models originally exported by ultralytics) ---
        # The sample model might not be exportable by ultralytics.YOLO directly for baseline comparison
        # We assume for now that the baseline works for our newly exported raw model.
        if (
            "raw" in model_filename or "nms" in model_filename
        ):  # Heuristic to check if it's our exported ultralytics model
            output_image_baseline_path = os.path.join(
                OUTPUT_DIR,
                f"detected_sample_image_{os.path.splitext(model_filename)[0]}_ultralytics_baseline.jpg",
            )

            print("\n--- Running Inference with Ultralytics Baseline ---")
            ultralytics_detections = get_ultralytics_baseline_detections(
                onnx_model_path=model_path,
                image_path=SAMPLE_IMAGE_PATH,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                iou_threshold=IOU_THRESHOLD,
                input_img_size=inference_session.input_shape[
                    2
                ],  # Use input size from ONNXInferenceSession
                class_names=GLOBAL_CLASS_NAMES,
            )

            print(
                f"Ultralytics Baseline found {len(ultralytics_detections)} detections."
            )
            for det in ultralytics_detections:
                print(
                    f"  Baseline - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}"
                )

            visualize_detections(
                SAMPLE_IMAGE_PATH,
                ultralytics_detections,
                output_image_baseline_path,
                GLOBAL_CLASS_NAMES,
                title=f"UL Baseline: {model_filename}",
            )

            # --- Comparison ---
            print("\n--- Comparison ---")
            print(f"Custom Post-Processor Detections: {len(custom_detections)}")
            print(f"Ultralytics Baseline Detections: {len(ultralytics_detections)}")
            # Add more sophisticated comparison logic if needed (e.g., IoU matching, etc.)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ONNX image detector demo for one or more models."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to a specific ONNX model to run the demo on. If not provided, default models will be used.",
    )
    args = parser.parse_args()

    if args.model_path:
        if not os.path.exists(args.model_path):
            print(f"Error: Model not found at {args.model_path}")
            sys.exit(1)
        models_to_test = [args.model_path]
    else:
        # Define default models to test
        DEFAULT_MODELS_DIR = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        models_to_test = [
            os.path.join(
                DEFAULT_MODELS_DIR, "sample_models", "MDV6-yolov9-c-320-16b.onnx"
            ),  # Model 1: Provided sample
            os.path.join(
                DEFAULT_MODELS_DIR,
                "exported_models_test",
                "MDV6-yolov9-c_1280x1280_raw.onnx",
            ),  # Model 2: Newly exported raw YOLOv9
            os.path.join(
                DEFAULT_MODELS_DIR,
                "exported_models_test",
                "MDV6-yolov10-e_1280x1280_raw.onnx",
            ),  # Model 3: Newly exported raw YOLOv10
        ]

        # Ensure all exported models exist
        for model_path in models_to_test[1:]:  # Check exported models only
            if not os.path.exists(model_path):
                print(
                    f"Error: Exported model not found at {model_path}. Please run 'make export' (if it's the raw YOLOv9 model) and specific export commands for YOLOv10."
                )
                sys.exit(1)

    run_demo(models_to_test)

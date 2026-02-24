import argparse
import os
import sys
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add the project's top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

# Import necessary components
from PytorchWildlife_Export.export_tool import (
    main as export_tool_main,  # Import the main function of export_tool
)
from PytorchWildlife_Export.inference_utils.onnx_inference import ONNXInferenceSession
from PytorchWildlife_Export.postprocessors.ultralytics_baseline_utils import (
    get_ultralytics_baseline_detections,
)
from PytorchWildlife_Export.postprocessors.yolov_postprocessor import YOLOvPostProcessor

# --- Configuration ---
SAMPLE_IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "sample_image.jpg")
)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "demo_output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
GLOBAL_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

# Model parameters for YOLOv10 compatible export
YOLOV10_COMPATIBLE_VERSION = "MDV6-yolov10-c"
YOLOV10_COMPATIBLE_ONNX_PATH = os.path.join(
    OUTPUT_DIR, f"{YOLOV10_COMPATIBLE_VERSION}_demo_export.engine"
)
# YOLOV10_COMPATIBLE_ONNX_PATH = os.path.join(
#     "exported_models/MDV6-yolov10-c_float16_320_v9_compat_denorm_nhwc.onnx"
# )


YOLOV10_ORIGINAL_ONNX_PATH = os.path.join(
    OUTPUT_DIR, f"{YOLOV10_COMPATIBLE_VERSION}_320_raw.onnx"
)


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

    if title:
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        draw.text(
            (img_pil.width - title_width - 10, 10), title, fill="yellow", font=font
        )

    img_pil.save(output_path)
    print(f"Annotated image saved to: {output_path}")


# --- Main Demo Logic ---
def run_demo():
    print("\n--- Step 1: Export YOLOv10 (v9 Compatible Output) Model ---")

    # Export the YOLOv10 model with v9 compatible output
    export_tool_args = [
        "export_tool.py",  # dummy arg for argparse
        "--model_type",
        "yolov10_v9_compatible",
        "--model_version",
        YOLOV10_COMPATIBLE_VERSION,
        "--output_path",
        YOLOV10_COMPATIBLE_ONNX_PATH,
        "--format",
        # "int8",
        "float32",
        # "float16",
        "--opset",
        "18",
        "--runtime",
        "tensorrt",
        "--simplify",
        "--input_img_size",
        "320",
        # "--nhwc_input",
        # "--denormalized_input",
        # "--uint8_input",
    ]
    sys.argv = export_tool_args  # Set sys.argv for argparse
    export_tool_main()  # Run the export tool

    # TODO temp disable inf (need to do it in trt)
    return

    print("\n--- Step 2: Run Inference on the YOLOv10 (v9 Compatible) Model ---")
    inference_session = ONNXInferenceSession(
        onnx_model_path=YOLOV10_COMPATIBLE_ONNX_PATH,
        # normalize=False,  # TODO for now, we are testing non-normalized float
    )
    custom_post_processor = YOLOvPostProcessor()  # Use YOLOv9 post-processor

    custom_detections = inference_session.run_inference(
        image_path=SAMPLE_IMAGE_PATH,
        post_processor=custom_post_processor,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_names=GLOBAL_CLASS_NAMES,
    )

    output_image_custom_path = os.path.join(
        OUTPUT_DIR,
        f"detected_sample_image_{YOLOV10_COMPATIBLE_VERSION}_v9_compatible_custom_pp.jpg",
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
        title=f"V10 (v9-comp) Custom PP",
    )

    return

    print(
        "\n--- Step 3: Run Ultralytics Baseline on Original YOLOv10 Model for Comparison ---"
    )
    # Export the original YOLOv10 raw model for baseline if it doesn't exist
    if not os.path.exists(YOLOV10_ORIGINAL_ONNX_PATH):
        print(f"Exporting original YOLOv10 raw model to: {YOLOV10_ORIGINAL_ONNX_PATH}")
        export_tool_args = [
            "export_tool.py",
            "--model_type",
            "yolov9",  # Use yolov9 model_type for original YOLOv10 export
            "--model_version",
            YOLOV10_COMPATIBLE_VERSION,
            "--output_path",
            YOLOV10_ORIGINAL_ONNX_PATH,
            "--format",
            "float32",
            "--opset",
            "18",
            "--simplify",
            "--input_img_size",
            "1280",
        ]
        sys.argv = export_tool_args
        export_tool_main()

    ultralytics_detections = get_ultralytics_baseline_detections(
        onnx_model_path=YOLOV10_ORIGINAL_ONNX_PATH,
        image_path=SAMPLE_IMAGE_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        input_img_size=1280,  # Hardcoded for now
        class_names=GLOBAL_CLASS_NAMES,
    )

    output_image_baseline_path = os.path.join(
        OUTPUT_DIR,
        f"detected_sample_image_{YOLOV10_COMPATIBLE_VERSION}_original_ultralytics_baseline.jpg",
    )
    print(f"Ultralytics Baseline found {len(ultralytics_detections)} detections.")
    for det in ultralytics_detections:
        print(
            f"  Baseline - Class: {det['class_name']} ({det['class_id']}), Confidence: {det['confidence']:.2f}, Box: {det['box']}"
        )
    visualize_detections(
        SAMPLE_IMAGE_PATH,
        ultralytics_detections,
        output_image_baseline_path,
        GLOBAL_CLASS_NAMES,
        title=f"UL Baseline: {YOLOV10_COMPATIBLE_VERSION} Original",
    )

    print("\n--- Comparison ---")
    print(f"Custom PP (V10 compatible) Detections: {len(custom_detections)}")
    print(
        f"Ultralytics Baseline (V10 Original) Detections: {len(ultralytics_detections)}"
    )


if __name__ == "__main__":
    run_demo()

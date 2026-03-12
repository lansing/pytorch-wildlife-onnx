import os
import statistics
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add the project's top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

# Import necessary components
from PytorchWildlife_Export.export_tool import (
    main as export_tool_main,
)
from PytorchWildlife_Export.export_tool import (
    parse_args as export_parse_args,
)
from PytorchWildlife_Export.inference_utils.onnx_inference import (
    ONNXInferenceSession,
    preprocess_image,
)
from PytorchWildlife_Export.model_exporters.quant import (
    calibrate_node_scales,
    wrap_conv_nodes_in_fp16,
    wrap_node_in_int8_qdq,
    wrap_nodes_in_int8_qdq,
)
from PytorchWildlife_Export.model_exporters.trt_calibration_dataset import (
    TRTCalibrationDataLoader,
)
from PytorchWildlife_Export.postprocessors.yolov10_postprocessor import (
    YOLOv10PostProcessor,
)

# --- Configuration ---
SAMPLE_IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "sample_image.jpg")
)
OUTPUT_DIR = "/exported_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
GLOBAL_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

# Model parameters for YOLOv10 compatible export
YOLOV10_COMPATIBLE_VERSION = "MDV6-yolov10-e"
YOLOV10_COMPATIBLE_ONNX_PATH = os.path.join(
    OUTPUT_DIR, f"{YOLOV10_COMPATIBLE_VERSION}_quant_demo.onnx"
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
    export_args = export_parse_args(
        [
            "--model_type",
            "yolov10",
            "--model_version",
            YOLOV10_COMPATIBLE_VERSION,
            "--output_path",
            YOLOV10_COMPATIBLE_ONNX_PATH,
            "--format",
            "float32",
            "--opset",
            "18",
            "--runtime",
            "onnx",
            "--simplify",
            "--input_img_size",
            "640",
            # "--nhwc_input",
            # "--denormalized_input",
            # "--uint8_input",
        ]
    )

    # IMPORTANT: run this before any ultralytics or tensorrt stuff if you want to use cuda. otherw is
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        print(f"CUDA context initialized on device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device found; ORT will run on CPU.")

    export_tool_main(export_args)

    print(
        "\n--- Step 1b: Calibrate + wrap Conv nodes in INT8 QDQ (exclude model.23) ---"
    )

    import onnx

    NUM_CALIB_IMAGES = 50

    base_model = onnx.load(YOLOV10_COMPATIBLE_ONNX_PATH)

    do_quant = True

    if do_quant:
        calib_loader = TRTCalibrationDataLoader(
            input_size=640,
            num_images=NUM_CALIB_IMAGES,
        )

        model = wrap_nodes_in_int8_qdq(
            base_model,
            calib_loader,
            node_types=["Conv"],
            exclude=[
                # Head nodes
                "model.23",
            ],
        )
        mixed_onnx_path = YOLOV10_COMPATIBLE_ONNX_PATH.replace(
            ".onnx", "_mixed_int8.onnx"
        )
    else:
        print("SKIPPING QUANTIZATION. Just saving original full precision model.")
        model = base_model
        mixed_onnx_path = YOLOV10_COMPATIBLE_ONNX_PATH

    onnx.save(model, mixed_onnx_path)
    print(f"Final model saved to: {mixed_onnx_path}")

    print("\n--- Step 2: Run Inference on the YOLOv10 (v9 Compatible) Model ---")

    custom_post_processor = YOLOv10PostProcessor()  # Use YOLOv10 post-processor
    inference_session = ONNXInferenceSession(
        onnx_model_path=mixed_onnx_path,
        preferred_provider="TensorrtExecutionProvider",
        provider_options={
            "TensorrtExecutionProvider": {
                # Explicit quantization: Q/DQ nodes carry calibrated scales.
                # trt_int8_enable=False keeps STRONGLY_TYPED mode so TRT reads
                # the embedded scale/zero-point rather than a calibration table.
                "trt_int8_enable": False,
                "trt_fp16_enable": False,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/exported_models/trt_cache",
                "trt_detailed_build_log": True,
            },
        },
    )
    # ── Correctness check (single inference + visualisation) ──────────────
    print("Entering run_inference")
    detections = inference_session.run_inference(
        image_path=SAMPLE_IMAGE_PATH,
        post_processor=custom_post_processor,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_names=GLOBAL_CLASS_NAMES,
    )

    print(f"inference found {len(detections)} detections.")
    for det in detections:
        print(
            f"  Class: {det['class_name']} ({det['class_id']}), "
            f"Confidence: {det['confidence']:.2f}, Box: {det['box']}"
        )

    output_image_path = os.path.join(
        OUTPUT_DIR,
        f"detected_sample_image_{YOLOV10_COMPATIBLE_VERSION}_trt.jpg",
    )
    visualize_detections(
        SAMPLE_IMAGE_PATH,
        detections,
        output_image_path,
        GLOBAL_CLASS_NAMES,
        title=f"TRT: {YOLOV10_COMPATIBLE_VERSION}",
    )

    # ── Benchmark ──────────────────────────────────────────────────────────
    print("\n--- Step 3: Benchmark ---")
    bench = inference_session.benchmark(
        image_path=SAMPLE_IMAGE_PATH,
        warmup=100,
        iterations=1000,
    )
    print(
        f"\nLatency over {len(bench['latencies_ms'])} iterations "
        f"(after {bench['warmup_runs']} warmup runs):"
    )
    print(f"  mean : {bench['mean_ms']:.2f} ms")
    print(f"  p50  : {bench['p50_ms']:.2f} ms")
    print(f"  p99  : {bench['p99_ms']:.2f} ms")
    print(f"  min  : {bench['latencies_ms'][0]:.2f} ms")
    print(f"  max  : {bench['latencies_ms'][-1]:.2f} ms")
    print(f"Profile saved to: {bench['profile_path']}")
    print(f"  Analyse with:")
    print(f"    python3 PytorchWildlife_Export/tools/profile_analysis.py \\")
    print(f"      {bench['profile_path']} \\")
    print(f"      --warmup-runs {bench['warmup_runs']} \\")
    print(f"      --total-runs {bench['total_runs']}")


if __name__ == "__main__":
    run_demo()

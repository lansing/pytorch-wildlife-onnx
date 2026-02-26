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
from PytorchWildlife_Export.export_tool import main as export_tool_main, parse_args as export_parse_args
from PytorchWildlife_Export.inference_utils.onnx_inference import preprocess_image
from PytorchWildlife_Export.postprocessors.yolov_postprocessor import YOLOvPostProcessor

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
    OUTPUT_DIR, f"{YOLOV10_COMPATIBLE_VERSION}_demo_export.engine"
)
# YOLOV10_COMPATIBLE_ONNX_PATH = os.path.join(
#     "exported_models/MDV6-yolov10-c_float16_320_v9_compat_denorm_nhwc.onnx"
# )


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
    export_args = export_parse_args([
        "--model_type", "yolov10_v9_compatible",
        "--model_version", YOLOV10_COMPATIBLE_VERSION,
        "--output_path", YOLOV10_COMPATIBLE_ONNX_PATH,
        "--format", "int8",
        # "--format", "float32",
        # "--format", "float16",
        "--opset", "18",
        "--runtime", "tensorrt",
        "--simplify",
        "--input_img_size", "640",
        # "--nhwc_input",
        # "--denormalized_input",
        # "--uint8_input",
    ])
    export_tool_main(export_args)

    print("\n--- Step 2: Inference Validation (TensorRT engine) ---")

    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(YOLOV10_COMPATIBLE_ONNX_PATH, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # Discover I/O tensor names and shapes from the engine
    input_name = None
    output_name = None
    input_shape = None
    output_shape = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            input_name = name
            input_shape = shape
        else:
            output_name = name
            output_shape = shape
    print(f"Engine input:  {input_name} {input_shape}")
    print(f"Engine output: {output_name} {output_shape}")

    # Preprocess sample image — derive format from the export args used above
    tensor_format = "nhwc" if export_args.nhwc_input else "nchw"
    normalize = not (export_args.uint8_input or export_args.denormalized_input)
    preprocessed, original_dims, ratio_pad = preprocess_image(
        SAMPLE_IMAGE_PATH,
        list(input_shape),
        tensor_format=tensor_format,
        normalize=normalize,
        uint8_input=export_args.uint8_input,
    )

    # Allocate GPU buffers
    input_gpu = torch.from_numpy(preprocessed).contiguous().cuda()
    output_gpu = torch.empty(output_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, input_gpu.data_ptr())
    context.set_tensor_address(output_name, output_gpu.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    raw_output = output_gpu.cpu().numpy()
    post_processor = YOLOvPostProcessor()
    detections = post_processor.postprocess(
        raw_output=raw_output,
        original_dims=original_dims,
        input_shape=list(input_shape),
        confidence_threshold=CONFIDENCE_THRESHOLD,
        iou_threshold=IOU_THRESHOLD,
        class_names=GLOBAL_CLASS_NAMES,
        ratio_pad=ratio_pad,
    )
    print(f"TRT inference found {len(detections)} detections.")
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

    print("\n--- Step 3: Latency Benchmark (TensorRT engine) ---")

    WARMUP_STEPS = 50
    TIMED_STEPS = 100

    input_cpu = torch.from_numpy(preprocessed).contiguous()
    latencies_ms = []

    for step in range(WARMUP_STEPS + TIMED_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_gpu.copy_(input_cpu)  # H2D
        context.execute_async_v3(stream_handle=stream)
        _ = output_gpu.cpu()  # D2H
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if step >= WARMUP_STEPS:
            latencies_ms.append((t1 - t0) * 1000.0)

    avg_ms = statistics.mean(latencies_ms)
    p50_ms = statistics.median(latencies_ms)
    p99_ms = sorted(latencies_ms)[int(len(latencies_ms) * 0.99) - 1]
    fps = 1000.0 / avg_ms

    print(f"Benchmark results over {TIMED_STEPS} timed steps (after {WARMUP_STEPS} warmup):")
    print(f"  Avg latency : {avg_ms:.2f} ms")
    print(f"  P50 latency : {p50_ms:.2f} ms")
    print(f"  P99 latency : {p99_ms:.2f} ms")
    print(f"  Throughput  : {fps:.1f} FPS")


if __name__ == "__main__":
    run_demo()

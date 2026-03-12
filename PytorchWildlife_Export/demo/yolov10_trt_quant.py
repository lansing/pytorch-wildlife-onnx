"""
YOLOv10 export → explicit INT8 quantization → native TRT engine → benchmark.

Pipeline:
  Step 1  : Export YOLOv10 to float32 ONNX (no pre/post-processing merged in).
  Step 1b : Calibrate + wrap Conv nodes in INT8 QDQ pairs (explicit quant).
  Step 2  : Build TRT engine from the quantized ONNX via onnx2engine_explicit.
  Step 3  : Validate inference with the TRT engine (correctness check).
  Step 4  : Latency benchmark — H2D + inference + D2H, 100 warmup + 1000 timed.
"""

import os
import statistics
import sys
import time
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

from PytorchWildlife_Export.export_tool import (
    main as export_tool_main,
    parse_args as export_parse_args,
)
from PytorchWildlife_Export.inference_utils.onnx_inference import preprocess_image
from PytorchWildlife_Export.model_exporters.quant import (
    wrap_nodes_in_int8_qdq,
)
from PytorchWildlife_Export.model_exporters.trt_export import onnx2engine_explicit
from PytorchWildlife_Export.model_exporters.trt_calibration_dataset import (
    TRTCalibrationDataLoader,
)
from PytorchWildlife_Export.postprocessors.yolov10_postprocessor import (
    YOLOv10PostProcessor,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_IMAGE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "sample_image.jpg")
)
OUTPUT_DIR = "/exported_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
GLOBAL_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}

MODEL_VERSION = "MDV6-yolov10-e"
ONNX_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_VERSION}_trt_quant.onnx")
QUANT_ONNX_PATH = ONNX_PATH.replace(".onnx", "_int8.onnx")
ENGINE_PATH = QUANT_ONNX_PATH.replace(".onnx", ".engine")

NUM_CALIB_IMAGES = 50
WORKSPACE_GB = 4.0
WARMUP_STEPS = 100
TIMED_STEPS = 1000

# Set do_quant=False to skip quantization and build a plain fp32/fp16 engine
# (useful for latency baseline comparison).
do_quant = True


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------
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
        label = f"{class_names.get(det['class_id'], 'unknown')}: {det['confidence']:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="red")
        draw.text((x1 + 2, y1 - th - 2), label, fill="white", font=font)
    if title:
        tb = draw.textbbox((0, 0), title, font=font)
        draw.text((img_pil.width - (tb[2] - tb[0]) - 10, 10), title, fill="yellow", font=font)
    img_pil.save(output_path)
    print(f"Annotated image saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_demo():
    # ------------------------------------------------------------------
    # Step 1: Export YOLOv10 to float32 ONNX
    # ------------------------------------------------------------------
    print(f"\n--- Step 1: Export {MODEL_VERSION} to float32 ONNX ---")

    export_args = export_parse_args(
        [
            "--model_type", "yolov10",
            "--model_version", MODEL_VERSION,
            "--output_path", ONNX_PATH,
            "--format", "float32",
            "--opset", "18",
            "--runtime", "onnx",
            "--simplify",
            "--input_img_size", "640",
        ]
    )

    # Initialise CUDA context before any ultralytics / TRT work
    if torch.cuda.is_available():
        torch.zeros(1).cuda()
        print(f"CUDA context initialised on device: {torch.cuda.get_device_name(0)}")
    else:
        print("No CUDA device found.")

    export_tool_main(export_args)

    # ------------------------------------------------------------------
    # Step 1b: Explicit INT8 quantization via QDQ nodes
    # ------------------------------------------------------------------
    print("\n--- Step 1b: Calibrate + wrap Conv nodes in INT8 QDQ ---")

    import onnx

    base_model = onnx.load(ONNX_PATH)

    if do_quant:
        calib_loader = TRTCalibrationDataLoader(
            input_size=640,
            num_images=NUM_CALIB_IMAGES,
        )
        quant_model = wrap_nodes_in_int8_qdq(
            base_model,
            calib_loader,
            node_types=["Conv"],
            exclude=[
                # Detection head — final class/box layers are very sensitive.
                # Quantizing these collapses confidence scores.
                "model.23",
            ],
        )
        onnx.save(quant_model, QUANT_ONNX_PATH)
        print(f"Quantized ONNX saved to: {QUANT_ONNX_PATH}")
        engine_input_onnx = QUANT_ONNX_PATH
    else:
        print("Skipping quantization — building engine from float32 ONNX.")
        engine_input_onnx = ONNX_PATH

    # ------------------------------------------------------------------
    # Step 2: Build TRT engine from quantized ONNX
    # ------------------------------------------------------------------
    print("\n--- Step 2: Build TRT engine (explicit INT8 quantization) ---")

    onnx2engine_explicit(
        onnx_file=engine_input_onnx,
        engine_file=ENGINE_PATH,
        workspace_gb=WORKSPACE_GB,
        fp16_fallback=True,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Step 3: Load engine + inference validation
    # ------------------------------------------------------------------
    print("\n--- Step 3: TRT Inference Validation ---")

    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(ENGINE_PATH, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(trt_logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # Discover I/O names and shapes
    input_name = output_name = None
    input_shape = output_shape = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        shape = tuple(engine.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            input_name, input_shape = name, shape
        else:
            output_name, output_shape = name, shape
    print(f"Engine input : {input_name} {input_shape}")
    print(f"Engine output: {output_name} {output_shape}")

    # Preprocess — model uses NCHW float32 [0,1] (same as ONNX export)
    preprocessed, original_dims, ratio_pad = preprocess_image(
        SAMPLE_IMAGE_PATH,
        list(input_shape),
        tensor_format="nchw",
        normalize=True,
        uint8_input=False,
    )

    # GPU buffers
    input_gpu = torch.from_numpy(preprocessed).contiguous().cuda()
    output_gpu = torch.zeros(output_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, input_gpu.data_ptr())
    context.set_tensor_address(output_name, output_gpu.data_ptr())

    stream = torch.cuda.current_stream().cuda_stream
    context.execute_async_v3(stream_handle=stream)
    torch.cuda.synchronize()

    raw_output = output_gpu.cpu().numpy()
    post_processor = YOLOv10PostProcessor()
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

    out_img = os.path.join(OUTPUT_DIR, f"detected_{MODEL_VERSION}_trt_int8.jpg")
    visualize_detections(
        SAMPLE_IMAGE_PATH, detections, out_img, GLOBAL_CLASS_NAMES,
        title=f"TRT INT8: {MODEL_VERSION}",
    )

    # ------------------------------------------------------------------
    # Step 4: Latency benchmark — H2D + inference + D2H
    # ------------------------------------------------------------------
    print(f"\n--- Step 4: Latency Benchmark ({WARMUP_STEPS} warmup + {TIMED_STEPS} timed) ---")

    input_cpu = torch.from_numpy(preprocessed).contiguous()
    latencies_ms: List[float] = []

    for step in range(WARMUP_STEPS + TIMED_STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        input_gpu.copy_(input_cpu)                     # H2D
        context.execute_async_v3(stream_handle=stream) # Inference
        _ = output_gpu.cpu()                           # D2H
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if step >= WARMUP_STEPS:
            latencies_ms.append((t1 - t0) * 1000.0)
        if step > 0 and (step + 1) % 200 == 0:
            sofar = latencies_ms or [(t1 - t0) * 1000.0]
            print(f"  [{step+1}/{WARMUP_STEPS+TIMED_STEPS}] avg so far: {sum(sofar)/len(sofar):.2f} ms")

    latencies_ms.sort()
    n = len(latencies_ms)
    mean_ms = sum(latencies_ms) / n
    p50_ms = latencies_ms[n // 2]
    p99_ms = latencies_ms[int(n * 0.99)]
    fps = 1000.0 / mean_ms

    print(f"\nLatency over {n} timed steps (after {WARMUP_STEPS} warmup):")
    print(f"  mean : {mean_ms:.2f} ms  ({fps:.1f} FPS)")
    print(f"  p50  : {p50_ms:.2f} ms")
    print(f"  p99  : {p99_ms:.2f} ms")
    print(f"  min  : {latencies_ms[0]:.2f} ms")
    print(f"  max  : {latencies_ms[-1]:.2f} ms")


if __name__ == "__main__":
    run_demo()

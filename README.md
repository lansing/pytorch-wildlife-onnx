# pytorch-wildlife-onnx

A friendly way to convert the awesome MegaDetector V6 models to ONNX and TensorRT formats, particularly aimed at use with Frigate, LightNVR and other NVR applications.

**Key Features & Capabilities:**
*   **MegaDetector v6 Support:** Including compact and extra versions of MDV6 YOLOv9 and YOLOv10. Extra models support input resolutions up to 1280×1280 for higher-accuracy detection at range.
*   **YOLOv9 Compatibility for YOLOv10:** Intended for compatibility with systems like Frigate, which may not support YOLOv10 model output (which is different from previous YOLOs) we provide a unique YOLOv10 export variant that mimics YOLOv9 output, crucial for existing inference pipelines.
*   **ONNX and TensorRT Engine Export:** Export to `.onnx` for broad runtime compatibility (ONNX Runtime, OpenVINO, TensorRT-RTX, and more), or export directly to a compiled TensorRT `.engine` file for maximum throughput on NVIDIA GPUs.
*   **Precision Options:** FP32, FP16, and INT8 are all supported for both ONNX and TensorRT exports. INT8 uses explicit per-layer calibration with minimal accuracy loss — on a typical wildlife image, confidence scores drop by less than 0.01 compared to FP32. On NVIDIA Turing (RTX 20-series) or later, INT8 delivers approximately **1.3–1.5× throughput** over FP16. INT8 quantization is currently supported for YOLOv10 models only.
*   **Optional Input Preprocessing Baked In:** The exported model can optionally accept input in `uint8` dtype and/or `NHWC` tensor layout (and/or raw 0–255 float values). This moves three transforms that would otherwise run on the CPU — dtype cast, axis transpose, and /255 normalisation — into the model itself, where they execute on the GPU. Depending on your host CPU speed this can shave a few milliseconds off per-frame latency.
*   **Containerized:** Execution in a containerized environment for consistent results across platforms.
*   **TUI:** No problems, you can manage all the export parameters with ease.

<br><img width="640" height="533" alt="image" src="https://github.com/user-attachments/assets/9f9e18d7-cabb-4d43-aebd-c336f93b6b27" />

## Quickstart: Dockerized TUI Experience (Recommended)

For the most streamlined and hassle-free model export, we recommend using our Dockerized TUI.

### 1. Prerequisites
*   [Docker](https://docs.docker.com/get-docker/) or a compatible container runtime installed and running on your system.
*   For TensorRT engine export: an NVIDIA GPU with the NVIDIA Container Toolkit installed.

### 2. Run the Dockerized TUI

**ONNX export** (CPU-only Docker image, works on any machine):
```bash
./run_tui_in_docker.sh
```

**TensorRT engine export** (requires an NVIDIA GPU):
```bash
./run_tui_in_docker.sh --trt
```

The `--trt` flag builds the image from `Dockerfile.trt` and passes `--runtime nvidia` to the container so that the GPU is available during the TensorRT engine compilation step.

This script will:
*   Build the appropriate Docker image for the first run, which may take a few minutes.
*   Create local `checkpoints`, `exported_models`, and `cache` directories.
*   Launch the interactive TUI within a Docker container.
*   Use `exported_models` as the destination for your exported models and class metadata.

Follow the on-screen prompts in the TUI to select your desired model type, version, format, and other export options. Once complete, your exported model(s) and associated class files will be available in your local `exported_models` directory.


## Model Selection Recommendations for NVR Detectors (Frigate etc)

Choosing the right model and configuration for your object detection needs is crucial for balancing performance and accuracy. Here are some general guidelines:

*   **General Recommendation:** For most users, we suggest starting with the **YOLOv10 compact model in YOLOv9 compatible format**, exported at **float16 precision** and **320** image size. This offers a good balance of performance and compatibility.
    * Accepting the default options throughout the TUI experience will result in this export.
*   **Older Intel iGPU (8th-14th Gen) or Low-Power Edge AI Accelerators (e.g., Edge TPU):**
    *   You will mostly be restricted to using **compact models at 320px input size**.
    *   The YOLOv10 compact model, with its approximately 2.3 million parameters, is an exceptionally efficient choice for these environments.
*   **Current-Gen Intel iGPU or Low-Power Workstation GPUs:**
    *   It is generally feasible to run **compact models at 640px** input size.
    *   Alternatively, **extra models at 320px** can also provide good performance.
    *   Running extra models at 640px might be acceptable if you can tolerate slightly higher latency.
*   **Dedicated Discrete GPUs (from 2018 onwards):**
    *   Any "full fat" discrete GPU from this era should comfortably run **extra models at 640px or higher** without significant performance issues.

*   **Precision: float16 vs float32 vs int8:**
    *   **float16 is the recommended starting point for most hardware.** It delivers a substantial speedup over float32 with negligible accuracy loss on detection tasks.
    *   **INT8 for TensorRT engine exports (YOLOv10 only):** Users exporting TensorRT `.engine` files on NVIDIA Turing (RTX 20-series / GTX 16-series) or newer GPUs can use INT8 for a further latency reduction — approximately **1.3–1.5× throughput** over float16 — with minimal accuracy impact. This is the recommended path for maximum throughput on supported hardware. See [INT8 quantization deep dive](#int8-quantization-deep-dive) below.
    *   **INT8 for ONNX exports:** Do **not** use INT8 for ONNX models intended for the CUDA execution provider — ONNX Runtime's CUDA EP has no native INT8 kernel support, so the quantization nodes add overhead without any speed benefit. However, users targeting other ONNX-compatible runtimes such as **TensorRT EP**, **TensorRT-RTX**, or **OpenVINO** can experiment with INT8 ONNX exports and may see a performance improvement depending on their runtime's INT8 kernel support.
    *   **float32:** Only use float32 if you have a specific reason — e.g., a very old runtime or hardware that does not support float16.

*   **YOLOv10 vs YOLOv9:**
    *   YOLOv10 provides better animal recall with dramatically lower computational burden. We recommend using YOLOv10 over YOLOv9. Use the v9 compatibility mode if your software (i.e. Frigate) does not support YOLOv10 output.

## Frigate Configuration

Once you have exported your desired model, copy the exported model file and the generated `md.classes.txt` to a location accessible by your Frigate container (e.g. `/media/frigate/models/`).

### Standard configuration (ONNX, float16, NCHW, float input)

```yaml
model:
  model_type: yolo-generic
  width: 640  # Match to your export --input_img_size
  height: 640 # Match to your export --input_img_size
  input_tensor: nchw
  input_dtype: float
  path: /models/MDV6-yolov10-c_float16_640_v9_compat.onnx
  labelmap_path: /models/md.classes.txt

objects:
  track:
    - person
    - animal
    - vehicle
```

### Optimised configuration: GPU-side preprocessing (uint8, NHWC)

If you exported your model with the **uint8 input**, **denormalized**, and **NHWC** options enabled, Frigate can send the raw decoded frame directly to the model without any CPU-side preprocessing. Enable these options in the TUI at export time, then update your Frigate config:

```yaml
model:
  model_type: yolo-generic
  width: 640
  height: 640
  input_tensor: nhwc   # <-- changed
  input_dtype: int     # <-- changed
  path: /models/MDV6-yolov10-c_float16_640_v9_compat_denorm_nhwc_uint8input.onnx
  labelmap_path: /models/md.classes.txt
```

This moves the uint8→float32 cast, the NHWC→NCHW transpose, and the /255 normalisation from the CPU into the model where they execute on the GPU. Depending on your host CPU speed, this can reduce per-frame latency by a few milliseconds.

### Frigate and YOLOv10 compatibility

As of **Frigate 0.17**, Frigate does not support the native YOLOv10 output format. If you want to use a YOLOv10 model with Frigate, you **must** export it in **YOLOv9 compatibility mode** (`yolov10_v9_compatible` model type in the TUI). This produces a model whose output tensor is reshaped to match the YOLOv9 layout that Frigate expects.

*Note: Ensure the `path` and `labelmap_path` reflect the actual filenames and locations accessible within your Frigate container. Also, verify that `width` and `height` match the `input_img_size` chosen during the model export process.*

## INT8 quantization deep dive

INT8 quantization is currently supported for **YOLOv10 models only** (compact and extra, all input sizes). YOLOv9 models are not supported.

### Observed speedup

On NVIDIA Turing (RTX 20-series / GTX 16-series) and later, INT8 delivers approximately **1.3–1.5× throughput** compared to FP16 for the same model and input size. The exact gain varies by GPU generation, model variant, and input resolution.

### Hand-tuned quantization for YOLOv10

The INT8 export is not a generic ONNX quantization pass. The quantized node selection has been carefully hand-tuned for the YOLOv10 architecture to maximise TensorRT's ability to fuse operations into single high-throughput INT8 kernels:

- **Conv-only and blanket profiles:** Two quantization profiles are available. The default `conv` profile wraps only convolution weights in INT8 QDQ (Quantize/Dequantize) pairs. The `blanket` profile additionally quantizes residual `Add` and `MaxPool` nodes, which eliminates layout-reformat overhead on the shortcut paths in YOLOv10's C2f blocks and yields higher throughput on most NVIDIA GPUs.
- **Conv+SiLU+Add fusion:** The QDQ nodes are positioned so that TensorRT is likely to fuse a Conv → SiLU activation → residual Add sequence into a single INT8 Tensor Core kernel. This fusion is the primary source of the speedup over FP16 and depends on the quantization graph structure being correct — it will not occur with a naive or untargeted INT8 quantization pass.
- **Sensitivity-aware exclusions:** Certain layers are deliberately excluded from INT8 quantization. The input stem (`model.0`) is excluded because its 3-channel RGB input falls outside the INT8 Tensor Core path on most NVIDIA hardware. The detection head (`model.23`) is excluded because its output scales are sensitive to quantization noise in a way that affects NMS thresholding; keeping it in FP16 preserves detection accuracy at negligible throughput cost.
- **Float32 bias preservation:** Conv biases are kept as float32 inside the Conv node rather than being folded into a separate post-dequantize Add. This is required for TRT to apply the Conv+SiLU epilogue fusion.

### TensorRT is required for full benefit

The INT8 ONNX export contains explicit QDQ nodes that encode the per-layer calibration scales. The speedup described above is only realised when using a runtime that understands and can execute these nodes as INT8 kernels:

- **TensorRT `.engine` export** (via `--runtime tensorrt`) is the **strongly recommended path**. TRT reads the embedded QDQ scales directly (no separate calibration step at engine-build time), applies the fusion optimisations described above, and produces the fastest possible engine for the target GPU.
- **TensorRT-RTX and TensorRT EP for ONNX Runtime** should also benefit, as both backends perform similar QDQ-aware graph optimisation.
- **ONNX Runtime CUDA EP** will not benefit — it lacks native INT8 kernel dispatch for these node types and will run the quantized graph slower than FP16.
- **CPU and OpenVINO runtimes** may or may not benefit depending on their INT8 support; results will vary.

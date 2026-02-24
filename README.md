# pytorch-wildlife-onnx

A friendly way to convert the awesome MegaDetector V6 models to ONNX format, particularly aimed at use with Frigate, LightNVR and other NVR applications.

**Key Features & Capabilities:**
*   **MegaDetector v6 Support:** Including compact and extra versions of MDV6 YOLOv9 and YOLOv10.
*   **YOLOv9 Compatibility for YOLOv10:** Intended for compatibility with systems like Frigate, which may not support YOLOv10 model output (which is different from previous YOLOs) we provide a unique YOLOv10 export variant that mimics YOLOv9 output, crucial for existing inference pipelines.
*   **Precision Options:** FP16 by default for accelerated inference on most recent hardware, or FP32 if you prefer. Experimental INT8 quantization is also available (note: INT8 is currently not fully tested and not recommended for everyday use).
*   **Containerized:** Execution in a containerized environment for consistent results across platforms.
*   **TUI:** No problems, you can manage all the export parameters with ease.

<br><img width="640" height="533" alt="image" src="https://github.com/user-attachments/assets/9f9e18d7-cabb-4d43-aebd-c336f93b6b27" />

## Quickstart: Dockerized TUI Experience (Recommended)

For the most streamlined and hassle-free model export, we recommend using our Dockerized TUI.

### 1. Prerequisites
*   [Docker](https://docs.docker.com/get-docker/) or a compatible container runtime installed and running on your system.

### 2. Run the Dockerized TUI
Navigate to the root of this repository in your terminal and execute the provided script:
```bash
./run_tui_in_docker.sh
```
This script will:
*   Build a Docker image (named `pytorch-wildlife-export-tui`) for the first run, which may take a few minutes.
*   Create local `checkpoints` and `exported_models` directories.
*   Launch the interactive TUI within a Docker container.
*   Use `exported_models` as the destination for your exported models and class metadata

Follow the on-screen prompts in the TUI to select your desired model type, version, format, and other export options. Once complete, your exported ONNX model(s) and associated class files will be available in your local `exported_models` directory.


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
*   **Precision (float16 vs. float32):**
    *   **Always use float16 models** (pretty much) unless you have a specific, known reason to opt for float32 (e.g., compatibility issues with older hardware/software, or extreme precision requirements that outweigh performance gains). Float16 generally offers superior performance with minimal impact on accuracy for detection tasks.
*   **YOLOv10 vs YOLOv9**
    *   YOLOv10 provides better animal recall with dramatically lower computational burden. We recommend using YOLOv10 over YOLOv9. Use the v9 compatibility mode if your software (i.e. Frigate) does not support YOLOv10 output.

## Frigate Configuration Suggestion

Once you have exported your desired ONNX model, you can integrate it into your Frigate `config.yaml`. Remember to copy your exported `.onnx` model and the generated `md.classes.txt` file to a location accessible by your Frigate container (e.g., `/media/frigate/models/` or whatever is configured in your docker-compose.yaml).

Here's a template for your Frigate `config.yaml` detector block:

```yaml
model:
  model_type: yolo-generic
  width: 640  # <--- IMPORTANT: Match this to your export --input_img_size
  height: 640 # <--- IMPORTANT: Match this to your export --input_img_size
  input_tensor: nchw
  input_dtype: float
  path: /models/MDV6-yolov10-c_float16_640_v9_compat.onnx # <--- IMPORTANT: Update with your ONNX model filename
  labelmap_path: /models/md.classes.txt # <--- IMPORTANT: Ensure this path is correct

objects:
  track:
    - person
    - animal
    - vehicle # <--- IMPORTANT: Add this block to track MegaDetector classes
```
*Note: Ensure the `path` and `labelmap_path` reflect the actual filenames and locations accessible within your Frigate container. Also, verify that `width` and `height` match the `input_img_size` chosen during the model export process.*

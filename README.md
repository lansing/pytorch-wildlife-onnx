# pytorch-wildlife-onnx

A friendly way to convert the awesome MegaDetectory V6 models to ONNX format, particularly aimed at use with Frigate, LightNVR and other NVR applications.

**Key Features & Capabilities:**
*   **MegaDetector v6 Support:** Seamlessly export the latest MegaDetector v6 models, including compact and extra versions of MDV6 YOLOv9 and YOLOv10.
*   **YOLOv9 Compatibility for YOLOv10:** Intended for compatibility with systems like Frigate, which may not support YOLOv10 model output (different from previous YOLOs) we provide a unique YOLOv10 export variant that mimics YOLOv9 output, crucial for existing inference pipelines.
*   **Precision Options:** FP16 by default for accelerated inference on most recent hardware, or FP32 if you prefer. Experimental INT8 quantization is also available (note: INT8 is currently not fully tested and not recommended for everyday use).
*   **Containerized:** Super streamlined export process from a contained environment, ensuring consistent results across platforms.
*   **TUI:** Navigate complex export parameters with ease.

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


## Frigate Configuration Suggestion

Once you have exported your desired ONNX model, you can integrate it into your Frigate `config.yaml`. Remember to copy your exported `.onnx` model and the generated `md.classes.txt` file to a location accessible by your Frigate container (e.g., `/media/frigate/models/`).

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

## Model Selection Recommendations for NVR Detectors (Frigate etc)

Choosing the right model and configuration for your object detection needs is crucial for balancing performance and accuracy. Here are some general guidelines:

*   **General Quickstart:** For most users, we suggest starting with the **YOLOv10 compact model in YOLOv9 compatible format**, exported at **float16 precision**. This offers a good balance of performance and compatibility.
*   **Older Intel iGPU (8th-14th Gen) or Low-Power Edge AI Accelerators (e.g., Edge TPU):**
    *   Consider using the **compact models at 320px input size**.
    *   The YOLOv10 compact model, with its approximately 2.3 million parameters, is an exceptionally efficient choice for these environments.
*   **Current-Gen Intel iGPU or Low-Power Workstation GPUs:**
    *   It is generally feasible to run **compact models at 640px** input size.
    *   Alternatively, **extra models at 320px** can also provide good performance.
    *   Running extra models at 640px might be acceptable if you can tolerate slightly higher latency.
*   **Dedicated Discrete GPUs (from 2018 onwards):**
    *   Any "full fat" discrete GPU from this era should comfortably run **extra models at 640px or higher** without significant performance issues.
*   **Precision (float16 vs. float32):**
    *   **Always use float16 models** unless you have a specific, known reason to opt for float32 (e.g., compatibility issues with older hardware/software, or extreme precision requirements that outweigh performance gains). Float16 generally offers superior performance with minimal impact on accuracy for detection tasks.

## Host Runtime Details: Manual Setup

Most users can skip everything from here onward. 

If you prefer to run the export tools directly on your host system without Docker, follow these steps for manual environment setup and execution of the CLI or TUI scripts.

### 1. Setup Environment
Ensure you have `pyenv` installed. Then, set up your development environment by running:
```bash
make install
```
This will install Python 3.11.8, set up a `uv` virtual environment, and install all necessary dependencies.

### 2. Export a Model via CLI
You can export a `MegaDetectorV6 YOLOv9 compact` model to ONNX (float32, with simplification, *raw pre-NMS output*) using the `export_tool.py` CLI:
```bash
source .venv/bin/activate
python PytorchWildlife_Export/export_tool.py \
    --model_type yolov9 \
    --model_version MDV6-yolov9-c \
    --output_path exported_models_test/MDV6-yolov9-c_1280x1280_raw.onnx \
    --format float32 \
    --opset 18 \
    --simplify \
    --input_img_size 1280
```
*(The model weights will be downloaded on first export if not present locally.)*

### 3. Use the Interactive TUI
For an interactive experience on your host, you can run the TUI script directly after setting up the environment:
```bash
source .venv/bin/activate
python PytorchWildlife_Export/tui_export.py
```

### 4. Run the Inference Demo
After exporting a model, you can run the image detection demo:
```bash
source .venv/bin/activate
python PytorchWildlife_Export/demo/onnx_image_detector_demo.py
```
This will load the `exported_models_test/MDV6-yolov9-c_1280x1280_raw.onnx` model, run inference on `PytorchWildlife_Export/demo/sample_image.jpg`, and save two annotated images to `PytorchWildlife_Export/demo/demo_output/`: `detected_sample_image_custom_pp.jpg` (our custom post-processing) and `detected_sample_image_ultralytics_baseline.jpg` (Ultralytics' interpretation).

## Table of Contents
1. [Setup](#setup)
2. [Model Inspector/Validator](#model-inspectorvalidator)
3. [Model Loaders](#model-loaders)
4. [Model Exporters](#model-exporters)
5. [Inference Utilities](#inference-utilities)

## Setup

### 1. Clone the Repository
First, clone this repository including its submodules:
```bash
git clone --recurse-submodules <repository_url>
cd pytorch-wildlife-export
```
*(Replace `<repository_url>` with the actual URL of this repository)*

### 2. Python Environment Setup
We use `pyenv` for Python version management and `uv` for virtual environment and package management.
*(Note: This is automatically handled by `make install`)*

**a. Install Python 3.11:**
```bash
pyenv install 3.11.8
pyenv local 3.11.8
```

**b. Install `uv`:**
```bash
pip install uv
```

**c. Create and Activate Virtual Environment:**
```bash
uv venv
source .venv/bin/activate
```
*(Note: If you are using `fish` shell, the activation command might be `source .venv/bin/activate.fish`)*

**d. Install Dependencies:**
Navigate to the `PytorchWildlife_Export` directory and install the required Python packages.
```bash
uv pip install -r PytorchWildlife_Export/requirements.txt
```

## Model Inspector/Validator

The `model_validators` module contains utilities to inspect and validate ONNX models.

**To run the validator tests:**
```bash
source .venv/bin/activate
python PytorchWildlife_Export/tests/test_onnx_validator.py
```
This will run tests against a sample ONNX model (`sample_models/MDV6-yolov9-c-320-16b.onnx`) to ensure the validator can load models and perform a forward pass.

## Model Loaders

The `model_loaders` module provides classes to load PyTorch models for various architectures, allowing access to the underlying `torch.nn.Module`.

**To run the model loader tests:**
```bash
source .venv/bin/activate
python PytorchWildlife_Export/tests/test_model_loaders.py
```
This will test loading of `MegaDetectorV6 YOLOv9` and `RT-DETR` models.
*(Note: The RT-DETR model loader might download additional weights during its first run.)*

## Model Exporters

The `model_exporters` module contains tools to export PyTorch models to the ONNX format, with support for different numeric formats and graph simplification.

**To run the model exporter tests:**
*(This will also export models to `exported_models_test/` directory)*
```bash
source .venv/bin/activate
python PytorchWildlife_Export/tests/test_model_exporters.py
```
This will test exporting `MegaDetectorV6 YOLOv9` models to ONNX in `float32` and `float16` formats, including simplification, and then validate these exported models.

**Note on RT-DETR Export:**
The RT-DETR model export via `torch.onnx.export` currently produces a syntactically valid ONNX file but faces compatibility issues with `onnxruntime` during loading due to an initializer-related error. The corresponding test is currently skipped.

## Inference Utilities

The `inference_utils` module provides classes to perform inference using the exported ONNX models. These models are assumed to output raw (pre-NMS) predictions, and the post-processing is handled by custom code.

**To run the ONNX Image Detector Demo (YOLOv9):**
This demo script uses an exported `MegaDetectorV6 YOLOv9` ONNX model (raw output) to detect objects in a sample image and visualize the results using our custom post-processing, comparing them against the Ultralytics baseline.

First, ensure the required YOLOv9 1280x1280 ONNX model (raw output) for the demo is exported (this is done by running the `make export` command from the Quickstart).
```bash
source .venv/bin/activate
python PytorchWildlife_Export/demo/onnx_image_detector_demo.py
```
The script will load the `exported_models_test/MDV6-yolov9-c_1280x1280_raw.onnx` model, run inference on `PytorchWildlife_Export/demo/sample_image.jpg`, and save two annotated images to `PytorchWildlife_Export/demo/demo_output/`: `detected_sample_image_custom_pp.jpg` (our custom post-processing) and `detected_sample_image_ultralytics_baseline.jpg` (Ultralytics' interpretation).


---
**Development Notes:**
- All new scripts and utilities are located in the `PytorchWildlife_Export/` directory to maintain separation from the `CameraTraps` submodule.
- The `CameraTraps` repository is included as a Git submodule, pinned to a specific commit, and no modifications were made to it.

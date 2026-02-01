# pytorch-wildlife-onnx

A friendly way to convert the awesome MegaDetectory V6 models to ONNX format, particularly aimed at use with Frigate, LightNVR and other NVR applications.

**Key Features & Capabilities:**
*   **MegaDetector v6 Support:** Seamlessly export the latest MegaDetector v6 models, including compact and extra versions of MDV6 YOLOv9 and YOLOv10.
*   **YOLOv9 Compatibility for YOLOv10:** Intended for mpatibility with systems like Frigate, which may not support YOLOv10 model output (different from previous YOLOs) we provide a unique YOLOv10 export variant that mimics YOLOv9 output, crucial for existing inference pipelines.
*   **Flexible Precision:** FP16 (half-precision) models for accelerated inference or robust FP32 (single-precision) models. Experimental INT8 quantization is also available for ultra-low latency scenarios (note: INT8 is currently not fully tested and not recommended for everyday use).
*   **Containerized:** Super streamlined export process from a contained environment, ensuring consistent results across platforms.
*   **TUI:** Navigate complex export parameters with ease.

## Quickstart: Dockerized TUI Experience (Recommended)

For the most streamlined and hassle-free model export, we highly recommend using our Dockerized TUI. This approach sets up all dependencies in a consistent environment and guides you through the export process with an intuitive Text-based User Interface.

### 1. Prerequisites
*   [Docker](https://docs.docker.com/get-docker/) installed and running on your system.

### 2. Run the Dockerized TUI
Navigate to the root of this repository in your terminal and execute the provided script:
```bash
./run_tui_in_docker.sh
```
This script will:
*   Build a Docker image (named `pytorch-wildlife-export-tui`) for the first run, which may take a few minutes as it downloads and installs all dependencies. Subsequent runs will be much faster due to Docker's caching.
*   Automatically create local `checkpoints` and `exported_models` directories (if they don't already exist).
*   Launch the interactive TUI within a Docker container.
*   Mount your local `checkpoints` directory to `/root/.cache/torch/hub/checkpoints` inside the container. This allows model weights to be downloaded once and reused across different exports or runs, saving bandwidth and time.
*   Mount your local `exported_models` directory to `/exported_models` inside the container. All your exported ONNX models will be saved here and be accessible on your host machine.
*   The TUI will automatically use `/exported_models` as the destination for your exported models, bypassing the interactive prompt for the output directory.

Follow the on-screen prompts in the TUI to select your desired model type, version, format, and other export options. Once complete, your exported ONNX model(s) and associated class files will be available in your local `exported_models` directory.

## Host Runtime Details: Manual Setup

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

# PytorchWildlife Export Tools

This repository contains custom scripts and utilities to export models from the PytorchWildlife project into various formats (specifically ONNX initially) and to run inference with the exported models.

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

The `inference_utils` module provides classes to perform inference using the exported ONNX models.

**To run the ONNX Image Detector Demo (YOLOv9):**
This demo script uses an exported `MegaDetectorV6 YOLOv9` ONNX model to detect objects in a sample image and visualize the results.

First, ensure the required YOLOv9 1280x1280 ONNX model for the demo is exported (this is done by running `test_model_exporters.py` which includes a test for exporting this model).
```bash
source .venv/bin/activate
python PytorchWildlife_Export/demo/onnx_image_detector_demo.py
```
The script will load the `exported_models_test/MDV6-yolov9-c_1280x1280.onnx` model, run inference on `PytorchWildlife_Export/demo/sample_image.jpg`, and save the annotated image to `PytorchWildlife_Export/demo/demo_output/detected_sample_image.jpg`.

---
**Development Notes:**
- All new scripts and utilities are located in the `PytorchWildlife_Export/` directory to maintain separation from the `CameraTraps` submodule.
- The `CameraTraps` repository is included as a Git submodule, pinned to a specific commit, and no modifications were made to it.

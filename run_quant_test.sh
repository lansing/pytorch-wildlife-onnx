#!/bin/bash

# Define image name
IMAGE_NAME="pytorch-wildlife-export-trt"

# Define local directories to be mounted
LOCAL_CHECKPOINTS_DIR="checkpoints"
LOCAL_EXPORTED_MODELS_DIR="exported_models"
LOCAL_CALIB_CACHE_DIR="cache/calibration"
LOCAL_TRT_CACHE_DIR="cache/trt_engines"

# Define container paths for mounting
CONTAINER_CHECKPOINTS_DIR="/root/.cache/torch/hub/checkpoints"
CONTAINER_EXPORTED_MODELS_DIR="/exported_models"
CONTAINER_CALIB_CACHE_DIR="/root/.cache/pytorch_wildlife_export/calibration"

# Ensure local directories exist
mkdir -p "$LOCAL_CHECKPOINTS_DIR"
mkdir -p "$LOCAL_EXPORTED_MODELS_DIR"
mkdir -p "$LOCAL_CALIB_CACHE_DIR"
mkdir -p "$LOCAL_TRT_CACHE_DIR"

# Build the Docker image
# TODO add back --no-cache for distribution
#docker build --no-cache -t "$IMAGE_NAME" .
#docker build --build-arg CACHE_BUSTER=$(date +%s) -t "$IMAGE_NAME" .

#if [ $? -ne 0 ]; then
#    echo "Docker image build failed. Exiting."
#    exit 1
#fi
#
#echo "Running TUI in Docker container..."
#echo "Mounting $LOCAL_CHECKPOINTS_DIR to $CONTAINER_CHECKPOINTS_DIR"
#echo "Mounting $LOCAL_EXPORTED_MODELS_DIR to $CONTAINER_EXPORTED_MODELS_DIR"

# Run the Docker container
# The -it flag is crucial for interactive TUI applications
# --rm removes the container after it exits
# -v mounts the local directories to the container paths
# --entrypoint can be used to override the Dockerfile's ENTRYPOINT if needed, but not here
docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
    -v "$(pwd)/$LOCAL_CHECKPOINTS_DIR:$CONTAINER_CHECKPOINTS_DIR" \
    -v "$(pwd)/$LOCAL_EXPORTED_MODELS_DIR:$CONTAINER_EXPORTED_MODELS_DIR" \
    -v "$(pwd)/$LOCAL_CALIB_CACHE_DIR:$CONTAINER_CALIB_CACHE_DIR" \
    -v "$(pwd)/$LOCAL_TRT_CACHE_DIR:/exported_models/trt_cache" \
    -v "$(pwd):/app" \
    --entrypoint python3 \
    "$IMAGE_NAME" \
    PytorchWildlife_Export/demo/yolov10_trt_quant.py

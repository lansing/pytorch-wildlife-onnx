#!/bin/bash

# Define image name
IMAGE_NAME="pytorch-wildlife-export-trt"

# Define local directories to be mounted
LOCAL_CHECKPOINTS_DIR="checkpoints"
LOCAL_EXPORTED_MODELS_DIR="exported_models"
LOCAL_CALIB_CACHE_DIR="cache/calibration"

# Define container paths for mounting
CONTAINER_CHECKPOINTS_DIR="/root/.cache/torch/hub/checkpoints"
CONTAINER_EXPORTED_MODELS_DIR="/exported_models"
CONTAINER_CALIB_CACHE_DIR="/root/.cache/pytorch_wildlife_export/calibration"

# Ensure local directories exist
mkdir -p "$LOCAL_CHECKPOINTS_DIR"
mkdir -p "$LOCAL_EXPORTED_MODELS_DIR"
mkdir -p "$LOCAL_CALIB_CACHE_DIR"

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
docker run -it --runtime nvidia --rm \
    -v "$(pwd)/$LOCAL_CHECKPOINTS_DIR:$CONTAINER_CHECKPOINTS_DIR" \
    -v "$(pwd)/$LOCAL_EXPORTED_MODELS_DIR:$CONTAINER_EXPORTED_MODELS_DIR" \
    -v "$(pwd)/$LOCAL_EXPORTED_MODELS_DIR:/app/PytorchWildlife_Export/demo/demo_output" \
    -v "$(pwd)/$LOCAL_CALIB_CACHE_DIR:$CONTAINER_CALIB_CACHE_DIR" \
    -v "$(pwd):/app" \
    --entrypoint trtexec \
    "$IMAGE_NAME" \
    --loadEngine=/app/PytorchWildlife_Export/demo/demo_output/MDV6-yolov10-c_int8_640_denorm_nhwc_uint8input_rtx3050.engine \
    --iterations=1000 \
    --warmUp=200 \
    --device=1

     # --dumpLayerInfo \
     # --dumpProfile \
     # --profilingVerbosity=detailed

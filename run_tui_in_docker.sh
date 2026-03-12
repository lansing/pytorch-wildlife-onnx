#!/bin/bash

# Parse arguments
USE_TRT=0
for arg in "$@"; do
    case "$arg" in
        --trt) USE_TRT=1 ;;
        *) echo "Unknown option: $arg"; echo "Usage: $0 [--trt]"; exit 1 ;;
    esac
done

# Select image name and Dockerfile based on --trt flag
if [ "$USE_TRT" -eq 1 ]; then
    IMAGE_NAME="pytorch-wildlife-export-trt"
    DOCKERFILE="Dockerfile.trt"
else
    IMAGE_NAME="pytorch-wildlife-export-tui"
    DOCKERFILE="Dockerfile"
fi

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

echo "Building Docker image: $IMAGE_NAME (using $DOCKERFILE)..."
# TODO add back --no-cache for distribution
#docker build --no-cache -f "$DOCKERFILE" -t "$IMAGE_NAME" .
docker build --build-arg CACHE_BUSTER=$(date +%s) -f "$DOCKERFILE" -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    echo "Docker image build failed. Exiting."
    exit 1
fi

echo "Running TUI in Docker container..."
echo "Mounting $LOCAL_CHECKPOINTS_DIR to $CONTAINER_CHECKPOINTS_DIR"
echo "Mounting $LOCAL_EXPORTED_MODELS_DIR to $CONTAINER_EXPORTED_MODELS_DIR"

# For TRT we need GPU access; use --runtime nvidia
if [ "$USE_TRT" -eq 1 ]; then
    DOCKER_RUN_FLAGS="--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all"
else
    DOCKER_RUN_FLAGS=""
fi

docker run -it --rm $DOCKER_RUN_FLAGS \
    -v "$(pwd)/$LOCAL_CHECKPOINTS_DIR:$CONTAINER_CHECKPOINTS_DIR" \
    -v "$(pwd)/$LOCAL_EXPORTED_MODELS_DIR:$CONTAINER_EXPORTED_MODELS_DIR" \
    -v "$(pwd)/$LOCAL_CALIB_CACHE_DIR:$CONTAINER_CALIB_CACHE_DIR" \
    "$IMAGE_NAME" \
    --output-dir-cli "$CONTAINER_EXPORTED_MODELS_DIR"

echo "TUI exited. Check '$LOCAL_EXPORTED_MODELS_DIR' for exported models."

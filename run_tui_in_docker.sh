#!/bin/bash

# Define image name
IMAGE_NAME="pytorch-wildlife-export-tui"

# Define local directories to be mounted
LOCAL_CHECKPOINTS_DIR="checkpoints"
LOCAL_EXPORTED_MODELS_DIR="exported_models"

# Define container paths for mounting
CONTAINER_CHECKPOINTS_DIR="/root/.cache/torch/hub/checkpoints"
CONTAINER_EXPORTED_MODELS_DIR="/exported_models"

# Ensure local directories exist
mkdir -p "$LOCAL_CHECKPOINTS_DIR"
mkdir -p "$LOCAL_EXPORTED_MODELS_DIR"

echo "Building Docker image: $IMAGE_NAME..."
# Build the Docker image
docker build --no-cache -t "$IMAGE_NAME" .

if [ $? -ne 0 ]; then
    echo "Docker image build failed. Exiting."
    exit 1
fi

echo "Running TUI in Docker container..."
echo "Mounting $LOCAL_CHECKPOINTS_DIR to $CONTAINER_CHECKPOINTS_DIR"
echo "Mounting $LOCAL_EXPORTED_MODELS_DIR to $CONTAINER_EXPORTED_MODELS_DIR"

# Run the Docker container
# The -it flag is crucial for interactive TUI applications
# --rm removes the container after it exits
# -v mounts the local directories to the container paths
# --entrypoint can be used to override the Dockerfile's ENTRYPOINT if needed, but not here
docker run -it --rm \
    -v "$(pwd)/$LOCAL_CHECKPOINTS_DIR:$CONTAINER_CHECKPOINTS_DIR" \
    -v "$(pwd)/$LOCAL_EXPORTED_MODELS_DIR:$CONTAINER_EXPORTED_MODELS_DIR" \
    "$IMAGE_NAME" \
    --output-dir-cli "$CONTAINER_EXPORTED_MODELS_DIR"

echo "TUI exited. Check '$LOCAL_EXPORTED_MODELS_DIR' for exported models."

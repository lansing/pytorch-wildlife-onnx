#!/bin/bash

# Export MDV6-yolov10-e at 640px, INT8 (conv profile), TensorRT engine,
# with all input preprocessing baked in (uint8 + denormalized + NHWC).
#
# Mounts the current source tree into /app so the latest code changes are
# picked up without requiring a Docker image rebuild.
# Requires the pytorch-wildlife-export-trt image to already be built:
#   ./run_tui_in_docker.sh --trt   (just cancel out of the TUI after it builds)

IMAGE_NAME="pytorch-wildlife-export-trt"

LOCAL_CHECKPOINTS_DIR="checkpoints"
LOCAL_EXPORTED_MODELS_DIR="exported_models"
LOCAL_CALIB_CACHE_DIR="cache/calibration"

CONTAINER_CHECKPOINTS_DIR="/root/.cache/torch/hub/checkpoints"
CONTAINER_EXPORTED_MODELS_DIR="/exported_models"
CONTAINER_CALIB_CACHE_DIR="/root/.cache/pytorch_wildlife_export/calibration"

OUTPUT_FILENAME="MDV6-yolov10-e_int8_640_denorm_nhwc_uint8input.engine"

mkdir -p "$LOCAL_CHECKPOINTS_DIR"
mkdir -p "$LOCAL_EXPORTED_MODELS_DIR"
mkdir -p "$LOCAL_CALIB_CACHE_DIR"

echo "Exporting $OUTPUT_FILENAME ..."
echo "  model:   MDV6-yolov10-e"
echo "  format:  int8 / conv profile"
echo "  runtime: tensorrt engine"
echo "  input:   640px, uint8 NHWC denormalized"
echo ""

docker run --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all --rm \
    -v "$(pwd)/$LOCAL_CHECKPOINTS_DIR:$CONTAINER_CHECKPOINTS_DIR" \
    -v "$(pwd)/$LOCAL_EXPORTED_MODELS_DIR:$CONTAINER_EXPORTED_MODELS_DIR" \
    -v "$(pwd)/$LOCAL_CALIB_CACHE_DIR:$CONTAINER_CALIB_CACHE_DIR" \
    -v "$(pwd):/app" \
    --entrypoint python3 \
    "$IMAGE_NAME" \
    PytorchWildlife_Export/export_tool.py \
        --model_type      yolov10 \
        --model_version   MDV6-yolov10-e \
        --output_path     "$CONTAINER_EXPORTED_MODELS_DIR/$OUTPUT_FILENAME" \
        --format          int8 \
        --quant_profile   blanket \
        --runtime         tensorrt \
        --input_img_size  640 \
        --uint8_input \
        --denormalized_input \
        --nhwc_input \
        --simplify

if [ $? -eq 0 ]; then
    echo ""
    echo "Done. Engine saved to: $LOCAL_EXPORTED_MODELS_DIR/$OUTPUT_FILENAME"
else
    echo ""
    echo "Export failed."
    exit 1
fi

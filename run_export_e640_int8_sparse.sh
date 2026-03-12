#!/bin/bash

# Export MDV6-yolov10-e at 640px, INT8 (conv profile) + 2:4 structured sparsity.
# Same settings as run_export_e640_int8.sh but with --sparse_weights.
#
# On Turing (RTX 2080 Ti, CC 7.5): engine builds successfully but TRT does NOT
# select sparse Tensor Core kernels — no speedup expected.  The purpose of this
# build is to verify that:
#   1. TRT accepts the 2:4-pruned weights without errors
#   2. The layer dump shows SPARSE tactic names (or not — either answer is useful)
#
# On Ampere (CC 8.0+): sparse Tensor Core kernels should be selected for Conv
# layers, providing up to 2x math throughput improvement.

IMAGE_NAME="pytorch-wildlife-export-trt"

LOCAL_CHECKPOINTS_DIR="checkpoints"
LOCAL_EXPORTED_MODELS_DIR="exported_models"
LOCAL_CALIB_CACHE_DIR="cache/calibration"

CONTAINER_CHECKPOINTS_DIR="/root/.cache/torch/hub/checkpoints"
CONTAINER_EXPORTED_MODELS_DIR="/exported_models"
CONTAINER_CALIB_CACHE_DIR="/root/.cache/pytorch_wildlife_export/calibration"

OUTPUT_FILENAME="MDV6-yolov10-e_int8_sparse_640_denorm_nhwc_uint8input.engine"

mkdir -p "$LOCAL_CHECKPOINTS_DIR"
mkdir -p "$LOCAL_EXPORTED_MODELS_DIR"
mkdir -p "$LOCAL_CALIB_CACHE_DIR"

echo "Exporting $OUTPUT_FILENAME ..."
echo "  model:   MDV6-yolov10-e"
echo "  format:  int8 / conv profile + 2:4 sparse weights"
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
        --quant_profile   conv \
        --runtime         tensorrt \
        --input_img_size  640 \
        --uint8_input \
        --denormalized_input \
        --nhwc_input \
        --simplify \
        --sparse_weights

if [ $? -eq 0 ]; then
    echo ""
    echo "Done. Engine saved to: $LOCAL_EXPORTED_MODELS_DIR/$OUTPUT_FILENAME"
    echo ""
    echo "To check for sparse tactics in the layer dump, run trtexec with:"
    echo "  --loadEngine=$LOCAL_EXPORTED_MODELS_DIR/$OUTPUT_FILENAME"
    echo "  --dumpLayerInfo --profilingVerbosity=detailed"
    echo "Then grep for 'sparse' or 'SPARSE' in the TacticName fields."
else
    echo ""
    echo "Export failed."
    exit 1
fi

#!/bin/bash
set -e

IMAGE_NAME="pytorch-wildlife-export-trt"

# Usage:
#   ./build_tensorrt.sh            — normal build, uses Docker layer cache throughout
#   ./build_tensorrt.sh --fresh    — forces the code COPY layer to re-run, keeps pip cache
#   ./build_tensorrt.sh --no-cache — full rebuild from scratch, no cache at all

CACHE_BUSTER="stable"
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --fresh)
            # Bust only the COPY layer; the expensive pip install stage stays cached.
            CACHE_BUSTER=$(date +%s)
            ;;
        --no-cache)
            EXTRA_ARGS="--no-cache"
            ;;
        --help|-h)
            echo "Usage: $0 [--fresh | --no-cache]"
            echo "  (no flag)    Normal build — full Docker layer cache"
            echo "  --fresh      Re-copies code without re-running pip install"
            echo "  --no-cache   Full rebuild from scratch"
            exit 0
            ;;
    esac
done

echo "Building $IMAGE_NAME (CACHE_BUSTER=$CACHE_BUSTER)..."
docker build \
    -f Dockerfile.trt \
    --build-arg CACHE_BUSTER="$CACHE_BUSTER" \
    $EXTRA_ARGS \
    -t "$IMAGE_NAME" \
    .

echo ""
echo "Build complete:"
docker images "$IMAGE_NAME"

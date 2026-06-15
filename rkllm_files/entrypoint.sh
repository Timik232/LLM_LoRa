#!/bin/bash
set -e

# RKLLM Container Entrypoint Script
# Handles routing of commands to appropriate RKLLM conversion functions

# Set up logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

error() {
    log "ERROR: $*"
    exit 1
}

# If no command provided, just sleep to keep container running
if [ $# -eq 0 ]; then
    log "No command provided. Container is running but idle."
    # Keep container running
    tail -f /dev/null
    exit 0
fi

# Main command routing
case "$1" in
    convert)
        log "Starting RKLLM conversion..."
        shift  # Remove 'convert' from arguments
        python3 /app/rkllm_converting.py "$@"
        ;;

    *)
        log "Unknown command: $1"
        echo "Usage:"
        echo "  docker run rkllm_converter convert [options]"
        echo ""
        echo "Available commands:"
        echo "  convert    Convert model to RKLLM format"
        echo ""
        echo "Convert options:"
        echo "  --model-path PATH          Input model path"
        echo "  --output-path PATH         Output RKLLM file path"
        echo "  --target-platform PLATFORM Target platform (rk3588, rk3576, etc.)"
        echo "  --quantization TYPE        Quantization type (w8a8, w4a16, w4a16_g128)"
        echo "  --num-npu-core N          Number of NPU cores (1-3)"
        echo "  --do-parallelize          Enable model parallelization"
        echo "  --hybrid-quantization     Enable hybrid quantization"
        exit 1
        ;;
esac
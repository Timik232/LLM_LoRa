#!/bin/bash

echo "=== Starting Training Phase ==="
poetry run python main.py pipeline --skip_test=true

RKLLM_ENABLED=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(str(config['model']['rkllm']['enabled']).lower())
")

if [ "$RKLLM_ENABLED" = "true" ]; then
    echo "=== Starting RKLLM Conversion ==="

    # Extract parameters from Hydra config
    MODEL_PATH=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['paths']['merged_model_path'])
")

    OUTPUT_PATH=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
output_dir = config['model']['rkllm']['output_dir']
model_name = config['model']['new_model']
print(f'{output_dir}/{model_name}.rkllm')
")

    TARGET_PLATFORM=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['rkllm']['target_platform'])
")

    QUANTIZATION=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['rkllm']['quantization'])
")

    NPU_CORES=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(config['model']['rkllm']['num_npu_core'])
")

    # Build the conversion command
    CONVERSION_CMD="convert"
    CONVERSION_CMD="$CONVERSION_CMD --model-path /app/models/$MODEL_PATH"
    CONVERSION_CMD="$CONVERSION_CMD --output-path $OUTPUT_PATH"
    CONVERSION_CMD="$CONVERSION_CMD --target-platform $TARGET_PLATFORM"
    CONVERSION_CMD="$CONVERSION_CMD --quantization $QUANTIZATION"
    CONVERSION_CMD="$CONVERSION_CMD --num-npu-core $NPU_CORES"

    # Add optional parameters
    DO_PARALLELIZE=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(str(config['model']['rkllm']['do_parallelize']).lower())
")

    if [ "$DO_PARALLELIZE" = "true" ]; then
        CONVERSION_CMD="$CONVERSION_CMD --do-parallelize"
    fi

    HYBRID_QUANT=$(python3 -c "
import yaml
with open('.hydra/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
print(str(config['model']['rkllm']['hybrid_quantization']).lower())
")

    if [ "$HYBRID_QUANT" = "true" ]; then
        CONVERSION_CMD="$CONVERSION_CMD --hybrid-quantization"
    fi

    # Create output directory if it doesn't exist
    OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
    mkdir -p "/app/models/$OUTPUT_DIR"

    # Run the RKLLM converter container
    echo "Running RKLLM conversion: $CONVERSION_CMD"
    docker run --rm \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        rkllm_converter $CONVERSION_CMD

    if [ $? -eq 0 ]; then
        echo "RKLLM conversion completed successfully"
    else
        echo "RKLLM conversion failed"
        exit 1
    fi
fi


echo "=== Preparing Model for Ollama ==="
GGUF_DIR="models/custom-model"
mkdir -p "$GGUF_DIR"

cat <<EOF > "$GGUF_DIR/Modelfile"
FROM custom-model.gguf
PARAMETER temperature 0.7
EOF

GGUF_FILE="$GGUF_DIR/custom-model.gguf"

if [ ! -f "$GGUF_FILE" ]; then
    echo "Error: $GGUF_FILE does not exist."
    exit 1
fi

echo "=== Creating Model in Ollama ==="
MAX_ATTEMPTS=30
ATTEMPT_NUM=1
until curl -s http://ollama:11434 >/dev/null; do
    if [ $ATTEMPT_NUM -ge $MAX_ATTEMPTS ]; then
        echo "Ollama service not ready, exiting..."
        exit 1
    fi
    echo "Waiting for Ollama service..."
    sleep 1
    ATTEMPT_NUM=$((ATTEMPT_NUM+1))
done
echo "Ollama service is running successfully!"

HASH=$(sha256sum "$GGUF_FILE" | awk '{print $1}')
BLOB_NAME="sha256:$HASH"
echo "Calculated blob name: $BLOB_NAME"

curl -T "$GGUF_FILE" -X POST "http://ollama:11434/api/blobs/$BLOB_NAME"

if [ $? -ne 0 ]; then
    echo "Failed to upload blob"
    exit 1
fi

JSON_PAYLOAD=$(jq -n \
    --arg name "custom-model" \
    --arg blob_name "$BLOB_NAME" \
    --arg gguf_file "$GGUF_FILE" \
    '{name: $name, files: {
    "$gguf_file": $blob_name}}')

CREATE_RESPONSE=$(curl -X POST http://ollama:11434/api/create \
    -H "Content-Type: application/json" \
    -d "$JSON_PAYLOAD" \
    -s -w "\n%{http_code}")

HTTP_CODE=$(echo "$CREATE_RESPONSE" | tail -n1)
RESPONSE_BODY=$(echo "$CREATE_RESPONSE" | sed '$d')

if [ $? -eq 0 ] && [ "$HTTP_CODE" -lt 400 ]; then
    echo "Model creation confirmed"
else
    echo "Model creation failed"
    echo "Response: $RESPONSE_BODY"
    echo "HTTP Code: $HTTP_CODE"
    exit 1
fi

echo "=== Running Integration Tests ==="
poetry run python -m testing_model
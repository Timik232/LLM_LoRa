#!/bin/bash

echo "=== Starting Training Phase ==="
python -m training_model

echo "=== Preparing Model for Ollama ==="
GGUF_DIR="models/custom-model"
mkdir -p "$GGUF_DIR"

echo "$(ls "$GGUF_DIR")"

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
python -m testing_model

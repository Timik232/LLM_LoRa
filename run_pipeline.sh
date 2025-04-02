#!/bin/bash

# 1. Train the model
echo "=== Starting Training Phase ==="
python -m training_model

# 2. Create Modelfile
echo "=== Creating Modelfile ==="
mkdir -p /models/custom_model
cat > /models/custom_model/Modelfile.custom_model <<EOL
FROM /models/custom_model/custom_model.gguf
PARAMETER temperature 0.7
EOL
# 3. Wait for Ollama to be ready
echo "=== Waiting for Ollama ==="
while ! curl -s http://ollama:11434 > /dev/null; do
  echo "Ollama not ready yet, retrying in 5 seconds..."
  sleep 5
done

# 4. Create and push the model
echo "=== Registering Model with Ollama ==="
ollama create custom-model -f /models/custom_model/Modelfile.custom_model

# 5. Run tests
echo "=== Running Integration Tests ==="
python -m testing_model

# 6. Keep container alive for interaction
echo "=== Pipeline Complete ==="
echo "Model is ready for use at http://localhost:11434"
tail -f /dev/null

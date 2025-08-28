Docker Deployment and Ollama Integration
=========================================

The LLM-LoRA framework provides comprehensive Docker-based deployment solutions with integrated Ollama support for serving fine-tuned models. This enables scalable, containerized deployment of trained models with standardized APIs and easy integration.

Overview
--------

Docker deployment offers several advantages for LLM deployment:

* **Containerization**: Consistent deployment environments across different systems
* **Scalability**: Easy horizontal scaling with container orchestration
* **Isolation**: Secure and isolated runtime environments
* **Portability**: Deploy anywhere Docker is supported
* **Ollama Integration**: Standardized API for model serving and inference
* **Multi-Model Support**: Serve multiple models from a single container

**Key Components:**

* **Base Docker Images**: Pre-configured environments with dependencies
* **Model Containers**: Containers with trained models and inference runtime
* **Ollama Integration**: API-compatible model serving with Ollama
* **Orchestration**: Docker Compose and Kubernetes deployment configurations
* **Monitoring**: Built-in monitoring and logging capabilities

Docker Configuration
--------------------

Base Docker Images
~~~~~~~~~~~~~~~~~~

The framework provides several base images for different use cases:

**Training Image:**

.. code-block:: dockerfile

    # Dockerfile.train
    FROM nvidia/cuda:12.1-devel-ubuntu20.04

    # Install system dependencies
    RUN apt-get update && apt-get install -y \\
        python3.9 python3-pip git wget curl \\
        build-essential cmake ninja-build

    # Install Python dependencies
    COPY requirements.txt /tmp/
    RUN pip3 install -r /tmp/requirements.txt

    # Install PyTorch with CUDA support
    RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Set working directory
    WORKDIR /app

    # Copy framework code
    COPY . /app/

    # Entry point for training
    ENTRYPOINT ["python", "main.py"]

**Inference Image:**

.. code-block:: dockerfile

    # Dockerfile.inference
    FROM python:3.9-slim

    # Install runtime dependencies
    RUN apt-get update && apt-get install -y \\
        libgomp1 libgcc-s1 \\
        && rm -rf /var/lib/apt/lists/*

    # Install Python packages
    RUN pip install transformers torch fastapi uvicorn

    # Copy inference code
    COPY inference/ /app/inference/
    COPY models/ /app/models/

    WORKDIR /app

    # Expose API port
    EXPOSE 8000

    # Start inference server
    CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]

**Ollama-Compatible Image:**

.. code-block:: dockerfile

    # Dockerfile.ollama
    FROM ollama/ollama:latest

    # Copy custom model files
    COPY models/converted/ /models/

    # Create Ollama model configurations
    COPY ollama_configs/ /ollama_configs/

    # Setup script to register models
    COPY scripts/setup_ollama.sh /setup_ollama.sh
    RUN chmod +x /setup_ollama.sh

    # Expose Ollama API port
    EXPOSE 11434

    # Initialize and start Ollama
    ENTRYPOINT ["/setup_ollama.sh"]

Building Docker Images
~~~~~~~~~~~~~~~~~~~~~~

Use the provided scripts to build images:

.. code-block:: bash

    # Build training image
    docker build -f Dockerfile.train -t llmlora:train .

    # Build inference image
    docker build -f Dockerfile.inference -t llmlora:inference .

    # Build Ollama image
    docker build -f Dockerfile.ollama -t llmlora:ollama .

Using Docker Compose
~~~~~~~~~~~~~~~~~~~~~

Deploy the complete stack with Docker Compose:

.. code-block:: yaml

    # docker-compose.yml
    version: '3.8'

    services:
      # Training service
      training:
        build:
          context: .
          dockerfile: Dockerfile.train
        image: llmlora:train
        volumes:
          - ./data:/app/data
          - ./models:/app/models
          - ./configs:/app/configs
        environment:
          - CUDA_VISIBLE_DEVICES=0
          - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
        command: ["train_model", "--config-name=production"]
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]

      # Inference service
      inference:
        build:
          context: .
          dockerfile: Dockerfile.inference
        image: llmlora:inference
        ports:
          - "8000:8000"
        volumes:
          - ./models:/app/models
        environment:
          - MODEL_PATH=/app/models/trained_model
          - MAX_BATCH_SIZE=4
        depends_on:
          - training
        restart: unless-stopped

      # Ollama service
      ollama:
        build:
          context: .
          dockerfile: Dockerfile.ollama
        image: llmlora:ollama
        ports:
          - "11434:11434"
        volumes:
          - ./models:/models
          - ollama_data:/root/.ollama
        environment:
          - OLLAMA_HOST=0.0.0.0
          - OLLAMA_MODELS=/models
        restart: unless-stopped

      # Redis for caching
      redis:
        image: redis:7-alpine
        ports:
          - "6379:6379"
        volumes:
          - redis_data:/data
        restart: unless-stopped

      # Monitoring
      prometheus:
        image: prom/prometheus
        ports:
          - "9090:9090"
        volumes:
          - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
          - prometheus_data:/prometheus
        restart: unless-stopped

    volumes:
      ollama_data:
      redis_data:
      prometheus_data:

Ollama Integration
------------------

Model Conversion for Ollama
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convert trained models to Ollama-compatible format:

.. code-block:: bash

    # Convert model to GGUF format (required for Ollama)
    python main.py convert_to_gguf \\
        --model_path=models/trained_model \\
        --output_path=models/ollama/model.gguf \\
        --quantize=true \\
        --qtype=q4_1

Creating Ollama Model Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create Ollama model configuration:

.. code-block:: bash

    # Create Modelfile for Ollama
    cat > models/ollama/Modelfile << EOF
    FROM ./model.gguf

    # Set model parameters
    PARAMETER temperature 0.7
    PARAMETER top_p 0.9
    PARAMETER top_k 40
    PARAMETER repeat_penalty 1.1

    # Set system prompt
    SYSTEM """
    You are a helpful AI assistant trained on domain-specific data.
    Provide accurate, helpful, and contextually appropriate responses.
    """

    # Set template
    TEMPLATE """{{ if .System }}{{ .System }}{{ end }}{{ if .Prompt }}User: {{ .Prompt }}
    Assistant: {{ end }}"""
    EOF

Registering Models with Ollama
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Register the converted model with Ollama:

.. code-block:: bash

    # Navigate to model directory
    cd models/ollama/

    # Create model in Ollama
    ollama create my-custom-model -f Modelfile

    # Verify model creation
    ollama list

    # Test model
    ollama run my-custom-model "Hello, how are you?"

Ollama API Integration
~~~~~~~~~~~~~~~~~~~~~~

Use the Ollama API for inference:

.. code-block:: python

    import requests
    import json

    class OllamaClient:
        def __init__(self, base_url="http://localhost:11434"):
            self.base_url = base_url

        def generate(self, model_name, prompt, stream=False):
            url = f"{self.base_url}/api/generate"

            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": stream
            }

            response = requests.post(url, json=payload)

            if stream:
                for line in response.iter_lines():
                    if line:
                        yield json.loads(line)
            else:
                return response.json()

        def chat(self, model_name, messages):
            url = f"{self.base_url}/api/chat"

            payload = {
                "model": model_name,
                "messages": messages
            }

            response = requests.post(url, json=payload)
            return response.json()

    # Usage example
    client = OllamaClient()

    # Generate text
    result = client.generate("my-custom-model", "Explain quantum computing")
    print(result['response'])

    # Chat interface
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]
    chat_result = client.chat("my-custom-model", messages)
    print(chat_result['message']['content'])

Deployment Configurations
-------------------------

Development Environment
~~~~~~~~~~~~~~~~~~~~~~~

Quick development setup:

.. code-block:: yaml

    # docker-compose.dev.yml
    version: '3.8'

    services:
      dev:
        build:
          context: .
          dockerfile: Dockerfile.train
        volumes:
          - .:/app
          - ./data:/app/data
          - ./models:/app/models
        environment:
          - PYTHONPATH=/app
        stdin_open: true
        tty: true
        command: bash
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  count: 1
                  capabilities: [gpu]

.. code-block:: bash

    # Start development environment
    docker-compose -f docker-compose.dev.yml up -d

    # Access development container
    docker-compose -f docker-compose.dev.yml exec dev bash

    # Train models interactively
    python main.py train_model --config-name=dev_config

Production Environment
~~~~~~~~~~~~~~~~~~~~~~

Production-ready deployment:

.. code-block:: yaml

    # docker-compose.prod.yml
    version: '3.8'

    services:
      # Load balancer
      nginx:
        image: nginx:alpine
        ports:
          - "80:80"
          - "443:443"
        volumes:
          - ./nginx/nginx.conf:/etc/nginx/nginx.conf
          - ./ssl:/etc/nginx/ssl
        depends_on:
          - inference-1
          - inference-2
        restart: always

      # Multiple inference instances
      inference-1:
        image: llmlora:inference
        environment:
          - MODEL_PATH=/app/models/production_model
          - WORKER_ID=1
        volumes:
          - ./models:/app/models
        restart: always

      inference-2:
        image: llmlora:inference
        environment:
          - MODEL_PATH=/app/models/production_model
          - WORKER_ID=2
        volumes:
          - ./models:/app/models
        restart: always

      # Ollama service with persistence
      ollama-prod:
        image: llmlora:ollama
        ports:
          - "11434:11434"
        volumes:
          - ollama_models:/models
          - ollama_data:/root/.ollama
        environment:
          - OLLAMA_HOST=0.0.0.0
          - OLLAMA_NUM_PARALLEL=4
        restart: always

      # Database for logging
      postgres:
        image: postgres:15
        environment:
          - POSTGRES_DB=llmlora
          - POSTGRES_USER=llmlora
          - POSTGRES_PASSWORD=secure_password
        volumes:
          - postgres_data:/var/lib/postgresql/data
        restart: always

      # Monitoring stack
      grafana:
        image: grafana/grafana
        ports:
          - "3000:3000"
        volumes:
          - grafana_data:/var/lib/grafana
          - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
        environment:
          - GF_SECURITY_ADMIN_PASSWORD=admin_password
        restart: always

    volumes:
      ollama_models:
      ollama_data:
      postgres_data:
      grafana_data:

Kubernetes Deployment
~~~~~~~~~~~~~~~~~~~~~~

Deploy on Kubernetes cluster:

.. code-block:: yaml

    # k8s/deployment.yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: llmlora-inference
      labels:
        app: llmlora-inference
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: llmlora-inference
      template:
        metadata:
          labels:
            app: llmlora-inference
        spec:
          containers:
          - name: inference
            image: llmlora:inference
            ports:
            - containerPort: 8000
            env:
            - name: MODEL_PATH
              value: "/app/models/production_model"
            volumeMounts:
            - name: models
              mountPath: /app/models
            resources:
              requests:
                memory: "4Gi"
                cpu: "2"
              limits:
                memory: "8Gi"
                cpu: "4"
          volumes:
          - name: models
            persistentVolumeClaim:
              claimName: models-pvc

    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: llmlora-service
    spec:
      selector:
        app: llmlora-inference
      ports:
      - protocol: TCP
        port: 80
        targetPort: 8000
      type: LoadBalancer

Monitoring and Logging
----------------------

Application Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor container performance and model metrics:

.. code-block:: python

    # monitoring/metrics.py
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    import time
    import psutil

    # Define metrics
    REQUEST_COUNT = Counter('llmlora_requests_total', 'Total requests')
    REQUEST_LATENCY = Histogram('llmlora_request_duration_seconds', 'Request latency')
    MODEL_MEMORY = Gauge('llmlora_model_memory_bytes', 'Model memory usage')
    GPU_UTILIZATION = Gauge('llmlora_gpu_utilization_percent', 'GPU utilization')

    class MetricsCollector:
        def __init__(self):
            # Start metrics server
            start_http_server(8080)

        def record_request(self, latency):
            REQUEST_COUNT.inc()
            REQUEST_LATENCY.observe(latency)

        def update_system_metrics(self):
            # CPU and memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            MODEL_MEMORY.set(memory.used)

            # GPU metrics (if available)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    GPU_UTILIZATION.set(gpus[0].load * 100)
            except ImportError:
                pass

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

Configure structured logging:

.. code-block:: python

    # logging_config.py
    import logging
    import json
    from datetime import datetime

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }

            # Add extra fields
            if hasattr(record, 'user_id'):
                log_entry['user_id'] = record.user_id
            if hasattr(record, 'request_id'):
                log_entry['request_id'] = record.request_id

            return json.dumps(log_entry)

    def setup_logging():
        # Configure root logger
        logging.basicConfig(level=logging.INFO)

        # Add JSON formatter for container logs
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())

        logger = logging.getLogger('llmlora')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        return logger

Health Checks
~~~~~~~~~~~~~

Implement health check endpoints:

.. code-block:: python

    # health.py
    from fastapi import FastAPI
    import torch
    import psutil

    app = FastAPI()

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }

    @app.get("/health/detailed")
    async def detailed_health():
        return {
            "status": "healthy",
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

    @app.get("/health/ready")
    async def readiness_check():
        # Check if model is loaded and ready
        try:
            # Perform a quick inference test
            test_inference()
            return {"status": "ready"}
        except Exception as e:
            return {"status": "not_ready", "error": str(e)}

Scaling Strategies
------------------

Horizontal Scaling
~~~~~~~~~~~~~~~~~~

Scale inference services based on demand:

.. code-block:: yaml

    # docker-compose.scale.yml
    version: '3.8'

    services:
      inference:
        image: llmlora:inference
        environment:
          - MODEL_PATH=/app/models/production_model
        volumes:
          - ./models:/app/models
        deploy:
          replicas: 5
          update_config:
            parallelism: 2
            delay: 30s
          restart_policy:
            condition: on-failure
            max_attempts: 3
          resources:
            limits:
              memory: 4G
            reservations:
              memory: 2G

.. code-block:: bash

    # Scale services
    docker-compose -f docker-compose.scale.yml up --scale inference=10

Auto-scaling with Docker Swarm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Initialize swarm
    docker swarm init

    # Deploy stack with auto-scaling
    docker stack deploy -c docker-compose.scale.yml llmlora

    # Scale based on CPU usage
    docker service update --replicas-max-per-node=2 llmlora_inference

Load Balancing
~~~~~~~~~~~~~~

Configure Nginx for load balancing:

.. code-block:: nginx

    # nginx/nginx.conf
    upstream inference_backend {
        least_conn;
        server inference-1:8000 max_fails=3 fail_timeout=30s;
        server inference-2:8000 max_fails=3 fail_timeout=30s;
        server inference-3:8000 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;

        location / {
            proxy_pass http://inference_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;

            # Buffer settings
            proxy_buffering on;
            proxy_buffer_size 8k;
            proxy_buffers 16 8k;
        }
    }

Security Considerations
-----------------------

Container Security
~~~~~~~~~~~~~~~~~~

Implement security best practices:

.. code-block:: dockerfile

    # Secure Dockerfile
    FROM python:3.9-slim

    # Create non-root user
    RUN groupadd -r llmlora && useradd -r -g llmlora llmlora

    # Install dependencies
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY --chown=llmlora:llmlora . /app
    WORKDIR /app

    # Switch to non-root user
    USER llmlora

    # Set security options
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1

    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
        CMD python health_check.py

    CMD ["python", "server.py"]

Network Security
~~~~~~~~~~~~~~~~

Configure secure networking:

.. code-block:: yaml

    # docker-compose.secure.yml
    version: '3.8'

    services:
      inference:
        image: llmlora:inference
        networks:
          - internal
        # Don't expose ports directly

      nginx:
        image: nginx:alpine
        ports:
          - "443:443"  # Only HTTPS
        networks:
          - internal
          - external
        volumes:
          - ./ssl:/etc/nginx/ssl:ro

    networks:
      internal:
        driver: bridge
        internal: true
      external:
        driver: bridge

Secrets Management
~~~~~~~~~~~~~~~~~~

Use Docker secrets for sensitive data:

.. code-block:: bash

    # Create secrets
    echo "database_password" | docker secret create db_password -
    echo "api_key" | docker secret create api_key -

.. code-block:: yaml

    # docker-compose.secrets.yml
    services:
      inference:
        image: llmlora:inference
        secrets:
          - db_password
          - api_key
        environment:
          - DB_PASSWORD_FILE=/run/secrets/db_password
          - API_KEY_FILE=/run/secrets/api_key

    secrets:
      db_password:
        external: true
      api_key:
        external: true

Deployment Automation
---------------------

CI/CD Pipeline
~~~~~~~~~~~~~~

Automate deployment with GitHub Actions:

.. code-block:: yaml

    # .github/workflows/deploy.yml
    name: Deploy to Production

    on:
      push:
        branches: [main]

    jobs:
      build-and-deploy:
        runs-on: ubuntu-latest

        steps:
        - uses: actions/checkout@v3

        - name: Build Docker images
          run: |
            docker build -f Dockerfile.inference -t llmlora:inference .
            docker build -f Dockerfile.ollama -t llmlora:ollama .

        - name: Run tests
          run: |
            docker run --rm llmlora:inference python -m pytest tests/

        - name: Deploy to staging
          if: github.ref == 'refs/heads/develop'
          run: |
            docker-compose -f docker-compose.staging.yml up -d

        - name: Deploy to production
          if: github.ref == 'refs/heads/main'
          run: |
            docker-compose -f docker-compose.prod.yml up -d

Backup and Recovery
~~~~~~~~~~~~~~~~~~~

Implement backup strategies:

.. code-block:: bash

    #!/bin/bash
    # backup.sh

    # Backup models
    docker run --rm -v llmlora_models:/data -v $(pwd):/backup alpine \\
        tar czf /backup/models-backup-$(date +%Y%m%d).tar.gz /data

    # Backup database
    docker exec llmlora_postgres pg_dump -U llmlora llmlora > \\
        backup/db-backup-$(date +%Y%m%d).sql

    # Upload to cloud storage
    aws s3 cp backup/ s3://llmlora-backups/$(date +%Y%m%d)/ --recursive

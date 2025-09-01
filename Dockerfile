FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.11 \
        python3.11-distutils \
        python3.11-dev \
        python3.11-venv \
        vim \
        git \
        build-essential \
        cmake  \
        curl \
        wget \
        jq  \
        libcurl4-openssl-dev \
        && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN python3.11 -m ensurepip && python3.11 -m pip install --upgrade pip

RUN pip install --upgrade --ignore-installed wheel==0.45.1

WORKDIR /llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git . && \
    cmake -B build && \
    cmake --build build --config Release

# Install RKLLM dependencies
RUN pip install rknn-toolkit2 -i https://mirrors.aliyun.com/pypi/simple

# Download and install RKLLM toolkit
WORKDIR /tmp
RUN wget -q https://github.com/airockchip/rknn-llm/releases/download/v1.0.11/rkllm_toolkit-1.0.11-cp311-cp311-linux_x86_64.whl || \
    wget -q https://github.com/airockchip/rknn-llm/releases/download/v1.0.10/rkllm_toolkit-1.0.10-cp311-cp311-linux_x86_64.whl || \
    wget -q https://github.com/airockchip/rknn-llm/releases/download/v1.0.9/rkllm_toolkit-1.0.9-cp311-cp311-linux_x86_64.whl || \
    echo "No RKLLM toolkit wheel found, trying source installation"

# Install the downloaded wheel file (try multiple versions)
RUN pip install rkllm_toolkit-*.whl 2>/dev/null || echo "Warning: RKLLM toolkit wheel installation failed"

# Fallback: Try to install from source if wheel installation failed
RUN if ! python -c "import rkllm" 2>/dev/null; then \
        echo "Attempting source installation of RKLLM toolkit..." && \
        git clone https://github.com/airockchip/rknn-llm.git --depth 1 || true && \
        if [ -d "rknn-llm/rkllm-toolkit" ]; then \
            cd rknn-llm/rkllm-toolkit && \
            pip install . 2>/dev/null || echo "Source installation also failed"; \
        fi && \
        rm -rf rknn-llm; \
    fi

# Clean up downloaded files
RUN rm -f rkllm_toolkit-*.whl

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root --only main

# Create symlink for llama.cpp so training code can find it at expected relative path
RUN ln -s /llama.cpp ./llama.cpp

# Create symlink for quantize executable to match Windows naming convention
RUN ln -s /llama.cpp/build/bin/llama-quantize /app/llama.cpp/llama-quantize.exe

COPY training_model ./training_model
COPY testing_model ./testing_model
COPY evaluation ./evaluation
COPY data ./data
COPY conf ./conf
COPY main.py .
COPY run_pipeline.sh .

RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]

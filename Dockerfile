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
    apt-transport-https \
    ca-certificates \
    gnupg \
        && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN python3.11 -m ensurepip && python3.11 -m pip install --upgrade pip

RUN pip install --upgrade --ignore-installed wheel==0.45.1

# Install Docker CLI so the training container can control Docker to run the RKLLM converter
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
        > /etc/apt/sources.list.d/docker.list && \
    apt-get update && apt-get install -y docker-ce-cli && rm -rf /var/lib/apt/lists/*

WORKDIR /llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git . && \
    cmake -B build && \
    cmake --build build --config Release


WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    pip install pyyaml && \
    poetry lock --no-interaction --no-ansi || true && \
    poetry install --no-root --only main --no-interaction --no-ansi

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

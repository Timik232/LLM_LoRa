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

WORKDIR /tmp
RUN git clone -b release-v1.2.1 https://github.com/airockchip/rknn-llm.git && \
    pip install rknn-llm/rkllm-toolkit/rkllm_toolkit-1.2.1-cp311-cp311-linux_x86_64.whl && \
    rm -rf rknn-llm

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry install --no-root --only main

COPY training_model ./training_model
COPY testing_model ./testing_model
COPY evaluation ./evaluation
COPY data ./data
COPY conf ./conf
COPY main.py .
COPY run_pipeline.sh .

RUN chmod +x run_pipeline.sh

CMD ["./run_pipeline.sh"]
#CMD python -m training_model

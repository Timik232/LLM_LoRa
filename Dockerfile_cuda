FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

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
        cmake && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

RUN python3.11 -m ensurepip && python3.11 -m pip install --upgrade pip

RUN pip install --upgrade --ignore-installed wheel==0.45.1

WORKDIR /training_model
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root --only main

COPY training_model ./training_model
COPY testing_model ./testing_model
COPY data ./data
COPY main.py .

WORKDIR /llama.cpp
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    git clone https://github.com/ggml-org/llama.cpp.git . && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH cmake -B build -DGGML_CUDA=ON && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH cmake --build build --config Release

WORKDIR /training_model

CMD python -m training_model

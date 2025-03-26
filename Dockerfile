FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    vim \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /usr/bin/python3 /usr/bin/python


WORKDIR /training_model
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-root

COPY training_model ./training_model
COPY testing_model ./testing_model
COPY data ./data
COPY main.py .

WORKDIR /llama.cpp
RUN git clone https://github.com/ggml-org/llama.cpp.git . && \
    cmake -B build && \
    cmake --build build --config Release

WORKDIR /training_model

CMD python -m training_model

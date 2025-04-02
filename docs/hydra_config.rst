.. _configuration-reference:

Hydra Configuration Reference
=================================

This document describes the Hydra configuration structure and parameters used for model training and management.

Configuration Overview
----------------------------

The configuration is organized into several sections controlling different aspects of the training pipeline:

.. code-block:: yaml

    defaults:
      - _self_
      - override hydra/job_logging: disabled
      - override hydra/hydra_logging: disabled

    model:
      # Model architecture and training parameters
      # ...

    training:
      # Training hyperparameters
      # ...

    paths:
      # Directory paths and system locations
      # ...

Main Configuration Sections
---------------------------

Defaults Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Defaults Overrides
    :widths: 30 70
    :header-rows: 1

    * - Key
      - Description
    * - ``_self_``
      - Includes current config in composition hierarchy
    * - ``override hydra/job_logging``
      - Disables Hydra's default job logging
    * - ``override hydra/hydra_logging``
      - Disables Hydra's internal system logging

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - model_name
      - Base model identifier from Hugging Face Hub
      - "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
    * - lora_r
      - LoRA rank dimension
      - 16
    * - lora_alpha
      - LoRA alpha scaling factor
      - 32
    * - qtype
      - Quantization type for GGUF conversion
      - "q4_1"
    * - torch_dtype
      - Base model dtype (float16/float32)
      - "float16"

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Training Parameters (Key Items)
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - per_device_train_batch_size
      - Batch size per GPU
      - 1
    * - gradient_accumulation_steps
      - Number of update steps before backward pass
      - 4
    * - learning_rate
      - Initial learning rate
      - 2e-5
    * - max_seq_length
      - Maximum input sequence length
      - 2048
    * - gradient_checkpointing
      - Enable memory-efficient training
      - true

Paths Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Path Directories
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Example
    * - data_dir
      - Input dataset directory
      - "data"
    * - output_dir
      - Trained model output directory
      - "models"
    * - llama_cpp_dir
      - Path to llama.cpp installation
      - "../llama.cpp"
    * - quantized_path
      - llama.cpp quantizer executable path
      - "build/bin/llama-quantize"

Training Pipeline Workflow
--------------------------

The complete training process follows these stages:

1. **Initialization**
    - Configure logging and environment
    - Load base model with 4-bit quantization
    - Prepare tokenizer with custom padding

2. **Data Preparation**
    - Load dataset from JSON files
    - Generate chat-formatted prompts
    - Tokenize with sequence length truncation

3. **Model Training**
    - Apply LoRA configuration to base model
    - Train using either SFTTrainer or GRPO
    - Merge adapter weights with base model

4. **Model Conversion**
    - Convert merged model to GGUF format
    - Quantize using llama.cpp tools
    - Save final weights to output directory

.. code-block:: python

    # Simplified pipeline flow
    def train_pipeline(cfg):
        steps = train(cfg)
        with TemporaryDirectory() as tmp_dir:
            model_merge_for_converting(cfg, steps, tmp_dir)
            convert_to_gguf(tmp_dir, ...)
            quantize_model(...)
            copy_final_weights(...)

Important Implementation Notes
------------------------------

LoRA Configuration
~~~~~~~~~~~~~~~~~~

The model uses Low-Rank Adaptation with these key settings:

.. list-table:: LoRA Parameters
    :widths: 30 50 20
    :header-rows: 1

    * - Module
      - Target Layers
      - Parameters
    * - peft.LoraConfig
      - proj layers (q_proj, v_proj, etc)
      - r=16, alpha=32
    * - Modules to Save
      - lm_head
      - -

Quantization Setup
~~~~~~~~~~~~~~~~~~

The system supports two-stage quantization:

1. **Training Quantization**
    - 4-bit NFQuant via BitsAndBytes
    - Compatible dtype: float16

2. **Post-Training Quantization**
    - GGUF conversion with llama.cpp
    - Supported types: q4_0, q4_1, etc

.. note::
    For optimal performance, ensure llama.cpp is compiled with CUDA support
    when quantizing on GPU systems.

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

Custom logging setup includes:

- Hydra logging disabled for cleaner outputs
- W&B integration for experiment tracking
- Custom logging levels via ``logging_config.py``

.. warning::
    The ``hf_token`` field must be updated with a valid Hugging Face token
    when using private models or datasets.

Environment Requirements
------------------------

The system requires these key dependencies:

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- Hydra 1.3+
- llama.cpp (latest version)

Full configuration schema available in ``conf/config.yaml``

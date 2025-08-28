.. _configuration-reference:

Hydra Configuration Reference
=================================

This document describes the Hydra configuration structure and parameters used for model training and management in the LLM-LoRA framework. The configuration supports multiple training methods including SFT, DPO, GRPO, and various deployment formats.

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
      model_name: "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it"
      lora_r: 16
      lora_alpha: 32
      torch_dtype: "float16"
      qtype: "q4_1"

    training:
      # Training hyperparameters and methods
      training_method: "sft"  # Options: sft, dpo, grpo
      per_device_train_batch_size: 1
      gradient_accumulation_steps: 4
      learning_rate: 2e-5
      max_seq_length: 2048
      gradient_checkpointing: true

    paths:
      # Directory paths and system locations
      data_dir: "data"
      output_dir: "models"
      llama_cpp_dir: "../llama.cpp"
      quantized_path: "build/bin/llama-quantize"

    conversion:
      # Model conversion and quantization options
      convert_to_gguf: true
      convert_to_rkllm: false
      quantize_model: true

    fire_cli:
      # Fire CLI configuration
      enable_cli: true
      available_commands: ["train_model", "evaluate_model", "convert_to_gguf", "convert_to_rkllm"]

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
    * - training_method
      - Training approach (sft/dpo/grpo)
      - "sft"
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

Training Methods Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Method-Specific Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Usage
    * - training_method: "sft"
      - Supervised Fine-tuning with instruction-response pairs
      - Standard fine-tuning
    * - training_method: "dpo"
      - Direct Preference Optimization for alignment
      - Preference-based training
    * - training_method: "grpo"
      - Group Relative Policy Optimization
      - Advanced preference learning
    * - beta (DPO/GRPO)
      - Temperature parameter for preference learning
      - 0.1-0.5 typical range
    * - preference_dataset
      - Dataset path for preference training methods
      - Required for DPO/GRPO

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
    * - rkllm_toolkit_path
      - Path to RKLLM conversion toolkit
      - "path/to/rkllm-toolkit"

Conversion Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Conversion Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - convert_to_gguf
      - Enable GGUF format conversion
      - true
    * - convert_to_rkllm
      - Enable RKLLM format conversion for Rockchip NPU
      - false
    * - quantize_model
      - Apply post-training quantization
      - true
    * - gguf_quantization_type
      - GGUF quantization method (q4_0, q4_1, q8_0)
      - "q4_1"
    * - rkllm_target_platform
      - Target Rockchip platform (rk3588, rk3576)
      - "rk3588"

Fire CLI Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: CLI Interface Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enable_cli
      - Enable Fire CLI interface
      - true
    * - available_commands
      - List of available CLI commands
      - ["train_model", "evaluate_model", "convert_to_gguf"]
    * - cli_help_enabled
      - Enable automatic help generation
      - true

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
    - Train using selected method (SFT/DPO/GRPO)
    - Merge adapter weights with base model

4. **Model Conversion**
    - Convert merged model to GGUF format (optional)
    - Convert to RKLLM format for Rockchip NPU (optional)
    - Quantize using llama.cpp tools
    - Save final weights to output directory

5. **Evaluation**
    - Run model evaluation using configured metrics
    - Generate evaluation reports
    - Compare with baseline models

.. code-block:: python

    # Simplified pipeline flow with new methods
    def train_pipeline(cfg):
        if cfg.training.training_method == "sft":
            steps = sft_train(cfg)
        elif cfg.training.training_method == "dpo":
            steps = dpo_train(cfg)
        elif cfg.training.training_method == "grpo":
            steps = grpo_train(cfg)

        with TemporaryDirectory() as tmp_dir:
            model_merge_for_converting(cfg, steps, tmp_dir)

            if cfg.conversion.convert_to_gguf:
                convert_to_gguf(tmp_dir, cfg)

            if cfg.conversion.convert_to_rkllm:
                convert_to_rkllm(tmp_dir, cfg)

            if cfg.conversion.quantize_model:
                quantize_model(cfg)

            copy_final_weights(cfg)
Important Implementation Notes
------------------------------

Training Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports three training approaches:

.. list-table:: Training Method Comparison
    :widths: 20 40 40
    :header-rows: 1

    * - Method
      - Use Case
      - Key Benefits
    * - SFT
      - Instruction following, task-specific adaptation
      - Simple, stable, well-established
    * - DPO
      - Human preference alignment without RL
      - Direct optimization, no reward model needed
    * - GRPO
      - Advanced preference learning
      - Group-based optimization, improved alignment

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
      - lm_head, embed_tokens
      - -

Model Conversion Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports multiple output formats:

.. list-table:: Conversion Options
    :widths: 25 50 25
    :header-rows: 1

    * - Format
      - Use Case
      - Platform
    * - PyTorch
      - Development, fine-tuning
      - GPU/CPU
    * - GGUF
      - CPU inference, llama.cpp
      - CPU/GPU
    * - RKLLM
      - NPU acceleration
      - Rockchip devices

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

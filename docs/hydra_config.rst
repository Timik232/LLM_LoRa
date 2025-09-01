.. _configuration-reference:

Hydra Configuration Reference
=================================

This document describes the Hydra configuration structure and parameters used for model training and management in the LLM-LoRA framework. The configuration supports multiple training methods including SFT, DPO, GRPO, and various deployment formats.

Configuration Overview
----------------------------

The configuration is organized into several sections controlling different aspects of the training pipeline. The framework uses **Hydra Config Groups** for environment-specific settings, providing automatic configuration management without manual file editing.

.. code-block:: yaml

    defaults:
      - _self_
      - override hydra/job_logging: disabled
      - override hydra/hydra_logging: disabled
      - environment: docker  # Default environment config (use 'local' for Windows development)

    model:
      # Model architecture and training parameters
      model_name: "RefalMachine/RuadaptQwen3-4B-Instruct"
      new_model: "4b-chat-vika"
      torch_dtype: "bfloat16"
      attn_implementation: "eager"
      train_steps: 60
      outfile: "custom-model.gguf"

      # Quantization and GGUF conversion settings
      quant:
        enabled: true  # Enable GGUF conversion in training pipeline (NEW)
        qtype: "q4_1"
        gguf_dir: "${paths.gguf_directory}"
        use_8bit: false

      # RKLLM conversion for Rockchip NPU
      rkllm:
        enabled: true
        target_platform: "rk3588"
        quantization: "w8a8"
        output_dir: "rkllm_models"
        do_parallelize: false
        hybrid_quantization: false
        num_npu_core: 1

      # LoRA configuration
      lora:
        r: 16
        alpha: 32
        dropout: 0.1

      model_type: "auto"  # gemma, gemma3n, or auto

    training:
      # Training hyperparameters and method selection
      use_sft: true      # Enable Supervised Fine-Tuning
      use_grpo: false    # Enable Group Relative Policy Optimization
      use_dpo: false     # Enable Direct Preference Optimization

      per_device_train_batch_size: 1
      gradient_accumulation_steps: 6
      num_train_epochs: 1.1
      learning_rate: 2e-5
      max_seq_length: 2048
      gradient_checkpointing: true
      fp16: true
      bf16: false

    data_preparation:
      # Data preparation method selection
      method: "classic"  # Options: "classic" | "game"
      # classic: Simple instruction/output format with basic prompting
      # game: Complex game format with history, actions, and system messages

    paths:
      # Directory paths and system locations
      data_dir: "data"
      output_dir: "models"
      train_data: "dog_dataset.json"
      llama_cpp_dir: "llama.cpp"
      venv_python_path: "T:/projects/LLM_LoRa/venv/Scripts/python.exe"
      quantized_path: "llama-quantize.exe"
      gguf_directory: "custom-model"
      final_weights_path: "models"

    logging:
      # Experiment tracking configuration
      logging_backend: "mlflow"  # Options: wandb, mlflow, none
      wandb:
        project_name: "Gemma vika train"
        anonymous: "allow"
      mlflow:
        experiment_name: "vika-experiment"
        tracking_uri: "http://mlflow:5000"

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
    * - ``environment``
      - Selects environment-specific configuration (docker/local)

Environment Configuration Groups
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework uses Hydra Config Groups to automatically handle environment-specific settings, eliminating the need to manually comment/uncomment configuration lines.

**Available Environment Configurations:**

.. list-table:: Environment Config Groups
    :widths: 25 30 45
    :header-rows: 1

    * - Environment
      - Config File
      - Description
    * - docker (default)
      - ``conf/environment/docker.yaml``
      - Docker/production environment with system python
    * - local
      - ``conf/environment/local.yaml``
      - Local Windows development with virtual environment

**Usage Examples:**

.. code-block:: bash

    # Use Docker/production environment (default)
    python main.py

    # Use local development environment
    python main.py environment=local

    # Combine with other overrides
    python main.py environment=local model.train_steps=100

**Environment Config Files:**

.. code-block:: yaml

    # conf/environment/docker.yaml
    # @package paths
    # Docker/Production environment configuration
    venv_python_path: "python"  # Use system python in Docker containers

.. code-block:: yaml

    # conf/environment/local.yaml
    # @package paths
    # Local Windows development environment configuration
    venv_python_path: "T:/projects/LLM_LoRa/venv/Scripts/python.exe"

The ``@package paths`` directive ensures these settings only override the ``paths`` section of the main configuration, keeping files small and focused on environment differences.

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
      - "RefalMachine/RuadaptQwen3-4B-Instruct"
    * - new_model
      - Output name for the fine-tuned model
      - "4b-chat-vika"
    * - torch_dtype
      - Model precision (float16/bfloat16/float32)
      - "bfloat16"
    * - attn_implementation
      - Attention implementation (eager/flash_attention_2)
      - "eager"
    * - train_steps
      - Number of training steps
      - 60
    * - outfile
      - Output filename for GGUF conversion
      - "custom-model.gguf"
    * - model_type
      - Model architecture type (auto/gemma/gemma3n)
      - "auto"

Model Quantization Configuration (model.quant)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Quantization Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - **NEW**: Enable GGUF conversion in training pipeline
      - true
    * - qtype
      - Quantization type for GGUF conversion (q4_0, q4_1, q8_0)
      - "q4_1"
    * - gguf_dir
      - Output directory for GGUF files
      - "${paths.gguf_directory}"
    * - use_8bit
      - Use 8-bit quantization instead of 4-bit
      - false

The ``enabled`` parameter allows you to disable GGUF conversion for Docker deployments or when only the HuggingFace model format is needed:

.. code-block:: bash

    # Disable GGUF conversion
    python main.py model.quant.enabled=false

RKLLM Configuration (model.rkllm)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: RKLLM Conversion Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable RKLLM conversion for Rockchip NPU
      - true
    * - target_platform
      - Target Rockchip platform (rk3588, rk3576, etc.)
      - "rk3588"
    * - quantization
      - RKLLM quantization type (w8a8, w4a16, w4a16_g128)
      - "w8a8"
    * - output_dir
      - Output directory for RKLLM models
      - "rkllm_models"
    * - do_parallelize
      - Enable model parallelization for larger models
      - false
    * - hybrid_quantization
      - Enable hybrid quantization
      - false
    * - num_npu_core
      - Number of NPU cores to use (1-3)
      - 1

LoRA Configuration (model.lora)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: LoRA Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - r
      - LoRA rank dimension
      - 16
    * - alpha
      - LoRA alpha scaling factor
      - 32
    * - dropout
      - LoRA dropout rate
      - 0.1

Data Preparation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Data Preparation Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - method
      - Data preparation method selection
      - "classic"
    * - method: "classic"
      - Simple instruction/output format for standard fine-tuning
      - Expects "instruction" and "output" fields
    * - method: "game"
      - Complex game format with conversation history and actions
      - Expects "prompt" and "answer" fields with structured data

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Training Parameters (Key Items)
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - use_sft
      - Enable Supervised Fine-Tuning
      - true
    * - use_grpo
      - Enable Group Relative Policy Optimization
      - false
    * - use_dpo
      - Enable Direct Preference Optimization
      - false
    * - per_device_train_batch_size
      - Batch size per GPU
      - 1
    * - gradient_accumulation_steps
      - Number of update steps before backward pass
      - 6
    * - num_train_epochs
      - Number of training epochs
      - 1.1
    * - learning_rate
      - Initial learning rate
      - 2e-5
    * - max_seq_length
      - Maximum input sequence length
      - 2048
    * - gradient_checkpointing
      - Enable memory-efficient training
      - true
    * - fp16
      - Use 16-bit floating point precision
      - true
    * - bf16
      - Use bfloat16 precision
      - false

Training Methods Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports three training methods that can be enabled independently:

.. list-table:: Training Method Flags
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Usage
    * - use_sft: true
      - Supervised Fine-tuning with instruction-response pairs
      - Standard fine-tuning approach
    * - use_grpo: true
      - Group Relative Policy Optimization
      - Advanced preference learning (requires preference data)
    * - use_dpo: true
      - Direct Preference Optimization for alignment
      - Preference-based training (requires preference data)

Multiple training methods can be chained together in a single pipeline run.

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
    * - venv_python_path
      - Python executable path (managed by environment configs)
      - Auto-configured via environment parameter

Model Conversion and Output Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework supports multiple model output formats controlled by configuration flags:

.. list-table:: Conversion Control Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter Path
      - Description
      - Default
    * - model.quant.enabled
      - Enable GGUF format conversion and quantization
      - true
    * - model.rkllm.enabled
      - Enable RKLLM format conversion for Rockchip NPU
      - true

**GGUF Conversion Control:**

When ``model.quant.enabled=true`` (default):
- Model is converted to GGUF format using llama.cpp
- Quantization is applied according to ``model.quant.qtype``
- Output is saved to ``model.quant.gguf_dir``

When ``model.quant.enabled=false``:
- GGUF conversion is skipped
- Only the merged HuggingFace model is saved
- Useful for Docker deployments or when GGUF is not needed

**RKLLM Conversion Control:**

When ``model.rkllm.enabled=true``:
- Additional RKLLM format is generated for Rockchip NPU
- Uses ``model.rkllm.target_platform`` and ``model.rkllm.quantization``
- Output is saved to ``model.rkllm.output_dir``

.. code-block:: bash

    # Examples of conversion control
    python main.py model.quant.enabled=false  # Skip GGUF conversion
    python main.py model.rkllm.enabled=false  # Skip RKLLM conversion
    python main.py model.quant.enabled=false model.rkllm.enabled=false  # Only HF model

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
      - [\"train\", \"test\", \"convert\", \"optimize\", \"pipeline\"]
    * - cli_help_enabled
      - Enable automatic help generation
      - true

Training Pipeline Workflow
--------------------------

The complete training process follows these stages:

1. **Initialization**
    - Configure logging backend (wandb/mlflow/none)
    - Load base model with 4-bit quantization
    - Prepare tokenizer with custom padding

2. **Data Preparation**
    - Load dataset from JSON files
    - Apply selected data preparation method (classic or game format)
    - Generate chat-formatted prompts based on method selection
    - Tokenize with sequence length truncation

3. **Model Training**
    - Apply LoRA configuration to base model
    - Train using enabled methods (SFT/DPO/GRPO can be chained)
    - Merge adapter weights with base model

4. **Model Conversion (Conditional)**
    - **HuggingFace Model**: Always saved to output directory
    - **GGUF Conversion**: Only if ``model.quant.enabled=true`` (default)
    - **RKLLM Conversion**: Only if ``model.rkllm.enabled=true``
    - **Quantization**: Applied during GGUF conversion using llama.cpp

5. **Evaluation** (Optional)
    - Run model evaluation using configured metrics
    - Generate evaluation reports
    - Compare with baseline models

.. code-block:: python

    # Updated pipeline flow with conditional conversion
    def train_pipeline(cfg):
        # Training phase (multiple methods can be chained)
        if cfg.training.use_sft:
            steps = sft_train(cfg)

        if cfg.training.use_grpo:
            steps = grpo_train(cfg)

        if cfg.training.use_dpo:
            steps = dpo_train(cfg)

        with TemporaryDirectory() as tmp_dir:
            # Always merge model
            model_merge_for_converting(cfg, steps, tmp_dir)

            # Conditional GGUF conversion (NEW LOGIC)
            if cfg.model.quant.get("enabled", True):
            # Use CLI for conversion instead\n            subprocess.run([\"python\", \"main.py\", \"convert\", \"--gguf=True\"])
                quantize_model(cfg)
                copy_data(quantized_file, cfg.model.quant.gguf_dir)
            else:
                # Alternative: copy merged HF model directly
                merged_output_dir = os.path.join(cfg.paths.output_dir, "merged_model")
                shutil.copytree(tmp_dir, merged_output_dir)

            # Conditional RKLLM conversion (always optional)
            if cfg.model.rkllm.get("enabled", False):
                convert_to_rkllm(tmp_dir, cfg)

**Docker Deployment Example:**

For containerized deployments where GGUF is not needed:

.. code-block:: bash

    # Run training without GGUF conversion
    docker-compose exec llm_training python main.py model.quant.enabled=false

This saves significant time and disk space in Docker environments where only the HuggingFace model format is required.
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

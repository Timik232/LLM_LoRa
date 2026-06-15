.. _configuration-reference:

Hydra Configuration Reference
=============================

This document describes the Hydra configuration structure and parameters used for model training and management in the LLM-LoRA framework. The configuration supports multiple training methods including SFT, DPO, GRPO, and various deployment formats.

Configuration Overview
----------------------

The configuration is organized into several sections controlling different aspects of the training pipeline. The framework uses **Hydra Config Groups** for environment-specific settings, providing automatic configuration management without manual file editing.

.. code-block:: yaml

    defaults:
      - _self_  # Add _self_ first to include the current config in composition
      - override hydra/job_logging: disabled  # Properly override Hydra's logging
      - override hydra/hydra_logging: disabled
    model:
      model_name: "AnatoliiPotapov/T-lite-instruct-0.1"
      new_model: "4b-chat-vika"
      torch_dtype: "float16"
      attn_implementation: "flash_attention_2"
      train_steps: 60
      outfile: "custom-model.gguf" # if you change this, change the model in run_pipeline
      quant:
        enabled: true
        convert_to_gguf: true # Enable GGUF conversion in training pipeline
        qtype: "q4_1"
        gguf_dir: "${paths.gguf_directory}" # if you change this, change in run_pipeline
        use_8bit: false
      rkllm:
        enabled: false # Enable RKLLM conversion
        target_platform: "rk3588" # Target Rockchip platform: rk3588, rk3576, etc.
        quantization: "w8a8" # RKLLM quantization: w8a8, w4a16, w4a16_g128
        output_dir: "rkllm_models" # Output directory for RKLLM models
        do_parallelize: false # Enable model parallelization for larger models
        hybrid_quantization: false # Enable hybrid quantization
        num_npu_core: 1 # Number of NPU cores to use (1-3)
      lora:
        r: 8
        alpha: 16
        dropout: 0.1
      model_type: "auto" # gemma or gemma3n or auto
    training:
      per_device_train_batch_size: 1
      per_device_eval_batch_size: 1
      gradient_accumulation_steps: 6
      num_train_epochs: 1.5
      eval_steps: 50
      logging_steps: 5
      warmup_steps: 10
      learning_rate: 2e-5
      fp16: false
      bf16: false
      weight_decay: 0.05
      max_seq_length: 2048
      optim: "paged_adamw_8bit"
      neftune_noise_alpha: 0
      gradient_checkpointing: true
      save_total_limit: 5
      load_best: false
      use_grpo: false
      use_sft: true
      use_dpo: false
      seed: 42
      adam_mini:
        enabled: true  # Enable Adam-mini optimizer (default: false, uses standard optimizer)
        # Standard optimizer parameters (recommended to use same values as AdamW)
        learning_rate: ${training.learning_rate}  # Inherit from main training config
        weight_decay: ${training.weight_decay}    # Inherit from main training config
        beta1: 0.9                               # Adam beta1 parameter
        beta2: 0.999                             # Adam beta2 parameter
        eps: 1.0e-8                              # Adam epsilon parameter
        # Transformer-specific parameters (auto-detected from model config if null)
        dim: null          # Hidden dimension (auto-detected from model.config.hidden_size)
        n_heads: null      # Number of attention heads (auto-detected from model.config.num_attention_heads)
        n_kv_heads: null   # Number of key-value heads (auto-detected from model.config.num_key_value_heads, optional)
        # Special optimization for small training runs
        use_single_lr_for_values: false  # Set to true for training steps <10k-20k to speed up initial convergence
    memory_management:
      # Memory optimization settings for efficient training pipeline
      aggressive_cleanup: true          # Enable aggressive memory cleanup between phases
      log_memory_usage: true           # Log memory usage at key points in training
    data_preparation:
      method: "game" # Options: "classic" | "game"
      # classic: Simple instruction/output format with basic prompting
      # game: Complex game format with history, actions, and system messages
    grpo:
      val_data: "test_ru.json"
      train_data: "dataset_ru.json"
      max_completion_length: None
      num_generations: 2
      use_cache: false
      # GRPO-specific hyperparameters
      epsilon: 0.2  # KL penalty coefficient for policy gradient
      beta: 0.01    # KL regularization parameter
      loss_type: "sigmoid"  # Loss function type: "sigmoid" or "hinge"
      temperature: 0.7      # Sampling temperature for generation
      top_k: 50            # Top-k sampling parameter
      top_p: 0.95          # Top-p (nucleus) sampling parameter
      do_sample: true      # Enable sampling during generation
      response_length: 256 # Maximum response length for completions
      # Reference model settings (optional - uses same model by default)
      use_ref_model: false  # Whether to use a separate reference model
      ref_model_name: null  # Reference model name (if different from base model)
    dpo:
      val_data: "dpo_test.json"
      train_data: "dpo_dataset.json"
      max_length: 2048
      max_prompt_length: 1024
      max_target_length: 1024
      # DPO-specific hyperparameters
      beta: 0.1            # KL penalty coefficient for DPO loss
      loss_type: "sigmoid" # Loss function type: "sigmoid" or "hinge"
      # Reference model settings (optional - uses same model by default)
      use_ref_model: false # Whether to use a separate reference model for DPO
      ref_model_name: null # Reference model name (if different from base model)
    paths:
      data_dir: "data"
      test_data: "test_ru.json"
      train_data: "dataset_ru.json"
      gguf_directory: custom-model
      merged_model_path: "merged_model_fp16"
      output_dir: "models" # if you change this, change in run_pipeline
      llama_cpp_dir: "llama.cpp"
      venv_python_path: "${paths.venv_python_path:}"
      #  "T:/lm-studio/models/game-model"
      final_weights_path: "models" # if you change this, change in run_pipeline
    #  quantized_path: "build/bin/llama-quantize" # for local run path may be different
      quantized_path: "llama-quantize.exe"
    other:
      cutoff_len: 2048
      hf_login: false
    environment:
      use_dotenv: false  # Set to true for local development, false for Docker/production
    testing:
      data_source_mode: "separate_validation"  # Options: "auto_split", "separate_validation", "separate_files"
      test_split_ratio: 0.05  # Used for auto_split mode
      val_data_file: "test_ru.json"  # Validation file used for separate_validation mode
      manual_lmstudio_test: false # use only for local run
      test: true
      test_dataset: "${paths.data_dir}/test_ru.json"
      output_test_file: "../test.json"
      prompt_for_deepeval: "test_prompt"
    logging:
      log_level: "INFO"  # Logging level: "DEBUG" or "INFO"
      logging_backend: "mlflow" #wand mflow none
      wandb:
        project_name: "Gemma vika train"
        anonymous: "allow"
      mlflow:
        experiment_name: "vika-experiment"
        tracking_uri: "http://localhost:5000"
        log_artifacts: false
    hydra:
      run:
        dir: .
      job:
        chdir: true  # Address future Hydra working dir change warning

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
      - "AnatoliiPotapov/T-lite-instruct-0.1"
    * - new_model
      - Output name for the fine-tuned model
      - "4b-chat-vika"
    * - torch_dtype
      - Model precision (float16/bfloat16/float32)
      - "float16"
    * - attn_implementation
      - Attention implementation (eager/flash_attention_2)
      - "flash_attention_2"
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
      - Enable quantization in training pipeline
      - true
    * - convert_to_gguf
      - Enable GGUF conversion in training pipeline
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

The ``enabled`` parameter allows you to disable quantization for Docker deployments or when only the HuggingFace model format is needed:

.. code-block:: bash

    # Disable quantization
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
      - false
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
      - 8
    * - alpha
      - LoRA alpha scaling factor
      - 16
    * - dropout
      - LoRA dropout rate
      - 0.1

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
    * - per_device_eval_batch_size
      - Batch size per GPU for evaluation
      - 1
    * - gradient_accumulation_steps
      - Number of update steps before backward pass
      - 6
    * - num_train_epochs
      - Number of training epochs
      - 1.5
    * - eval_steps
      - Evaluation steps interval
      - 50
    * - logging_steps
      - Logging steps interval
      - 5
    * - warmup_steps
      - Warmup steps for learning rate scheduler
      - 10
    * - learning_rate
      - Initial learning rate
      - 2e-5
    * - fp16
      - Use 16-bit floating point precision
      - false
    * - bf16
      - Use bfloat16 precision
      - false
    * - weight_decay
      - Weight decay for optimizer
      - 0.05
    * - max_seq_length
      - Maximum input sequence length
      - 2048
    * - optim
      - Optimizer type
      - "paged_adamw_8bit"
    * - neftune_noise_alpha
      - NEFTune noise alpha
      - 0
    * - gradient_checkpointing
      - Enable memory-efficient training
      - true
    * - save_total_limit
      - Limit on number of saved checkpoints
      - 5
    * - load_best
      - Load best model at end
      - false
    * - seed
      - Random seed
      - 42

Adam Mini Configuration (training.adam_mini)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Adam Mini Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable Adam-mini optimizer
      - true
    * - learning_rate
      - Learning rate (inherits from training)
      - ${training.learning_rate}
    * - weight_decay
      - Weight decay (inherits from training)
      - ${training.weight_decay}
    * - beta1
      - Adam beta1 parameter
      - 0.9
    * - beta2
      - Adam beta2 parameter
      - 0.999
    * - eps
      - Adam epsilon parameter
      - 1.0e-8
    * - dim
      - Hidden dimension (auto-detected)
      - null
    * - n_heads
      - Number of attention heads (auto-detected)
      - null
    * - n_kv_heads
      - Number of key-value heads (auto-detected)
      - null
    * - use_single_lr_for_values
      - Use single LR for values in small training runs
      - false

Memory Management Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Memory Management Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - aggressive_cleanup
      - Enable aggressive memory cleanup between phases
      - true
    * - log_memory_usage
      - Log memory usage at key points in training
      - true

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
      - "game"
    * - method: "classic"
      - Simple instruction/output format for standard fine-tuning
      - Expects "instruction" and "output" fields
    * - method: "game"
      - Complex game format with conversation history and actions
      - Expects "prompt" and "answer" fields with structured data

GRPO Configuration
~~~~~~~~~~~~~~~~~~

.. list-table:: GRPO Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - val_data
      - Validation data file
      - "test_ru.json"
    * - train_data
      - Training data file
      - "dataset_ru.json"
    * - max_completion_length
      - Maximum completion length
      - None
    * - num_generations
      - Number of generations
      - 2
    * - use_cache
      - Use cache during generation
      - false
    * - epsilon
      - KL penalty coefficient for policy gradient
      - 0.2
    * - beta
      - KL regularization parameter
      - 0.01
    * - loss_type
      - Loss function type ("sigmoid" or "hinge")
      - "sigmoid"
    * - temperature
      - Sampling temperature for generation
      - 0.7
    * - top_k
      - Top-k sampling parameter
      - 50
    * - top_p
      - Top-p (nucleus) sampling parameter
      - 0.95
    * - do_sample
      - Enable sampling during generation
      - true
    * - response_length
      - Maximum response length for completions
      - 256
    * - use_ref_model
      - Whether to use a separate reference model
      - false
    * - ref_model_name
      - Reference model name (if different from base model)
      - null

DPO Configuration
~~~~~~~~~~~~~~~~~

.. list-table:: DPO Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - val_data
      - Validation data file
      - "dpo_test.json"
    * - train_data
      - Training data file
      - "dpo_dataset.json"
    * - max_length
      - Maximum sequence length
      - 2048
    * - max_prompt_length
      - Maximum prompt length
      - 1024
    * - max_target_length
      - Maximum target length
      - 1024
    * - beta
      - KL penalty coefficient for DPO loss
      - 0.1
    * - loss_type
      - Loss function type ("sigmoid" or "hinge")
      - "sigmoid"
    * - use_ref_model
      - Whether to use a separate reference model for DPO
      - false
    * - ref_model_name
      - Reference model name (if different from base model)
      - null

Optuna Hyperparameter Optimization Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``optuna`` section provides configurable hyperparameter optimization with the ability to enable/disable individual parameters.

.. list-table:: Optuna Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable/disable Optuna hyperparameter optimization
      - false
    * - n_trials
      - Number of optimization trials
      - 10

**Hyperparameter Search Spaces:**

Each hyperparameter can be independently enabled or disabled:

.. list-table:: Learning Rate Configuration (optuna.learning_rate)
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable learning rate tuning
      - true
    * - min
      - Minimum learning rate value
      - 1e-6
    * - max
      - Maximum learning rate value
      - 5e-5
    * - log_scale
      - Use log scale for sampling
      - true

.. list-table:: Epochs Configuration (optuna.num_train_epochs)
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable epochs tuning
      - true
    * - min
      - Minimum number of epochs
      - 0.5
    * - max
      - Maximum number of epochs
      - 2.0

.. list-table:: Weight Decay Configuration (optuna.weight_decay)
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable weight decay tuning
      - true
    * - min
      - Minimum weight decay value
      - 0.0
    * - max
      - Maximum weight decay value
      - 0.3

.. list-table:: Warmup Steps Configuration (optuna.warmup_steps)
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - enabled
      - Enable warmup steps tuning
      - true
    * - min
      - Minimum warmup steps
      - 0
    * - max
      - Maximum warmup steps
      - 500

**Usage Examples:**

.. code-block:: bash

    # Enable Optuna with default search spaces
    python main.py optuna.enabled=true

    # Disable learning rate tuning, keep others enabled
    python main.py optuna.enabled=true optuna.learning_rate.enabled=false

    # Run with more trials
    python main.py optuna.enabled=true optuna.n_trials=50

    # Customize search space for learning rate
    python main.py optuna.enabled=true optuna.learning_rate.min=1e-5 optuna.learning_rate.max=1e-4

    # Disable all tuning except learning rate
    python main.py optuna.enabled=true optuna.num_train_epochs.enabled=false optuna.weight_decay.enabled=false optuna.warmup_steps.enabled=false

Paths Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Path Directories
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - data_dir
      - Input dataset directory
      - "data"
    * - test_data
      - Test data file
      - "test_ru.json"
    * - train_data
      - Train data file
      - "dataset_ru.json"
    * - gguf_directory
      - GGUF output directory
      - "custom-model"
    * - merged_model_path
      - Merged model path
      - "merged_model_fp16"
    * - output_dir
      - Trained model output directory
      - "models"
    * - llama_cpp_dir
      - Path to llama.cpp installation
      - "llama.cpp"
    * - venv_python_path
      - Python executable path (managed by environment configs)
      - "${paths.venv_python_path:}"
    * - final_weights_path
      - Final weights path
      - "models"
    * - quantized_path
      - Path to quantization executable
      - "llama-quantize.exe"

Other Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Other Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - cutoff_len
      - Cutoff length for sequences
      - 2048
    * - hf_login
      - Enable Hugging Face login
      - false

Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Environment Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - use_dotenv
      - Set to true for local development, false for Docker/production
      - false

Testing Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Testing Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - data_source_mode
      - Data source handling mode for train/validation split
      - "separate_validation"
    * - test_split_ratio
      - Train/test split ratio for auto_split mode
      - 0.05
    * - val_data_file
      - Validation file for separate_validation mode
      - "test_ru.json"
    * - manual_lmstudio_test
      - Use only for local run
      - false
    * - test
      - Enable testing
      - true
    * - test_dataset
      - Test dataset file path
      - "${paths.data_dir}/test_ru.json"
    * - output_test_file
      - Output path for processed test file
      - "../test.json"
    * - prompt_for_deepeval
      - Prompt for DeepEval
      - "test_prompt"

Data Source Mode Configuration (testing.data_source_mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``data_source_mode`` parameter controls how training and validation datasets are loaded and processed. This configuration eliminates code duplication and provides flexible data handling approaches.

**Available Modes:**

.. list-table:: Data Source Mode Options
    :widths: 25 35 40
    :header-rows: 1

    * - Mode
      - File Requirements
      - Description
    * - auto_split
      - Single training file
      - Automatically splits one dataset file into train/validation sets
    * - separate_validation
      - Training file + validation file
      - Uses separate files for training and validation data
    * - separate_files
      - Training file + test file
      - Uses completely separate train and test datasets

**Mode Details:**

**1. auto_split Mode**

Uses a single dataset file and automatically creates train/validation splits using scikit-learn's ``train_test_split``.

.. code-block:: yaml

    testing:
      data_source_mode: "auto_split"
      test_split_ratio: 0.05  # 5% for validation, 95% for training
    paths:
      train_data: "dataset_ru.json"  # Single file with all data

**Required Configuration:**

- ``paths.train_data``: Path to the single dataset file
- ``testing.test_split_ratio``: Fraction of data to reserve for validation (0.0-1.0)

**File Format:**

The dataset must contain ``examples`` key with list of training samples and optional ``system`` key.

.. code-block:: json

    {
      "system": "You are a helpful assistant",
      "examples": [
        {"user": "Hello", "bot": "Hi there!", "system": "Be friendly"},
        {"user": "How are you?", "bot": "I'm doing well!", "system": "Be positive"}
      ]
    }

**2. separate_validation Mode**

Uses one file for training and a separate file for validation. Ideal when you have a predefined validation set.

.. code-block:: yaml

    testing:
      data_source_mode: "separate_validation"
      val_data_file: "validation_set.json"
    paths:
      train_data: "training_set.json"

**Required Configuration:**

- ``paths.train_data``: Path to training dataset file
- ``testing.val_data_file``: Path to validation dataset file

**File Format:**

Both files must follow the same structure as ``auto_split`` mode with ``examples`` and optional ``system`` keys.

**3. separate_files Mode**

Uses completely separate training and test files. Most flexible option for custom dataset arrangements.

.. code-block:: yaml

    testing:
      data_source_mode: "separate_files"
    paths:
      train_data: "custom_train.json"
      test_data: "custom_test.json"

**Required Configuration:**

- ``paths.train_data``: Path to training dataset file
- ``paths.test_data``: Path to test dataset file

**File Format:**

Files can have any valid JSON structure - the most flexible mode that doesn't enforce ``examples`` key validation.

**Usage Examples:**

.. code-block:: bash

    # Use auto-split with custom ratio
    python main.py testing.data_source_mode=auto_split testing.test_split_ratio=0.1

    # Use separate validation file
    python main.py testing.data_source_mode=separate_validation testing.val_data_file=my_val.json

    # Use completely separate files
    python main.py testing.data_source_mode=separate_files paths.test_data=my_test.json

**Mode Comparison:**

.. list-table:: Data Source Mode Comparison
    :widths: 20 25 30 25
    :header-rows: 1

    * - Feature
      - auto_split
      - separate_validation
      - separate_files
    * - File Count
      - 1 file
      - 2 files
      - 2 files
    * - Split Control
      - Automatic (configurable ratio)
      - Manual (predefined files)
      - Manual (custom files)
    * - Structure Validation
      - Enforced (examples key required)
      - Enforced (examples key required)
      - Flexible (any JSON structure)
    * - Best For
      - Simple datasets, quick prototyping
      - Predefined validation sets
      - Complex custom datasets

**Implementation Notes:**

The ``data_source_mode`` implementation uses helper functions to eliminate code duplication:

- ``_load_json_file()``: Standardized JSON loading with error handling
- ``_validate_dataset_structure()``: Validates required dataset structure for modes that need it
- ``_process_auto_split_mode()``: Handles single-file splitting
- ``_process_separate_validation_mode()``: Handles train + validation files
- ``_process_separate_files_mode()``: Handles separate train/test files

This refactored approach reduces the main ``data_preparation()`` function from ~200 lines to ~50 lines while maintaining all functionality.

Logging Configuration
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Logging Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - log_level
      - Logging level ("DEBUG" or "INFO")
      - "INFO"
    * - logging_backend
      - Logging backend (wandb, mlflow, none)
      - "mlflow"

WandB Configuration (logging.wandb)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: WandB Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - project_name
      - WandB project name
      - "Gemma vika train"
    * - anonymous
      - Allow anonymous WandB usage
      - "allow"

MLflow Configuration (logging.mlflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: MLflow Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - experiment_name
      - MLflow experiment name
      - "vika-experiment"
    * - tracking_uri
      - MLflow tracking URI
      - "http://localhost:5000"
    * - log_artifacts
      - Log artifacts to MLflow
      - false

Hydra Configuration
~~~~~~~~~~~~~~~~~~~

.. list-table:: Hydra Parameters
    :widths: 25 50 25
    :header-rows: 1

    * - Parameter
      - Description
      - Default
    * - run.dir
      - Hydra run directory
      - "."
    * - job.chdir
      - Address future Hydra working dir change warning
      - true

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

Model Conversion and Output Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
      - false

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
            # Conditional GGUF conversion
            if cfg.model.quant.get("enabled", True):
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
      - r=8, alpha=16
    * - Modules to Save
      - lm_head, embed_tokens
      -

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

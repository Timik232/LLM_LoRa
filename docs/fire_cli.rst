Fire CLI Interface
==================

The Fire CLI provides a modern command-line interface for the LLM-LoRA framework, powered by Python Fire. This interface simplifies training, evaluation, and model conversion operations through intuitive commands.

Overview
--------

The Fire CLI interface offers:

* **Simple Command Structure**: Natural language-like commands using Python Fire
* **Automatic Help Generation**: Built-in help system with parameter documentation
* **Interactive Mode**: Step-by-step guidance for complex operations
* **Batch Processing**: Support for multiple model processing and batch training
* **Configuration Integration**: Seamless integration with Hydra configuration system
* **Progress Monitoring**: Real-time training and conversion progress tracking

Available Commands
------------------

Core Training Commands
~~~~~~~~~~~~~~~~~~~~~~

train_model
^^^^^^^^^^^

Train a model using the specified configuration and method.

.. code-block:: bash

    # Basic training with default configuration
    python main.py train_model

    # Training with specific configuration
    python main.py train_model --config-name=custom_config

    # Training with method override
    python main.py train_model --training.training_method=dpo

    # Training with custom parameters
    python main.py train_model --model.lora_r=32 --training.learning_rate=1e-4

**Parameters:**

* ``config_name`` (optional): Configuration file to use (default: "config")
* ``training_method`` (optional): Override training method (sft/dpo/grpo)
* ``output_dir`` (optional): Custom output directory for trained model
* ``resume_from_checkpoint`` (optional): Path to checkpoint for resuming training

evaluate_model
^^^^^^^^^^^^^^

Evaluate a trained model using specified metrics and test datasets.

.. code-block:: bash

    # Basic model evaluation
    python main.py evaluate_model --model_path=models/trained_model

    # Evaluation with specific test set
    python main.py evaluate_model --model_path=models/trained_model --test_data=data/test.json

    # Evaluation with custom metrics
    python main.py evaluate_model --model_path=models/trained_model --metrics=["bleu","rouge","perplexity"]

**Parameters:**

* ``model_path`` (required): Path to the trained model directory
* ``test_data`` (optional): Path to test dataset (default: from config)
* ``metrics`` (optional): List of evaluation metrics to compute
* ``output_file`` (optional): Path to save evaluation results

Model Conversion Commands
~~~~~~~~~~~~~~~~~~~~~~~~~

convert_to_gguf
^^^^^^^^^^^^^^^

Convert trained model to GGUF format for llama.cpp inference.

.. code-block:: bash

    # Basic GGUF conversion
    python main.py convert_to_gguf --model_path=models/trained_model

    # Conversion with quantization
    python main.py convert_to_gguf --model_path=models/trained_model --quantize=true --qtype=q4_1

    # Custom output path
    python main.py convert_to_gguf --model_path=models/trained_model --output_path=models/gguf/

**Parameters:**

* ``model_path`` (required): Path to the PyTorch model
* ``output_path`` (optional): Output directory for GGUF files
* ``quantize`` (optional): Apply quantization during conversion
* ``qtype`` (optional): Quantization type (q4_0, q4_1, q8_0, etc.)

convert_to_rkllm
^^^^^^^^^^^^^^^^

Convert model to RKLLM format for Rockchip NPU deployment.

.. code-block:: bash

    # Basic RKLLM conversion
    python main.py convert_to_rkllm --model_path=models/trained_model

    # Conversion for specific platform
    python main.py convert_to_rkllm --model_path=models/trained_model --platform=rk3588

    # With optimization settings
    python main.py convert_to_rkllm --model_path=models/trained_model --optimize=true --precision=int8

**Parameters:**

* ``model_path`` (required): Path to the model to convert
* ``platform`` (optional): Target Rockchip platform (rk3588, rk3576)
* ``optimize`` (optional): Enable optimization for NPU
* ``precision`` (optional): Inference precision (int8, int16, float16)

Utility Commands
~~~~~~~~~~~~~~~~

list_models
^^^^^^^^^^^

List available trained models and their information.

.. code-block:: bash

    # List all models
    python main.py list_models

    # List models with details
    python main.py list_models --detailed=true

    # Filter by training method
    python main.py list_models --filter_method=dpo

**Parameters:**

* ``detailed`` (optional): Show detailed model information
* ``filter_method`` (optional): Filter by training method
* ``output_format`` (optional): Output format (table, json, yaml)

clean_checkpoints
^^^^^^^^^^^^^^^^^

Clean up old checkpoints and temporary files.

.. code-block:: bash

    # Clean all old checkpoints
    python main.py clean_checkpoints

    # Clean checkpoints older than specific days
    python main.py clean_checkpoints --days=7

    # Dry run to see what would be deleted
    python main.py clean_checkpoints --dry_run=true

**Parameters:**

* ``days`` (optional): Remove checkpoints older than N days (default: 30)
* ``dry_run`` (optional): Show what would be deleted without actually deleting
* ``keep_best`` (optional): Keep best performing checkpoints

Configuration and Help
----------------------

Getting Help
~~~~~~~~~~~~

The Fire CLI provides comprehensive help information:

.. code-block:: bash

    # General help
    python main.py --help

    # Help for specific command
    python main.py train_model --help

    # Help for specific parameter
    python main.py train_model -- --help

    # Interactive help mode
    python main.py -- --help

Configuration Override
~~~~~~~~~~~~~~~~~~~~~~

Override configuration parameters directly from command line:

.. code-block:: bash

    # Override model parameters
    python main.py train_model model.lora_r=64 model.lora_alpha=128

    # Override training parameters
    python main.py train_model training.learning_rate=3e-5 training.max_seq_length=4096

    # Override paths
    python main.py train_model paths.data_dir=custom_data paths.output_dir=custom_output

Batch Operations
~~~~~~~~~~~~~~~~

Process multiple models or configurations:

.. code-block:: bash

    # Train multiple configurations
    python main.py train_model --config-name=config1,config2,config3

    # Convert multiple models
    python main.py convert_to_gguf --model_path=models/model1,models/model2

    # Evaluate multiple models
    python main.py evaluate_model --model_path=models/* --output_file=evaluation_results.json

Advanced Usage
--------------

Pipeline Operations
~~~~~~~~~~~~~~~~~~~

Chain multiple operations together:

.. code-block:: bash

    # Train, evaluate, and convert in sequence
    python main.py train_model && \\
    python main.py evaluate_model --model_path=models/latest && \\
    python main.py convert_to_gguf --model_path=models/latest

Custom Scripts Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use Fire CLI in custom scripts:

.. code-block:: python

    import fire
    from main import LLMLoRAFramework

    # Create framework instance
    framework = LLMLoRAFramework()

    # Use Fire to expose all methods
    if __name__ == '__main__':
        fire.Fire(framework)

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

Configure behavior using environment variables:

.. code-block:: bash

    # Set default configuration
    export LLMLORA_CONFIG=production_config

    # Set output directory
    export LLMLORA_OUTPUT_DIR=models/production

    # Enable verbose logging
    export LLMLORA_VERBOSE=true

    # Run with environment variables
    python main.py train_model

Common Usage Patterns
----------------------

Development Workflow
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # 1. Train a model
    python main.py train_model --config-name=dev_config

    # 2. Evaluate the model
    python main.py evaluate_model --model_path=models/latest

    # 3. Convert for deployment
    python main.py convert_to_gguf --model_path=models/latest --quantize=true

Production Deployment
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # 1. Train with production settings
    python main.py train_model --config-name=production

    # 2. Thorough evaluation
    python main.py evaluate_model --model_path=models/production --metrics=all

    # 3. Multi-format conversion
    python main.py convert_to_gguf --model_path=models/production
    python main.py convert_to_rkllm --model_path=models/production

Experimentation
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Quick hyperparameter testing
    python main.py train_model training.learning_rate=1e-5,2e-5,3e-5

    # Method comparison
    python main.py train_model training.training_method=sft,dpo,grpo

    # Architecture experiments
    python main.py train_model model.lora_r=16,32,64

Error Handling
--------------

The Fire CLI provides detailed error messages and suggestions:

* **Configuration errors**: Missing or invalid configuration parameters
* **Model errors**: Issues with model loading or architecture
* **Data errors**: Problems with dataset format or accessibility
* **Conversion errors**: Issues with model format conversion
* **Resource errors**: Insufficient memory or disk space

For debugging, enable verbose mode:

.. code-block:: bash

    python main.py train_model --verbose=true --debug=true

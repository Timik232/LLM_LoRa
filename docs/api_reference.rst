API Reference
=============

Complete API reference for the LLM-LoRA framework modules, classes, and functions. This reference provides detailed documentation for all public interfaces and utilities.

Core Training Modules
----------------------

training_model.one_file_train
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main training orchestrator providing the complete training pipeline.

.. automodule:: training_model.one_file_train
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: training_model.one_file_train.train
.. autofunction:: training_model.one_file_train.main
.. autofunction:: training_model.one_file_train.setup_logging
.. autofunction:: training_model.one_file_train.load_model_and_tokenizer

training_model.dpo_train
~~~~~~~~~~~~~~~~~~~~~~~~~

Direct Preference Optimization training implementation.

.. automodule:: training_model.dpo_train
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: training_model.dpo_train.DPOTrainer
   :members:
   :undoc-members:

.. autoclass:: training_model.dpo_train.DPOConfig
   :members:
   :undoc-members:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: training_model.dpo_train.dpo_train
.. autofunction:: training_model.dpo_train.compute_dpo_loss
.. autofunction:: training_model.dpo_train.prepare_preference_data

training_model.grpo_train
~~~~~~~~~~~~~~~~~~~~~~~~~

Group Relative Policy Optimization training implementation.

.. automodule:: training_model.grpo_train
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: training_model.grpo_train.GRPOTrainer
   :members:
   :undoc-members:

.. autoclass:: training_model.grpo_train.GRPOConfig
   :members:
   :undoc-members:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: training_model.grpo_train.grpo_train
.. autofunction:: training_model.grpo_train.compute_grpo_loss
.. autofunction:: training_model.grpo_train.prepare_group_data

Utility Modules
---------------

training_model.utils
~~~~~~~~~~~~~~~~~~~~~

Core utility functions for model operations and data processing.

.. automodule:: training_model.utils
   :members:
   :undoc-members:
   :show-inheritance:

Model Operations
^^^^^^^^^^^^^^^^

.. autofunction:: training_model.utils.load_model_with_quantization
.. autofunction:: training_model.utils.setup_lora_config
.. autofunction:: training_model.utils.merge_lora_weights
.. autofunction:: training_model.utils.save_model_safely

Data Processing
^^^^^^^^^^^^^^^

.. autofunction:: training_model.utils.prepare_chat_dataset
.. autofunction:: training_model.utils.tokenize_dataset
.. autofunction:: training_model.utils.format_chat_prompt

Model Conversion
^^^^^^^^^^^^^^^^

.. autofunction:: training_model.utils.convert_to_gguf
.. autofunction:: training_model.utils.convert_to_rkllm
.. autofunction:: training_model.utils.quantize_model

training_model.vika_utils
~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized utilities for Vikhr model variants.

.. automodule:: training_model.vika_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: training_model.vika_utils.load_vika_model
.. autofunction:: training_model.vika_utils.setup_vika_tokenizer
.. autofunction:: training_model.vika_utils.vika_chat_format

Configuration and Logging
--------------------------

training_model.logging_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Logging configuration and setup utilities.

.. automodule:: training_model.logging_config
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: training_model.logging_config.setup_logging
.. autofunction:: training_model.logging_config.configure_transformers_logging
.. autofunction:: training_model.logging_config.setup_wandb_logging

training_model.optuna
~~~~~~~~~~~~~~~~~~~~~

Hyperparameter optimization using Optuna.

.. automodule:: training_model.optuna
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: training_model.optuna.OptunaOptimizer
   :members:
   :undoc-members:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: training_model.optuna.optimize_hyperparameters
.. autofunction:: training_model.optuna.create_study
.. autofunction:: training_model.optuna.objective_function

Evaluation Modules
------------------

evaluation.model_evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

General model evaluation utilities and metrics.

.. automodule:: evaluation.model_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: evaluation.model_evaluation.ModelEvaluator
   :members:
   :undoc-members:

.. autoclass:: evaluation.model_evaluation.EvaluationConfig
   :members:
   :undoc-members:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: evaluation.model_evaluation.evaluate_model
.. autofunction:: evaluation.model_evaluation.compute_perplexity
.. autofunction:: evaluation.model_evaluation.compute_bleu_score
.. autofunction:: evaluation.model_evaluation.compute_rouge_score

evaluation.deepeval_integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DeepEval framework integration for advanced evaluation.

.. automodule:: evaluation.deepeval_integration
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: evaluation.deepeval_integration.DeepEvalMetrics
   :members:
   :undoc-members:

.. autoclass:: evaluation.deepeval_integration.FaithfulnessMetric
   :members:
   :undoc-members:

.. autoclass:: evaluation.deepeval_integration.BiasMetric
   :members:
   :undoc-members:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: evaluation.deepeval_integration.run_deepeval
.. autofunction:: evaluation.deepeval_integration.setup_deepeval_metrics
.. autofunction:: evaluation.deepeval_integration.generate_evaluation_report

evaluation.game_evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Specialized evaluation for game-based conversational AI.

.. automodule:: evaluation.game_evaluation
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: evaluation.game_evaluation.GameEvaluator
   :members:
   :undoc-members:

.. autoclass:: evaluation.game_evaluation.DialogueCoherenceMetric
   :members:
   :undoc-members:

Key Functions
^^^^^^^^^^^^^

.. autofunction:: evaluation.game_evaluation.evaluate_game_model
.. autofunction:: evaluation.game_evaluation.compute_coherence_score
.. autofunction:: evaluation.game_evaluation.evaluate_character_consistency

Fire CLI Interface
------------------

Main CLI Controller
~~~~~~~~~~~~~~~~~~~

.. automodule:: main
   :members:
   :undoc-members:
   :show-inheritance:

Key Classes
^^^^^^^^^^^

.. autoclass:: main.LLMLoRAFramework
   :members:
   :undoc-members:

CLI Methods
^^^^^^^^^^^

.. automethod:: main.LLMLoRAFramework.train_model
.. automethod:: main.LLMLoRAFramework.evaluate_model
.. automethod:: main.LLMLoRAFramework.convert_to_gguf
.. automethod:: main.LLMLoRAFramework.convert_to_rkllm
.. automethod:: main.LLMLoRAFramework.list_models
.. automethod:: main.LLMLoRAFramework.clean_checkpoints

Configuration Classes
---------------------

Model Configuration
~~~~~~~~~~~~~~~~~~~

.. autoclass:: training_model.config.ModelConfig
   :members:
   :undoc-members:

Training Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: training_model.config.TrainingConfig
   :members:
   :undoc-members:

Paths Configuration
~~~~~~~~~~~~~~~~~~~

.. autoclass:: training_model.config.PathsConfig
   :members:
   :undoc-members:

Conversion Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: training_model.config.ConversionConfig
   :members:
   :undoc-members:

Data Classes and Types
----------------------

Training Data Types
~~~~~~~~~~~~~~~~~~~

.. autoclass:: training_model.types.InstructionData
   :members:
   :undoc-members:

.. autoclass:: training_model.types.PreferenceData
   :members:
   :undoc-members:

.. autoclass:: training_model.types.GroupPreferenceData
   :members:
   :undoc-members:

Evaluation Data Types
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evaluation.types.EvaluationResult
   :members:
   :undoc-members:

.. autoclass:: evaluation.types.MetricResult
   :members:
   :undoc-members:

.. autoclass:: evaluation.types.TestCase
   :members:
   :undoc-members:

Model Types
~~~~~~~~~~~

.. autoclass:: training_model.types.ModelOutput
   :members:
   :undoc-members:

.. autoclass:: training_model.types.TrainingMetrics
   :members:
   :undoc-members:

Constants and Enums
-------------------

Training Constants
~~~~~~~~~~~~~~~~~~

.. automodule:: training_model.constants
   :members:
   :undoc-members:

.. data:: training_model.constants.SUPPORTED_MODELS

   List of supported base model architectures.

.. data:: training_model.constants.DEFAULT_LORA_CONFIG

   Default LoRA configuration parameters.

.. data:: training_model.constants.QUANTIZATION_TYPES

   Supported quantization types for GGUF conversion.

Training Methods Enum
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: training_model.enums.TrainingMethod
   :members:
   :undoc-members:

.. autoclass:: training_model.enums.QuantizationType
   :members:
   :undoc-members:

.. autoclass:: training_model.enums.ModelFormat
   :members:
   :undoc-members:

Evaluation Enums
~~~~~~~~~~~~~~~~

.. autoclass:: evaluation.enums.EvaluationMetric
   :members:
   :undoc-members:

.. autoclass:: evaluation.enums.EvaluationType
   :members:
   :undoc-members:

Exception Classes
-----------------

Training Exceptions
~~~~~~~~~~~~~~~~~~~

.. autoexception:: training_model.exceptions.TrainingError
.. autoexception:: training_model.exceptions.ModelLoadError
.. autoexception:: training_model.exceptions.DataPreparationError
.. autoexception:: training_model.exceptions.ConversionError

Evaluation Exceptions
~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: evaluation.exceptions.EvaluationError
.. autoexception:: evaluation.exceptions.MetricComputationError
.. autoexception:: evaluation.exceptions.TestCaseError

Configuration Exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: training_model.exceptions.ConfigurationError
.. autoexception:: training_model.exceptions.InvalidParameterError

Helper Functions
----------------

File System Utilities
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: training_model.file_utils
   :members:
   :undoc-members:

.. autofunction:: training_model.file_utils.ensure_directory
.. autofunction:: training_model.file_utils.copy_model_files
.. autofunction:: training_model.file_utils.cleanup_temp_files

GPU Utilities
~~~~~~~~~~~~~

.. automodule:: training_model.gpu_utils
   :members:
   :undoc-members:

.. autofunction:: training_model.gpu_utils.get_gpu_info
.. autofunction:: training_model.gpu_utils.clear_gpu_cache
.. autofunction:: training_model.gpu_utils.optimize_gpu_memory

Memory Management
~~~~~~~~~~~~~~~~~

.. automodule:: training_model.memory_utils
   :members:
   :undoc-members:

.. autofunction:: training_model.memory_utils.monitor_memory_usage
.. autofunction:: training_model.memory_utils.optimize_batch_size
.. autofunction:: training_model.memory_utils.clear_memory_cache

Version Information
-------------------

Framework Version
~~~~~~~~~~~~~~~~~

.. autodata:: training_model.__version__

   Current version of the LLM-LoRA framework.

.. autodata:: training_model.__author__

   Framework author information.

.. autodata:: training_model.__description__

   Framework description.

Dependency Versions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: training_model.version.get_dependency_versions

   Get versions of key dependencies.

.. autofunction:: training_model.version.check_compatibility

   Check compatibility with current environment.

Usage Examples
--------------

Basic Training Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from training_model.one_file_train import train
    from omegaconf import OmegaConf

    # Load configuration
    config = OmegaConf.load("conf/config.yaml")

    # Run training
    results = train(config)

    print(f"Training completed. Model saved to: {results.model_path}")

DPO Training Example
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from training_model.dpo_train import dpo_train
    from training_model.utils import load_preference_dataset

    # Load preference data
    dataset = load_preference_dataset("data/preferences.json")

    # Configure DPO training
    config = {
        "model_name": "models/sft_base",
        "beta": 0.1,
        "learning_rate": 1e-5
    }

    # Run DPO training
    model = dpo_train(config, dataset)

Model Evaluation Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from evaluation.model_evaluation import ModelEvaluator

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path="models/trained_model",
        metrics=["bleu", "rouge", "perplexity"]
    )

    # Run evaluation
    results = evaluator.evaluate(test_dataset="data/test.json")

    # Print results
    for metric, score in results.items():
        print(f"{metric}: {score}")

Model Conversion Example
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from training_model.utils import convert_to_gguf, convert_to_rkllm

    # Convert to GGUF
    gguf_path = convert_to_gguf(
        model_path="models/trained_model",
        output_path="models/gguf/",
        quantization_type="q4_1"
    )

    # Convert to RKLLM
    rkllm_path = convert_to_rkllm(
        model_path="models/trained_model",
        platform="rk3588",
        optimization_level=2
    )

Fire CLI Usage
~~~~~~~~~~~~~~

.. code-block:: python

    import fire
    from main import LLMLoRAFramework

    # Use Fire CLI programmatically
    framework = LLMLoRAFramework()

    # Train model
    framework.train_model(config_name="custom_config")

    # Evaluate model
    results = framework.evaluate_model(model_path="models/latest")

    # Convert model
    framework.convert_to_gguf(model_path="models/latest")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from omegaconf import OmegaConf
    from training_model.config import ModelConfig, TrainingConfig

    # Create advanced configuration
    config = OmegaConf.create({
        "model": {
            "model_name": "Vikhrmodels/Vikhr-YandexGPT-5-Lite-8B-it",
            "lora_r": 64,
            "lora_alpha": 128,
            "quantization": True
        },
        "training": {
            "training_method": "grpo",
            "batch_size": 2,
            "learning_rate": 5e-6,
            "epochs": 1
        },
        "conversion": {
            "convert_to_gguf": True,
            "convert_to_rkllm": True,
            "quantization_type": "q4_1"
        }
    })

    # Run training with advanced config
    results = train(config)

Custom Metrics Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from evaluation.model_evaluation import ModelEvaluator
    from evaluation.types import MetricResult

    class CustomMetric:
        def compute(self, predictions, references):
            # Implement custom metric logic
            score = custom_metric_computation(predictions, references)
            return MetricResult(name="custom_metric", score=score)

    # Use custom metric
    evaluator = ModelEvaluator(model_path="models/trained_model")
    evaluator.add_metric(CustomMetric())

    results = evaluator.evaluate(test_dataset="data/test.json")

Troubleshooting Reference
-------------------------

Common Error Codes
~~~~~~~~~~~~~~~~~~~

.. list-table:: Error Codes
    :widths: 15 30 55
    :header-rows: 1

    * - Code
      - Error Type
      - Description
    * - E001
      - Model Loading Error
      - Failed to load base model or checkpoint
    * - E002
      - Data Format Error
      - Invalid dataset format or structure
    * - E003
      - Configuration Error
      - Invalid or missing configuration parameters
    * - E004
      - Memory Error
      - Insufficient GPU or system memory
    * - E005
      - Conversion Error
      - Model format conversion failure
    * - E006
      - Evaluation Error
      - Error during model evaluation
    * - E007
      - Dependency Error
      - Missing or incompatible dependencies

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: training_model.optimization.optimize_training_performance
.. autofunction:: training_model.optimization.optimize_inference_performance
.. autofunction:: training_model.optimization.auto_tune_hyperparameters

Debugging Utilities
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: training_model.debug.enable_debug_mode
.. autofunction:: training_model.debug.print_model_info
.. autofunction:: training_model.debug.trace_memory_usage

Migration Guide
---------------

Version Migration
~~~~~~~~~~~~~~~~~

.. autofunction:: training_model.migration.migrate_config_v1_to_v2
.. autofunction:: training_model.migration.update_model_format
.. autofunction:: training_model.migration.check_compatibility

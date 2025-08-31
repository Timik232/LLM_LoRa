training\_model package
=======================

The training_model package contains the core training functionality for the LLM-LoRA framework. This package provides multiple training approaches including supervised fine-tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO).

Overview
--------

The training pipeline supports:

* **Supervised Fine-tuning (SFT)**: Traditional language model fine-tuning using instruction-response pairs
* **Direct Preference Optimization (DPO)**: Training models to align with human preferences without reinforcement learning
* **Group Relative Policy Optimization (GRPO)**: Advanced preference optimization for improved model alignment
* **LoRA Integration**: Parameter-efficient fine-tuning using Low-Rank Adaptation
* **Quantization Support**: 4-bit and 8-bit training quantization via BitsAndBytes
* **Model Conversion**: Automatic conversion to GGUF and RKLLM formats
* **Fire CLI Integration**: Command-line interface for easy training execution

Training Workflow
-----------------

1. **Configuration Loading**: Hydra-based configuration management
2. **Model Initialization**: Load base model with quantization settings
3. **Data Preparation**: Process datasets into chat format with proper tokenization
4. **LoRA Setup**: Configure parameter-efficient fine-tuning adapters
5. **Training Execution**: Run training with chosen method (SFT/DPO/GRPO)
6. **Model Merging**: Merge LoRA adapters with base model
7. **Format Conversion**: Convert to GGUF/RKLLM formats as needed
8. **Quantization**: Apply post-training quantization

Submodules
----------

training\_model.one\_file\_train module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Main training orchestrator providing the complete training pipeline from data loading to model conversion.

.. automodule:: training_model.one_file_train
   :members:
   :show-inheritance:
   :undoc-members:

training\_model.dpo\_train module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Direct Preference Optimization training implementation for preference-based model alignment.

.. automodule:: training_model.dpo_train
   :members:
   :show-inheritance:
   :undoc-members:

training\_model.grpo\_train module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Group Relative Policy Optimization training for advanced preference learning.

.. automodule:: training_model.grpo_train
   :members:
   :show-inheritance:
   :undoc-members:

training\_model.logging\_config module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Logging configuration and setup for training monitoring and debugging.

.. automodule:: training_model.logging_config
   :members:
   :show-inheritance:
   :undoc-members:

training\_model.optuna module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hyperparameter optimization using Optuna for automated training parameter tuning.

.. automodule:: training_model.optuna
   :members:
   :show-inheritance:
   :undoc-members:

training\_model.auth\_utils module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Authentication and logging utility functions for external services.

.. automodule:: training_model.auth_utils
   :members:
   :show-inheritance:
   :undoc-members:

training\_model.data\_preparation module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Data preparation utilities for different dataset formats including classic and game formats.

.. automodule:: training_model.data_preparation
   :members:
   :show-inheritance:
   :undoc-members:

Module contents
---------------

.. automodule:: training_model
   :members:
   :show-inheritance:
   :undoc-members:

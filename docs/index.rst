.. llm-lora documentation master file

LLM-LoRA: Advanced Language Model Fine-tuning Framework
=======================================================

LLM-LoRA is a comprehensive framework for fine-tuning Large Language Models using Low-Rank Adaptation (LoRA) techniques. This toolkit provides state-of-the-art training methods, evaluation capabilities, and deployment tools for creating specialized language models.

🚀 **Key Features**
===================

* **Advanced Training Methods**: Support for Supervised Fine-tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO)
* **LoRA Integration**: Efficient parameter-efficient fine-tuning using PEFT library
* **Multi-Platform Support**: Train on GPU and deploy to various platforms including Rockchip NPU via RKLLM conversion
* **Fire CLI Interface**: Modern command-line interface powered by Python Fire for easy interaction
* **Comprehensive Evaluation**: Built-in evaluation framework with DeepEval integration
* **Model Conversion Pipeline**: Seamless conversion to GGUF format and quantization support
* **Docker Deployment**: Container-based deployment with Ollama integration
* **Hydra Configuration**: Flexible configuration management with YAML-based setup

📋 **Supported Models**
=======================

* **Base Models**: Vikhr-YandexGPT, LLaMA variants, and other Hugging Face compatible models
* **Quantization**: 4-bit/8-bit training quantization and post-training GGUF quantization
* **Formats**: Support for PyTorch, GGUF, and RKLLM model formats

⚡ **Quick Start**
==================

1. **Installation**::

    pip install -r requirements.txt or poetry install --no-root

2. **Basic Training**::

    python main.py train --config-name=config

3. **Using Fire CLI**::

    python main.py --help
    python main.py train_model
    python main.py evaluate_model

4. **Model Conversion**::

    python main.py convert_to_gguf
    python main.py convert_to_rkllm

🏗️ **Architecture Overview**
=============================

The framework is organized into several key components:

* **training_model/**: Core training logic, utilities, and training method implementations
* **evaluation/**: Model evaluation and testing framework with unit and integration tests
* **conf/**: Hydra configuration files for different training scenarios
* **data/**: Training datasets and data processing utilities
* **models/**: Output directory for trained models and checkpoints

📚 **Documentation Sections**
==============================

.. toctree::
   :maxdepth: 2
   :caption: Configuration:

   hydra_config

.. toctree::
   :maxdepth: 2
   :caption: Training & Methods:

   training_model
   training_methods
   fire_cli

.. toctree::
   :maxdepth: 2
   :caption: Evaluation & Testing:

   evaluation

.. toctree::
   :maxdepth: 2
   :caption: Deployment & Conversion:

   rkllm_conversion
   docker_deployment

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api_reference

🔧 **Requirements**
===================

* Python 3.8+
* PyTorch 2.0+
* CUDA-compatible GPU (recommended)
* 16GB+ RAM for training
* llama.cpp for GGUF conversion
* RKLLM toolkit for Rockchip NPU deployment (optional)

📄 **License & Contributing**
=============================

This project is developed by Timur Komolov. For issues, feature requests, or contributions, please refer to the project repository.

<div align="center">

# 🚀 LLM-LoRa

**Memory-efficient LLM fine-tuning with LoRa and advanced quantization**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-supported-blue.svg)](docker-compose.yaml)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.53.0+-yellow.svg)](https://huggingface.co/transformers/)

[Quick Start](#quick-start) • [Features](#features) • [Installation](#installation) • [Documentation](#documentation) • [Examples](#examples)

</div>

## 🎯 Overview

**LLM-LoRa** is a comprehensive framework for memory-efficient fine-tuning of Large Language Models using LoRa (Low-Rank Adaptation) and advanced quantization techniques. Train state-of-the-art LLMs with significantly reduced memory requirements while maintaining performance.

### ✨ Key Benefits

- **🧠 Memory Efficient**: Train large models with up to 70% less GPU memory using LoRa and quantization
- **⚡ Multiple Methods**: Support for SFT (Supervised Fine-Tuning) and GRPO (DeepSeek's method) training
- **🔧 Advanced Quantization**: Built-in support for bitsandbytes quantization (4-bit, 8-bit)
- **🐳 Production Ready**: Full Docker support with GPU acceleration and Ollama integration
- **📊 Experiment Tracking**: Integrated MLflow and Weights & Biases logging
- **⚙️ Flexible Configuration**: Hydra-based configuration management

## 🚀 Quick Start

Get started in under 2 minutes:

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/timik232/LLM_LoRa.git
cd LLM_LoRa

# Configure training parameters
cp conf/config.yaml my_config.yaml
# Edit my_config.yaml with your settings

# Start training
docker-compose up
```

### Using Poetry

```bash
# Install dependencies
poetry install --no-root

# Start training
python main.py
```

### Quick Training Example

```python
# Basic training with default settings
python main.py model.model_name=microsoft/DialoGPT-small

# Custom configuration
python main.py \
  model.model_name=microsoft/DialoGPT-medium \
  model.train_steps=1000 \
  training.learning_rate=2e-5 \
  training.use_grpo=true
```

## ✨ Features

<table>
<tr>
<td width="50%">

### 🧠 **Memory Optimization**
- LoRa (Low-Rank Adaptation) fine-tuning
- 4-bit and 8-bit quantization
- Gradient checkpointing
- Memory-efficient attention patterns

### ⚡ **Training Methods**
- **SFT**: Supervised Fine-Tuning
- **GRPO**: Generalized Reward-based Policy Optimization
- Custom loss functions and optimizers
- Mixed precision training

</td>
<td width="50%">

### 🛠️ **Production Features**
- Docker containerization with GPU support
- Ollama model serving integration
- Hydra configuration management
- DVC for data and model versioning

### 📊 **Monitoring & Evaluation**
- MLflow experiment tracking
- Weights & Biases integration
- DeepEval automated evaluation
- Custom metrics and logging

</td>
</tr>
</table>

### 🎯 **Supported Model Formats**
- **LoRa Adapters**: Efficient fine-tuned weights
- **Merged Models**: Full model with LoRa weights integrated
- **GGUF**: Quantized format for efficient inference
- **RKLLM**: Rockchip NPU format for edge deployment

## 📦 Installation

### Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: Minimum 8GB GPU memory (varies by model size)
- **Storage**: 70GB+ for full Docker setup
- **Python**: 3.11 to 3.13

### Option 1: Docker (Recommended)

```bash
# Install NVIDIA Container Toolkit (if not already installed)
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/

# Build and run
docker-compose build
docker-compose up
```

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/timik232/LLM_LoRa.git
cd LLM_LoRa

# Install Poetry (if not installed)
pip install poetry

# Install dependencies
poetry install --no-root

# Verify installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 3: pip (Basic)

```bash
# Install from requirements
pip install -r requirements.txt
```

## 🔧 Configuration

The project uses Hydra for configuration management. Key configuration file:

- `conf/config.yaml`: Main configuration

### Example Configuration

```yaml
# conf/config.yaml
model:
  model_name: "microsoft/DialoGPT-medium"
  train_steps: 1000
  lora:
    r: 16
    alpha: 32
    dropout: 0.1
  quantization:
    load_in_4bit: true
    bnb_4bit_compute_dtype: "float16"

training:
  learning_rate: 2e-5
  batch_size: 4
  gradient_accumulation_steps: 4
  use_grpo: false

paths:
  train_data: "data/train.json"
  output_dir: "models/output"
```

## 📖 Usage Examples

### Basic Training

```bash
# Train with default settings
python main.py

# Train specific model
python main.py model.model_name=microsoft/DialoGPT-large

# Enable GRPO training
python main.py training.use_grpo=true
```

### Advanced Training

```bash
# Full pipeline with custom settings
python main.py \
  model.model_name=microsoft/DialoGPT-medium \
  model.train_steps=2000 \
  training.learning_rate=1e-4 \
  training.batch_size=8 \
  model.lora.r=32 \
  logging.use_mlflow=true
```

### Model Serving

```bash
# After training, serve with Ollama
docker-compose up ollama

# Test the model
curl -X POST http://localhost:11434/api/generate \
  -d '{"model": "custom-model", "prompt": "Hello, how are you?"}'
```

### Evaluation

```bash
# Run evaluation suite
python -m testing_model

# Custom evaluation
python -m testing_model --config-name=eval_config
```

## 🏗️ Project Structure

```
├── main.py                     # Main entry point
├── conf/                       # Configuration files
│   ├── config.yaml            # Main config
├── training_model/            # Training logic
│   ├── one_file_train.py     # Core training
│   ├── grpo_train.py         # GRPO implementation
│   └── data_preparation.py   # Data preprocessing
├── testing_model/            # Evaluation logic
│   ├── test.py              # Test implementations
│   └── deepeval_func.py     # DeepEval integration
├── data/                     # Training datasets
├── models/                   # Output models
├── docs/                     # Documentation
└── docker-compose.yaml      # Docker configuration
```

## 📊 Performance

### Supported Models

- **Any HuggingFace transformer** model

## 📚 Documentation

- **[Full Documentation](docs/)** - Comprehensive guides and API reference
- **[Training Methods](docs/training_methods.rst)** - SFT vs GRPO comparison
- **[Configuration Guide](docs/hydra_config.rst)** - Detailed config options
- **[Docker Deployment](docs/docker_deployment.rst)** - Production deployment
- **[API Reference](docs/api_reference.rst)** - Complete API documentation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup
git clone https://github.com/timik232/LLM_LoRa.git
cd LLM_LoRa

# Install with dev dependencies
poetry install

# Setup pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size or enable gradient accumulation
python main.py training.batch_size=2 training.gradient_accumulation_steps=8
```

**Docker GPU Issues**
```bash
# Verify NVIDIA container toolkit
nvidia-container-cli info
```

**Model Loading Errors**
- Ensure sufficient disk space (70GB+ for full setup)
- Check HuggingFace token for private models
- Verify CUDA compatibility

For more issues, check our [Issues](https://github.com/timik232/LLM_LoRa/issues) page.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **HuggingFace Transformers** for the transformer implementations
- **Microsoft PEFT** for LoRa implementation
- **bitsandbytes** for quantization support
- **TRL** for reinforcement learning methods
- The open-source ML community

## 📈 Roadmap

- [ ] Support for more model architectures (Mamba, RetNet)
- [ ] Multi-GPU training support
- [ ] Web UI for training management
- [ ] Advanced evaluation metrics
- [ ] Model compression techniques
- [ ] Edge deployment optimization

---

<div align="center">

**Star ⭐ this repository if it helped you!**

[Report Bug](https://github.com/timik232/LLM_LoRa/issues) • [Request Feature](https://github.com/timik232/LLM_LoRa/issues) • [Discussions](https://github.com/timik232/LLM_LoRa/discussions)

</div>
